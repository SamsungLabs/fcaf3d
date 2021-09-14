import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import reduce_mean, build_assigner
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import Scale, bias_init_with_prob

from mmdet3d.core.post_processing import aligned_3d_nms, box3d_multiclass_nms


class SparseNeckWithHead(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_convs,
                 n_reg_outs,
                 voxel_size,
                 pts_threshold,
                 assigner,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pts_threshold = pts_threshold
        self._init_layers(in_channels, out_channels, n_convs, n_reg_outs, n_classes)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, in_channels, out_channels, n_convs, n_reg_outs, n_classes):
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

        # head layers
        self.cls_convs = nn.Sequential(*[
            self._make_block(out_channels, out_channels)
            for _ in range(n_convs)
        ])
        self.reg_convs = nn.Sequential(*[
            self._make_block(out_channels, out_channels)
            for _ in range(n_convs)
        ])
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        for module in self.cls_convs.modules():
            if type(module) == ME.MinkowskiConvolution:
                nn.init.normal_(module.kernel, std=.01)
        for module in self.reg_convs.modules():
            if type(module) == ME.MinkowskiConvolution:
                nn.init.normal_(module.kernel, std=.01)
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def forward(self, x):
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, scores)

            out = self.__getattr__(f'out_block_{i}')(x)
            out = self.forward_single(out, self.scales[i])
            scores = out[-1]
            outs.append(out[:-1])
        return zip(*outs[::-1])

    def _prune(self, x, scores):
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()  # todo: [:, 1:] / 2 ?
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros((len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points,
             gt_bboxes,
             gt_labels,
             img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        loss_centerness, loss_bbox, loss_cls = [], [], []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        )

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        with torch.no_grad():
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets),
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        return loss_centerness, loss_bbox, loss_cls

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    # per scale
    def forward_single(self, x, scale):
        raise NotImplementedError

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        raise NotImplementedError

    def _bbox_to_loss(self, bbox):
        raise NotImplementedError

    def _nms(self, bboxes, scores, img_meta):
        raise NotImplementedError


@HEADS.register_module()
class ScanNetSparseNeckWithHead(SparseNeckWithHead):
    def forward_single(self, x, scale):
        cls = self.cls_convs(x)
        reg = self.reg_convs(x)
        centerness = self.centerness_conv(reg).features
        bbox_pred = torch.exp(scale(self.reg_conv(reg).features))
        scores = self.cls_conv(cls)
        cls_score = scores.features
        scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            # todo: do we need + .5?
            points[i] = points[i] * self.voxel_size

        return centernesses, bbox_preds, cls_scores, points, scores

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max ->
        # x, y, z, w, l, h
        return torch.stack([
            points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
            points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
            points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

    @staticmethod
    def _aligned_bbox_to_limits(bbox):
        # x, y, z, w, l, h ->
        # x_min, y_min, z_min, x_max, y_max, z_max
        return torch.stack((
            bbox[:, 0] - bbox[:, 3] / 2.,
            bbox[:, 1] - bbox[:, 4] / 2.,
            bbox[:, 2] - bbox[:, 5] / 2.,
            bbox[:, 0] + bbox[:, 3] / 2.,
            bbox[:, 1] + bbox[:, 4] / 2.,
            bbox[:, 2] + bbox[:, 5] / 2.,
        ), dim=1)

    def _bbox_to_loss(self, bbox):
        return self._aligned_bbox_to_limits(bbox)

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        bboxes = self._aligned_bbox_to_limits(bboxes)
        ids = aligned_3d_nms(bboxes, scores, labels, self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        # x_min, y_min, z_min, x_max, y_max, z_max ->
        # x, y, z, w, l, h
        bboxes = torch.stack((
            (bboxes[:, 0] + bboxes[:, 3]) / 2.,
            (bboxes[:, 1] + bboxes[:, 4]) / 2.,
            (bboxes[:, 2] + bboxes[:, 5]) / 2.,
            bboxes[:, 3] - bboxes[:, 0],
            bboxes[:, 4] - bboxes[:, 1],
            bboxes[:, 5] - bboxes[:, 2]
        ), dim=1)
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5), box_dim=6, with_yaw=False)
        return bboxes, scores[ids], labels[ids]


@HEADS.register_module()
class SunRgbdSparseNeckWithHead(SparseNeckWithHead):
    def forward_single(self, x, scale):
        cls = self.cls_convs(x)
        reg = self.reg_convs(x)
        centerness = self.centerness_conv(reg).features
        scores = self.cls_conv(cls)
        cls_score = scores.features
        scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.reg_conv(reg).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return centernesses, bbox_preds, cls_scores, points, scores

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, sin(2a)ln(q), cos(2a)ln(q) ->
        # x_center, y_center, z_center, w, l, h, alpha
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack((
            points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
            points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
            points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2,
            scale / (1 + q),
            scale * q / (1 + q),
            bbox_pred[:, 5] + bbox_pred[:, 4],
            alpha
        ), dim=-1)

    @staticmethod
    def _bbox_to_loss(bbox):
        return bbox

    def _nms(self, bboxes, scores, img_meta):
        # Add a dummy background class to the end. Nms needs to be fixed in the future.
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
        # x, y, z, w, l, h, alpha ->
        # x_min, y_min, x_max, y_max, alpha
        bboxes_for_nms = torch.stack((
            bboxes[:, 0] - bboxes[:, 3] / 2,
            bboxes[:, 1] - bboxes[:, 4] / 2,
            bboxes[:, 0] + bboxes[:, 3] / 2,
            bboxes[:, 1] + bboxes[:, 4] / 2,
            bboxes[:, 6]
        ), dim=1)
        bboxes, scores, labels = box3d_multiclass_nms(
            mlvl_bboxes=bboxes,
            mlvl_bboxes_for_nms=bboxes_for_nms,
            mlvl_scores=scores,
            score_thr=self.test_cfg.score_thr,
            max_num=self.test_cfg.nms_pre,
            cfg=self.test_cfg
        )
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5))
        return bboxes, scores, labels
