import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import multi_apply, reduce_mean, build_assigner, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import Scale, bias_init_with_prob

from mmdet3d.core.post_processing import aligned_3d_nms


class SparseFcos3DHead(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 n_convs,
                 n_reg_outs,
                 voxel_size,
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
        self.n_classes = n_classes
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_convs, n_reg_outs)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, n_channels, n_convs, n_reg_outs):
        self.cls_convs = nn.Sequential(*[
            self._make_block(n_channels, n_channels)
            for _ in range(n_convs)
        ])
        self.reg_convs = nn.Sequential(*[
            self._make_block(n_channels, n_channels)
            for _ in range(n_convs)
        ])
        self.centerness_conv = ME.MinkowskiConvolution(n_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(n_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(n_channels, self.n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(self.assigner.n_scales)])

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
        return multi_apply(self.forward_single, x, self.scales)

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
                self._bbox_pred_to_loss(pos_points, pos_bbox_preds),
                self._bbox_pred_to_loss(pos_points, pos_bbox_targets),
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

            bboxes = self._bbox_pred_to_result(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    # per scale
    def forward_single(self, x, scale):
        raise NotImplementedError

    def _bbox_pred_to_loss(self, points, bbox_preds):
        raise NotImplementedError

    def _bbox_pred_to_result(self, points, bbox_preds):
        raise NotImplementedError

    def _nms(self, bboxes, scores, img_meta):
        raise NotImplementedError


@HEADS.register_module()
class ScanNetSparseFcos3DHead(SparseFcos3DHead):
    def forward_single(self, x, scale):
        cls = self.cls_convs(x)
        reg = self.reg_convs(x)
        centerness = self.centerness_conv(reg).features
        bbox_pred = torch.exp(scale(self.reg_conv(reg).features))
        cls_score = self.cls_conv(cls).features

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            # todo: do we need + .5?
            points[i] = points[i] * self.voxel_size

        return centernesses, bbox_preds, cls_scores, points

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return aligned_bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return aligned_bbox_pred_to_bbox(points, bbox_preds)

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = aligned_3d_nms(bboxes, scores, labels, self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
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


def aligned_bbox_pred_to_bbox(points, bbox_pred):
    return torch.stack([
        points[:, 0] - bbox_pred[:, 0],
        points[:, 1] - bbox_pred[:, 2],
        points[:, 2] - bbox_pred[:, 4],
        points[:, 0] + bbox_pred[:, 1],
        points[:, 1] + bbox_pred[:, 3],
        points[:, 2] + bbox_pred[:, 5]
    ], -1)


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    # todo: sqrt ?
    return torch.sqrt(centerness_targets)


@BBOX_ASSIGNERS.register_module()
class Fcos3dAssigner(BaseAssigner):
    def __init__(self, regress_ranges):
        self.regress_ranges = regress_ranges
        self.n_scales = len(regress_ranges)

    def assign(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        assert len(points) == len(self.regress_ranges)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i]).expand(len(points[i]), 2)
            for i in range(len(points))
        ]
        # concat all levels points and regress ranges
        regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        points = torch.cat(points, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        regress_ranges = regress_ranges[:, None, :].expand(n_points, n_boxes, 2)
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.dims), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        xs = xs[:, None].expand(n_points, n_boxes)
        ys = ys[:, None].expand(n_points, n_boxes)
        zs = zs[:, None].expand(n_points, n_boxes)

        dx_min = xs - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - xs
        dy_min = ys - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - ys
        dz_min = zs - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - zs
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_regress_range, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, bbox_targets, labels


@BBOX_ASSIGNERS.register_module()
class Fcos3dAssignerV2(BaseAssigner):
    def __init__(self, regress_ranges, topk):
        self.regress_ranges = regress_ranges
        self.n_scales = len(regress_ranges)
        self.topk = topk

    def assign(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        assert len(points) == len(self.regress_ranges)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i]).expand(len(points[i]), 2)
            for i in range(len(points))
        ]
        # concat all levels points and regress ranges
        regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        points = torch.cat(points, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        regress_ranges = regress_ranges[:, None, :].expand(n_points, n_boxes, 2)
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.dims), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        xs = xs[:, None].expand(n_points, n_boxes)
        ys = ys[:, None].expand(n_points, n_boxes)
        zs = zs[:, None].expand(n_points, n_boxes)

        dx_min = xs - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - xs
        dy_min = ys - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - ys
        dz_min = zs - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - zs
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_regress_range, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.topk, dim=0).values[-1]
        inside_top_centerness = centerness > top_centerness.unsqueeze(0)


        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_regress_range, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness, volumes, torch.ones_like(volumes) * float_max)  # todo: ???
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, bbox_targets, labels


@BBOX_ASSIGNERS.register_module()
class Atss3dAssigner(BaseAssigner):
    def __init__(self, anchor_sizes):
        self.anchor_sizes = anchor_sizes
        self.n_scales = len(anchor_sizes)

    def assign(self, points, gt_bboxes, gt_labels):
        # 1. for each object get anchors inside it
        # 2. for each object and anchor inside it get iou
        # 3. for each object compute mean and std valid iou
        # 4. assign object to anchors with iou > mean + std
        # 5. for each anchor select object with min volume

        float_max = 1e8
