import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import build_assigner
from mmdet.models.builder import HEADS, LOSSES, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.losses import CrossEntropyLoss

from mmdet3d.ops.pcdet_nms import pcdet_nms_normal_gpu


@HEADS.register_module()
class GsdnNeckWithHead(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,
                 voxel_size,
                 anchors,
                 strides,
                 prune_threshold,
                 prune_assigner,
                 bbox_assigner,
                 loss_prune=dict(
                     type='BalancedCrossEntropyLoss',
                     reduction='sum',
                     use_sigmoid=True,
                     loss_weight=1.),
                 loss_anchor=dict(
                     type='BalancedCrossEntropyLoss',
                     reduction='sum',
                     use_sigmoid=True,
                     loss_weight=1.),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.),
                 loss_bbox=dict(
                     type='SmoothL1Loss',
                     loss_weight=10.),  # todo: it is .1 in the paper
                 train_cfg=None,
                 test_cfg=None):
        super(GsdnNeckWithHead, self).__init__()
        self.n_classes = n_classes
        self.voxel_size = voxel_size
        self.anchors = anchors
        self.strides = strides
        self.n_anchors = len(anchors)
        self.prune_threshold = prune_threshold
        self.prune_assiner = build_assigner(prune_assigner)
        self.bbox_assigner = build_assigner(bbox_assigner)
        self.loss_prune = build_loss(loss_prune)
        self.loss_anchor = build_loss(loss_anchor)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU()
        )

    def _init_layers(self, in_channels):
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i < len(in_channels) - 1:
                self.__setattr__(f'block_{i}', self._make_block(in_channels[i], in_channels[i]))
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
                self.__setattr__(
                    f'prune_conv_{i}',
                    ME.MinkowskiConvolution(in_channels[i], 1, kernel_size=1, bias=True, dimension=3))
            out_channels = (self.n_classes + 7) * self.n_anchors
            self.__setattr__(
                f'final_conv_{i}',
                ME.MinkowskiConvolution(in_channels[i], out_channels, kernel_size=1, bias=True, dimension=3))

    def init_weights(self):
        pass

    def forward(self, x):
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = inputs[i] + x if x is not None else inputs[i]
                x = self.__getattr__(f'block_{i}')(x)
            # skip pruning for the first scale
            prune_pred = self.__getattr__(f'prune_conv_{i}')(x) if i > 0 else None
            final_pred = self.__getattr__(f'final_conv_{i}')(x)
            outs.append(self._split_preds(prune_pred, final_pred))
            if i > 0:
                prune_pred = prune_pred.features.squeeze(1).sigmoid() > self.prune_threshold
                if prune_pred.sum() != 0:
                    x = self.pruning(x, prune_pred)
                    x = self.__getattr__(f'up_block_{i}')(x)
                else:  # do nothing if prune mask is empty
                    x = None
        return zip(*outs[::-1])

    def _split_preds(self, prune_pred, final_pred):
        """
        Returns:
            prune_preds (list[Tensor] | None): of shape (N,)
            anchor_preds (list[Tensor]): of shape (N, N_anchors)
            cls_preds (list[Tensor]): of shape (N, N_anchors, N_classes)
            bbox_preds (list[Tensor]): of shape (N, N_anchors, 6)
            points (list[Tensor]): of shape (N, 3)
        """
        prune_preds, anchor_preds, cls_preds, bbox_preds = [], [], [], []
        for permutation in final_pred.decomposition_permutations:
            prune = prune_pred.features.squeeze(-1)[permutation] if prune_pred is not None else None
            prune_preds.append(prune)
            final = final_pred.features[permutation]
            anchor_preds.append(final[:, :self.n_anchors])
            cls_pred = final[:, self.n_anchors:self.n_anchors * (1 + self.n_classes)]
            cls_preds.append(cls_pred.reshape(-1, self.n_anchors, self.n_classes))
            bbox_preds.append(final[:, -6 * self.n_anchors:].reshape(-1, self.n_anchors, 6))

        points = final_pred.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return prune_preds, anchor_preds, cls_preds, bbox_preds, points

    def loss(self, prune_preds, anchors_preds, cls_preds, bbox_preds, points,
             gt_bboxes, gt_labels, img_metas):
        losses_prune, losses_anchor, losses_cls, losses_bbox = [], [], [], []
        for i in range(len(img_metas)):
            loss_prune, loss_anchor, loss_cls, loss_bbox = self._loss_single(
                prune_preds=[x[i] if x is not None else None for x in prune_preds],
                anchor_preds=[x[i] for x in anchors_preds],
                cls_preds=[x[i] for x in cls_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                points=[x[i] for x in points],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i])
            losses_prune.append(loss_prune)
            losses_anchor.append(loss_anchor)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
        return dict(
            loss_prune=torch.mean(torch.stack(losses_prune)),
            loss_anchor=torch.mean(torch.stack(losses_anchor)),
            loss_cls=torch.mean(torch.stack(losses_cls)),
            loss_bbox=torch.mean(torch.stack(losses_bbox))
        )

    # per image
    def _loss_single(self, prune_preds, anchor_preds, cls_preds, bbox_preds, points,
                     gt_bboxes, gt_labels, img_meta):
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.dims), dim=-1).to(points[0].device)
        gt_labels = gt_labels.to(points[0].device)

        with torch.no_grad():
            # skip empty predictions for first level
            prune_targets = self.prune_assiner.assign(points[1:], gt_bboxes)
            assign_results = self.bbox_assigner.assign(points, gt_bboxes)

        loss_prune = self.loss_prune(torch.cat(prune_preds[1:]), torch.cat(prune_targets))
        anchor_preds = torch.cat(anchor_preds)
        # negative: -1 -> 0; ignore: -2 -> -1
        anchor_targets = torch.cat(assign_results) + 1
        anchor_targets = torch.where(anchor_targets > 0, 1, anchor_targets)
        loss_anchor = self.loss_anchor(anchor_preds.reshape(-1), anchor_targets.reshape(-1))

        cls_preds = torch.cat(cls_preds)
        bbox_preds = torch.cat(bbox_preds)
        pos_mask = anchor_targets > 0
        if torch.sum(pos_mask) > 0:
            cls_targets, bbox_targets = self._assign_results_to_targets(points, assign_results, gt_bboxes, gt_labels)
            loss_cls = self.loss_cls(cls_preds[pos_mask], cls_targets[pos_mask])
            loss_bbox = self.loss_bbox(bbox_preds[pos_mask], bbox_targets[pos_mask])
        else:
            loss_cls = cls_preds[pos_mask].sum()
            loss_bbox = bbox_preds[pos_mask].sum()
        return loss_prune, loss_anchor, loss_cls, loss_bbox

    def get_bboxes(self, prune_preds, anchor_preds, cls_preds, bbox_preds, points,
                   img_metas, rescale=False):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                anchor_preds=[x[i] for x in anchor_preds],
                cls_preds=[x[i] for x in cls_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self, anchor_preds, cls_preds, bbox_preds, points, img_meta):
        bboxes = self._bbox_preds_to_bbox(points, bbox_preds).reshape(-1, 6)
        scores = torch.cat(cls_preds).sigmoid() * torch.cat(anchor_preds).unsqueeze(-1).sigmoid()
        scores = scores.reshape(-1, self.n_classes)

        if len(scores) > self.test_cfg.nms_pre > 0:
            max_scores, _ = scores.max(dim=-1)
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bboxes = bboxes[ids]
            scores = scores[ids]

        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    def _nms(self, bboxes, scores, img_meta):
        n_classes = scores.shape[1]
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            class_bboxes = torch.cat((
                class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)

            nms_ids, _ = pcdet_nms_normal_gpu(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _bbox_preds_to_bbox(self, points, bbox_preds):
        """
        Args:
            points (list[Tensor]): of shape (N_i, 3)
            bbox_preds (list[Tensor]): of shape (N_i, N_anchors, 6)

        Returns:
            bboxes (Tensor): of shape (N, N_anchors, 6)
        """
        strides = [p.new_tensor(s).expand(len(p)) for p, s in zip(points, self.strides)]
        strides = torch.cat(strides) * self.voxel_size
        n_points = len(strides)
        strides = strides.reshape(n_points, 1, 1).expand(n_points, self.n_anchors, 3)
        points = torch.cat(points).reshape(n_points, 1, 3).expand(n_points, self.n_anchors, 3)
        bbox_preds = torch.cat(bbox_preds)
        anchors = points.new_tensor(self.anchors).unsqueeze(0).expand(n_points, self.n_anchors, 3)
        anchor_bboxes = torch.cat((points, anchors * strides), dim=-1)
        return torch.stack((
            anchor_bboxes[..., 0] + bbox_preds[..., 0] * anchor_bboxes[..., 3],
            anchor_bboxes[..., 1] + bbox_preds[..., 1] * anchor_bboxes[..., 4],
            anchor_bboxes[..., 2] + bbox_preds[..., 2] * anchor_bboxes[..., 5],
            torch.exp(bbox_preds[..., 3]) * anchor_bboxes[..., 3],
            torch.exp(bbox_preds[..., 4]) * anchor_bboxes[..., 4],
            torch.exp(bbox_preds[..., 5]) * anchor_bboxes[..., 5]
        ), dim=-1)

    def _assign_results_to_targets(self, points, assign_results, gt_bboxes, gt_labels):
        """
        Args:
            points (list[Tensor]): of shape (N_i, 3)
            assign_results (list[Tensor]): of shape (N_i, N_anchors)
            gt_bboxes (Tensor)
            gt_labels (Tensor)

        Returns:
            cls_targets (Tensor): of shape (N, N_anchors); N = sum_i N_i
            bbox_targets (Tensor): of shape (N, N_anchors, 6)
        """
        strides = [p.new_tensor(s).expand(len(p)) for p, s in zip(points, self.strides)]
        strides = torch.cat(strides) * self.voxel_size
        n_points = len(strides)
        strides = strides.reshape(n_points, 1, 1).expand(n_points, self.n_anchors, 3)
        points = torch.cat(points).reshape(n_points, 1, 3).expand(n_points, self.n_anchors, 3)
        assign_results = torch.cat(assign_results)
        # assign negative anchors with first object; will be ignored in loss
        assign_results = torch.where(assign_results < 0, 0, assign_results)
        anchors = points.new_tensor(self.anchors).unsqueeze(0).expand(n_points, self.n_anchors, 3)
        anchor_bboxes = torch.cat((points, anchors * strides), dim=-1)
        gt_bboxes = gt_bboxes[assign_results]
        bbox_targets = torch.stack((
            (gt_bboxes[..., 0] - anchor_bboxes[..., 0]) / anchor_bboxes[..., 3],
            (gt_bboxes[..., 1] - anchor_bboxes[..., 1]) / anchor_bboxes[..., 4],
            (gt_bboxes[..., 2] - anchor_bboxes[..., 2]) / anchor_bboxes[..., 5],
            torch.log(gt_bboxes[..., 3] / anchor_bboxes[..., 3]),
            torch.log(gt_bboxes[..., 4] / anchor_bboxes[..., 4]),
            torch.log(gt_bboxes[..., 5] / anchor_bboxes[..., 5])
        ), dim=-1)
        return gt_labels[assign_results], bbox_targets


@BBOX_ASSIGNERS.register_module()
class GsdnPruneAssigner:
    def __init__(self, voxel_size, strides, pos_iou_thr):
        self.voxel_size = voxel_size
        self.strides = strides
        self.pos_iou_thr = pos_iou_thr

    def assign(self, points, gt_bboxes):
        """Positive if there exists an object with center in the given voxel and
            its volume < voxel volume / 8 / pos_iou_thr.
            # todo: this is not exact following of the paper!

        Args:
            points (list[Tensor]): of shape (N_i, 3)
            gt_bboxes (Tensor)

        Returns:
            prune_targets (list[Tensor]): of shape (N_i,); 1 for positive, 0 for negative
        """
        if len(gt_bboxes) == 0:
            return [p.new_zeros((len(p), self.n_anchors), dtype=torch.long) for p in points]

        prune_targets = []
        for p, s in zip(points, self.strides):
            prune_targets.append(self._assign_single(p, s, gt_bboxes))
        return prune_targets

    def _assign_single(self, points, stride, gt_bboxes):
        n_points = len(points)
        n_objects = len(gt_bboxes)
        points = points.reshape(n_points, 1, 3).expand(n_points, n_objects, 3)
        anchor_size = points.new_tensor(self.voxel_size * stride).reshape(1, 1).expand(n_points, n_objects)
        gt_bboxes = gt_bboxes.reshape(1, n_objects, 6).expand(n_points, n_objects, 6)

        # condition 1: gt volume < anchor volume / 8 / pos_iou_thr
        anchor_volume = anchor_size * anchor_size * anchor_size
        gt_volume = gt_bboxes[..., 3] * gt_bboxes[..., 4] * gt_bboxes[..., 5]
        volume_condition = gt_volume < anchor_volume / 8 / self.pos_iou_thr

        # condition 2: gt center inside anchor box
        center_condition = torch.all(torch.cat((
            gt_bboxes[..., :3] < points + anchor_size.unsqueeze(-1) / 2,
            points - anchor_size.unsqueeze(-1) / 2 < gt_bboxes[..., :3]
        ), dim=-1), dim=-1)

        return torch.any(torch.logical_and(volume_condition, center_condition), dim=-1).long()


@BBOX_ASSIGNERS.register_module()
class GsdnBboxAssigner:
    def __init__(self,
                 voxel_size,
                 anchors,
                 strides,
                 pos_iou_thr,
                 neg_iou_thr,
                 iou_calculator=dict(type='AxisAlignedBboxOverlaps3D')):
        self.voxel_size = voxel_size
        self.anchors = anchors
        self.n_anchors = len(anchors)
        self.strides = strides
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

        self.oks = []
        self.alls = []
        self.assigned_anchors = []

    def assign(self, points, gt_bboxes):
        """
        Args:
            points (list[Tensor]): of shape (N_i, 3)
            gt_bboxes (Tensor)

        Returns:
            assign_results (list[Tensor]): of shape (N_i, N_anchors); object id for positive,
                -1 for negative, -2 for ignore
        """
        if len(gt_bboxes) == 0:
            return [p.new_ones((len(p), self.n_anchors) * -1, dtype=torch.long) for p in points]

        assign_results = []
        for p, s in zip(points, self.strides):
            assign_results.append(self._assign_single(p, s, gt_bboxes))

        return assign_results

    # per level
    def _assign_single(self, points, stride, gt_bboxes):
        n_points = len(points)
        n_objects = len(gt_bboxes)
        anchors = points.new_tensor(self.anchors).reshape(1, self.n_anchors, 3).expand(n_points, self.n_anchors, 3)
        anchors = anchors * stride * self.voxel_size
        points = points.reshape(n_points, 1, 3).expand(n_points, self.n_anchors, 3)
        anchor_bboxes = torch.cat((points, anchors), -1).reshape(-1, 6)
        ious = self.iou_calculator(
            self._bbox_to_calculator(anchor_bboxes),
            self._bbox_to_calculator(gt_bboxes)).reshape(n_points, self.n_anchors, n_objects)
        # print('ious.max(), ious.shape', ious.max(), ious.shape)
        max_ious, max_indices = torch.max(ious, dim=-1)
        max_indices = torch.where(torch.logical_and(
            max_ious < self.pos_iou_thr,
            self.neg_iou_thr < max_ious
        ), -2, max_indices)
        max_indices = torch.where(max_ious < self.neg_iou_thr, -1, max_indices)
        # for y in [-1, -2] + list(range(18)):
        #     print(y, (max_indices == y).sum())
        return max_indices

    @staticmethod
    def _bbox_to_calculator(bbox):
        return torch.stack((
            bbox[..., 0] - bbox[..., 3] / 2,
            bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2,
            bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2,
            bbox[..., 2] + bbox[..., 5] / 2
        ), dim=-1)


@LOSSES.register_module()
class BalancedCrossEntropyLoss(CrossEntropyLoss):
    def forward(self, cls_score, label, **kwargs):
        n_pos = max(torch.sum(label == 1), 1)
        n_neg = max(torch.sum(label == 0), 1)
        weight = cls_score.new_full((len(label),), .5 / n_pos)  # positive
        weight = torch.where(label == 0, .5 / n_neg, weight)  # negative
        weight = torch.where(label < 0, torch.zeros_like(weight), weight)  # ignore
        return super(BalancedCrossEntropyLoss, self).forward(cls_score, label, weight)
