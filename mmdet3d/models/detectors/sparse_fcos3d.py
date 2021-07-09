import torch
from torch.nn import functional as F
import MinkowskiEngine as ME

from mmdet.models import DETECTORS
from mmdet3d.models import build_backbone, build_neck, build_head
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .base import Base3DDetector


@DETECTORS.register_module()
class SparseFcos3D(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 auxiliary_head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # todo: utilize auxiliary_head
        # self.auxiliary_head = build_head(auxiliary_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.neck.init_weights()
        self.bbox_head.init_weights()
        # todo: utilize auxiliary_head
        # self.auxiliary_head.init_weights()

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d):
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(*loss_inputs)
        # todo: utilize auxiliary_head
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
