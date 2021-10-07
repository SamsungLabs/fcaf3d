import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

from mmdet3d.ops.rotated_iou import cal_giou_3d, cal_iou_3d
from mmdet3d.core.bbox import AxisAlignedBboxOverlaps3D


@weighted_loss
def iou_3d_loss(pred, target):
    return 1 - cal_iou_3d(pred[None, ...], target[None, ...])


@weighted_loss
def giou_3d_loss(pred, target):
    return cal_giou_3d(pred[None, ...], target[None, ...])[0][0]


@weighted_loss
def axis_aligned_iou_loss(pred, target):
    def _transform(bbox):
        return torch.stack((
            bbox[..., 0] - bbox[..., 3] / 2,
            bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2,
            bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2,
            bbox[..., 2] + bbox[..., 5] / 2,
        ), dim=-1)
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        _transform(pred), _transform(target), is_aligned=True)
    iou_loss = 1 - axis_aligned_iou
    return iou_loss


class IoU3DMixin(nn.Module):
    """Adapted from GIoULoss"""
    def __init__(self, loss_function, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.loss_function = loss_function
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * self.loss_function(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss


@LOSSES.register_module()
class IoU3DLoss(IoU3DMixin):
    def __init__(self, with_yaw=True, **kwargs):
        loss_function = iou_3d_loss if with_yaw else axis_aligned_iou_loss
        super().__init__(loss_function=loss_function, **kwargs)


@LOSSES.register_module()
class GIoU3DLoss(IoU3DMixin):
    def __init__(self, **kwargs):
        super().__init__(loss_function=giou_3d_loss, **kwargs)
