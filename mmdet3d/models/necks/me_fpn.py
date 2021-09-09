import torch
from torch import nn
import MinkowskiEngine as ME

from mmdet.models import NECKS, build_loss


@NECKS.register_module()
class MEFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 voxel_size,
                 n_outs=4,
                 pts_threshold=-1,
                 gt_threshold=2.5,
                 loss=dict(
                     type='CrossEntropyLoss',
                     reduction='none',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super(MEFPN3D, self).__init__()
        self.voxel_size = voxel_size
        self.n_outs = n_outs
        self.pts_threshold = pts_threshold
        self.gt_threshold = gt_threshold
        self.loss = build_loss(loss)
        self.n_scales = len(in_channels)
        self.pruning = ME.MinkowskiPruning()
        for i in range(self.n_scales):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
                if self.pts_threshold > 0:
                    self.__setattr__(
                        f'score_block_{i}',
                        ME.MinkowskiConvolution(in_channels[i - 1], 1, kernel_size=1, bias=True, dimension=3))
            if i < self.n_outs:
                self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

    def forward(self, inputs, gt_bboxes=None, gt_labels=None, img_metas=None):
        outs, losses = [], []
        x = inputs[-1]
        for i in range(self.n_scales - 1, -1, -1):
            if i < self.n_scales - 1:
                x = inputs[i] + x
            if i < self.n_outs:
                outs.append(self.__getattr__(f'out_block_{i}')(x))
            if i > 0:
                x = self.__getattr__(f'up_block_{i}')(x)
                if self.pts_threshold > 0:
                    score = self.__getattr__(f'score_block_{i}')(x)
                    mask, loss = self._prune(score, gt_bboxes, gt_labels, img_metas)
                    x = self.pruning(x, mask)
                    losses += loss
        outs = outs[::-1]
        if gt_bboxes is not None:  # training
            if self.pts_threshold > 0:
                losses = dict(loss_pruning=torch.mean(torch.cat(losses)))
            else:
                losses = dict()
            return outs, losses
        return outs

    def init_weights(self):
        pass

    def _prune(self, score, gt_bboxes, gt_labels, img_metas):
        size = score.tensor_stride[0] * self.voxel_size
        points = [point * self.voxel_size for point in score.decomposed_coordinates]
        permutations = score.decomposition_permutations
        scores = [score.features[permutation] for permutation in permutations]
        topks = [min(self.pts_threshold, len(permutation)) for permutation in permutations]

        masks = []
        for i in range(len(img_metas)):
            masks.append(self._get_score_mask(scores[i], topks[i]))

        losses = []
        if gt_bboxes is not None:  # training
            for i in range(len(img_metas)):
                gt_mask = self._get_gt_mask(points[i], gt_bboxes[i], gt_labels[i], img_metas[i], size)
                masks[i] = torch.logical_or(masks[i], gt_mask)
                losses.append(self.loss(scores[i], gt_mask.unsqueeze(1)).squeeze(1))

        prune_mask = score.features.new_zeros((len(score.features)), dtype=torch.bool)
        for permutation, mask in zip(permutations, masks):
            prune_mask[permutation[mask]] = True
        return prune_mask, losses

    def _get_score_mask(self, scores, topk):
        mask = scores.new_zeros((len(scores)), dtype=torch.bool)
        ids = torch.topk(scores.squeeze(1), topk, sorted=False).indices
        mask[ids] = True
        return mask

    def _get_gt_mask(self, points, gt_bboxes, gt_labels, img_meta, size):
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        sizes = points.new_ones((n_points, n_boxes)) * size
        centers = gt_bboxes.gravity_center.unsqueeze(0).to(points.device).expand(n_points, n_boxes, 3)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        distances = torch.sqrt(torch.pow(points - centers, 2).sum(dim=-1))
        mask = torch.any(distances < sizes * self.gt_threshold, dim=1)
        return mask

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

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )
