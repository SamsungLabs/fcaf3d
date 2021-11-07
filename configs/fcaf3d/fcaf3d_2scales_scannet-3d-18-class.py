_base_ = ['fcaf3d_scannet-3d-18-class.py']
voxel_size = 0.02

model = dict(
    backbone=dict(
        n_outs=2),
    neck_with_head=dict(
        assigner=dict(
            n_scales=2)))
