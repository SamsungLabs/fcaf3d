_base_ = ['fcaf3d_scannet-3d-18class.py']
voxel_size = 0.02

model = dict(
    voxel_size=voxel_size,
    backbone=dict(
        n_outs=2),
    neck_with_head=dict(
        in_channels=(64, 128),
        voxel_size=voxel_size,
        assigner=dict(
            n_scales=2)))
