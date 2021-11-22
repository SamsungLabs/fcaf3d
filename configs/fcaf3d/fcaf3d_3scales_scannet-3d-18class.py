_base_ = ['fcaf3d_scannet-3d-18class.py']

model = dict(
    backbone=dict(
        n_outs=3),
    neck_with_head=dict(
        in_channels=(64, 128, 256),
        assigner=dict(
            n_scales=3)))
