_base_ = ['fcaf3d_scannet-3d-18-class.py']

model = dict(
    backbone=dict(
        n_outs=3),
    neck_with_head=dict(
        assigner=dict(
            n_scales=3)))
