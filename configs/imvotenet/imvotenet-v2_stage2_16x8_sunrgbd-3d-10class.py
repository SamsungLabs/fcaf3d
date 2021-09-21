_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py',
    '../_base_/models/imvotenet_image.py'
]

class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    pts_backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    pts_bbox_heads=dict(
        common=dict(
            type='VoteHeadV2',
            num_classes=10,
            pred_layer_cfg=dict(
                in_channels=128, shared_conv_channels=(128, 128), bias=True),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            objectness_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.2, 0.8],
                reduction='sum',
                loss_weight=5.0),
            center_loss=dict(
                type='ChamferDistance',
                mode='l2',
                reduction='sum',
                loss_src_weight=10.0,
                loss_dst_weight=10.0),
            iou_loss=dict(type='IoU3DLoss', reduction='sum', loss_weight=3.0),
            semantic_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
        joint=dict(
            vote_module_cfg=dict(
                in_channels=512,
                vote_per_seed=1,
                gt_per_seed=3,
                conv_channels=(512, 256),
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    type='ChamferDistance',
                    mode='l1',
                    reduction='none',
                    loss_dst_weight=10.0)),
            vote_aggregation_cfg=dict(
                type='PointSAModule',
                num_point=256,
                radius=0.3,
                num_sample=16,
                mlp_channels=[512, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True)),
        pts=dict(
            vote_module_cfg=dict(
                in_channels=256,
                vote_per_seed=1,
                gt_per_seed=3,
                conv_channels=(256, 256),
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    type='ChamferDistance',
                    mode='l1',
                    reduction='none',
                    loss_dst_weight=10.0)),
            vote_aggregation_cfg=dict(
                type='PointSAModule',
                num_point=256,
                radius=0.3,
                num_sample=16,
                mlp_channels=[256, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True)),
        img=dict(
            vote_module_cfg=dict(
                in_channels=256,
                vote_per_seed=1,
                gt_per_seed=3,
                conv_channels=(256, 256),
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    type='ChamferDistance',
                    mode='l1',
                    reduction='none',
                    loss_dst_weight=10.0)),
            vote_aggregation_cfg=dict(
                type='PointSAModule',
                num_point=256,
                radius=0.3,
                num_sample=16,
                mlp_channels=[256, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True)),
        loss_weights=[0.4, 0.3, 0.3]),
    img_mlp=dict(
        in_channel=18,
        conv_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    fusion_layer=dict(
        type='VoteFusion',
        num_classes=len(class_names),
        max_imvote_per_pixel=3),
    num_sampled_seed=1024,
    freeze_img_branch=True,

    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')),
    test_cfg=dict(
        img_rcnn=dict(score_thr=0.1),
        pts=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True)))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='IndoorPointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'points', 'gt_bboxes_3d',
            'gt_labels_3d'
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='IndoorPointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ]),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img', 'points'])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
evaluation = dict(pipeline=eval_pipeline)

# may also use your own pre-trained image branch
load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth'  # noqa
