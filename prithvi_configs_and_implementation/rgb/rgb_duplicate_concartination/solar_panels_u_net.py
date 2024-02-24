custom_imports = dict(imports=['geospatial_fm'])
dataset_type = 'GeospatialDataset'
data_root = ''
num_frames = 1
img_size = 512
num_workers = 4
samples_per_gpu = 4
bands = [0, 1, 2, 3, 4, 5]
tile_size = 224
orig_nsize = 512
crop_size = (224, 224)
img_suffix = '_merged.tif'
seg_map_suffix = '_mask.tif'
ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True
experiment = 'Solar_mixed'
project_dir = 'work_dir_u_net_rgb'
work_dir = 'work_dir_u_net_rgb/Solar_mixed'
save_path = 'work_dir_u_net_rgb/Solar_mixed'
train_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='ConcartinateBands', bands=[3, 4, 5]),
    dict(
        type='Normalize',
        mean=[
            1740.5508625789564, 1989.266960548868, 2136.6044153018875,
            3594.146820739824, 3418.396550606708, 2755.2797671434832
        ],
        std=[
            410.28000498165164, 448.47107771197415, 540.9126273786808,
            777.9434626297228, 638.0564464391689, 589.4139642468862
        ],
        to_rgb=False),
    dict(type='RandomCrop', crop_size=(224, 224), cat_max_ratio=0.85),
    dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='ConcartinateBands', bands=[3, 4, 5]),
    dict(
        type='Normalize',
        mean=[
            1740.5508625789564, 1989.266960548868, 2136.6044153018875,
            3594.146820739824, 3418.396550606708, 2755.2797671434832
        ],
        std=[
            410.28000498165164, 448.47107771197415, 540.9126273786808,
            777.9434626297228, 638.0564464391689, 589.4139642468862
        ],
        to_rgb=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
CLASSES = ('Land', 'Solar Panels')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='GeospatialDataset',
        CLASSES=('Land', 'Solar Panels'),
        data_root='',
        img_dir='training',
        ann_dir='training',
        img_suffix='_merged.tif',
        seg_map_suffix='_mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='ConcartinateBands', bands=[3, 4, 5]),
            dict(
                type='Normalize',
                mean=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                std=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ],
                to_rgb=False),
            dict(type='RandomCrop', crop_size=(224, 224), cat_max_ratio=0.85),
            dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        ignore_index=-1),
    val=dict(
        type='GeospatialDataset',
        CLASSES=('Land', 'Solar Panels'),
        data_root='',
        img_dir='validation',
        ann_dir='validation',
        img_suffix='_merged.tif',
        seg_map_suffix='_mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='ConcartinateBands', bands=[3, 4, 5]),
            dict(
                type='Normalize',
                mean=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                std=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ],
                to_rgb=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ignore_index=-1),
    test=dict(
        type='GeospatialDataset',
        CLASSES=('Land', 'Solar Panels'),
        data_root='',
        img_dir='testing',
        ann_dir='testing',
        img_suffix='_merged.tif',
        seg_map_suffix='_mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='ConcartinateBands', bands=[3, 4, 5]),
            dict(
                type='Normalize',
                mean=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                std=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ],
                to_rgb=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ignore_index=-1))
ce_weights = [0.2, 0.8]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1,
    class_weight=[0.2, 0.8],
    avg_non_ignore=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=6,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.2, 0.8],
            avg_non_ignore=True)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.2, 0.8],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(112, 112), crop_size=(224, 224)))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=4e-05, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(
    by_epoch=True, interval=1000, out_dir='work_dir_u_net_rgb/Solar_mixed')
evaluation = dict(
    interval=1000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    by_epoch=False)
gpu_ids = range(0, 1)
auto_resume = False
