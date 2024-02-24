custom_imports = dict(imports=['geospatial_fm'])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
dataset_type = 'GeospatialDataset'
data_root = ''
num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4
img_norm_cfg = dict(
    means=[
        1740.5508625789564, 1989.266960548868, 2136.6044153018875,
        3594.146820739824, 3418.396550606708, 2755.2797671434832
    ],
    stds=[
        410.28000498165164, 448.47107771197415, 540.9126273786808,
        777.9434626297228, 638.0564464391689, 589.4139642468862
    ])
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
pretrained_weights_path = 'Prithvi_100M.pt'
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = 768
max_intervals = 10000
evaluation_interval = 1000
experiment = 'Solar_duplicated_c'
project_dir = 'work_dir_c_rgb'
work_dir = 'work_dir_c_rgb/Solar_duplicated'
save_path = 'work_dir_c_rgb/Solar_duplicated'
train_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='ConcartinateBands', bands=[3, 4, 5]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            1740.5508625789564, 1989.266960548868, 2136.6044153018875,
            3594.146820739824, 3418.396550606708, 2755.2797671434832
        ],
        stds=[
            410.28000498165164, 448.47107771197415, 540.9126273786808,
            777.9434626297228, 638.0564464391689, 589.4139642468862
        ]),
    dict(type='TorchRandomCrop', crop_size=(224, 224)),
    dict(type='Reshape', keys=['img'], new_shape=(6, 1, 224, 224)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, 224, 224)),
    dict(
        type='CastTensor',
        keys=['gt_semantic_seg'],
        new_type='torch.LongTensor'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='ConcartinateBands', bands=[3, 4, 5]),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            1740.5508625789564, 1989.266960548868, 2136.6044153018875,
            3594.146820739824, 3418.396550606708, 2755.2797671434832
        ],
        stds=[
            410.28000498165164, 448.47107771197415, 540.9126273786808,
            777.9434626297228, 638.0564464391689, 589.4139642468862
        ]),
    dict(
        type='Reshape',
        keys=['img'],
        new_shape=(6, 1, -1, -1),
        look_up=dict({
            '2': 1,
            '3': 2
        })),
    dict(type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
    dict(
        type='CollectTestList',
        keys=['img'],
        meta_keys=[
            'img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename',
            'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape',
            'scale_factor', 'img_norm_cfg'
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
            dict(type='RandomFlip', prob=0.5),
            dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                stds=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ]),
            dict(type='TorchRandomCrop', crop_size=(224, 224)),
            dict(type='Reshape', keys=['img'], new_shape=(6, 1, 224, 224)),
            dict(
                type='Reshape',
                keys=['gt_semantic_seg'],
                new_shape=(1, 224, 224)),
            dict(
                type='CastTensor',
                keys=['gt_semantic_seg'],
                new_type='torch.LongTensor'),
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
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                stds=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 1, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
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
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1740.5508625789564, 1989.266960548868, 2136.6044153018875,
                    3594.146820739824, 3418.396550606708, 2755.2797671434832
                ],
                stds=[
                    410.28000498165164, 448.47107771197415, 540.9126273786808,
                    777.9434626297228, 638.0564464391689, 589.4139642468862
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 1, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        ignore_index=-1))
optimizer = dict(type='AdamW', lr=4e-05, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
checkpoint_config = dict(
    by_epoch=True, interval=10, out_dir='work_dir_p_rgb/Solar_duplicated')
evaluation = dict(
    interval=1000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    by_epoch=False)
ce_weights = [0.2, 0.8]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1,
    class_weight=[0.2, 0.8],
    avg_non_ignore=True)
runner = dict(type='IterBasedRunner', max_iters=10000)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained='Prithvi_100M.pt',
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=768,
        output_embed_dim=768,
        drop_cls_token=True,
        Hp=14,
        Wp=14),
    decode_head=dict(
        num_classes=2,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.2, 0.8],
            avg_non_ignore=True)),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.2, 0.8],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(112, 112), crop_size=(224, 224)))
gpu_ids = range(0, 1)
auto_resume = False
