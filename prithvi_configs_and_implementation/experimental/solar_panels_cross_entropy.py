import os

custom_imports = dict(imports=["geospatial_fm"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
resume_from = ""
cudnn_benchmark = True

dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = ""

num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 8

img_norm_cfg = dict(
    means=[
        1740.5508625789564, 1989.266960548868, 2136.6044153018875, 3594.146820739824, 3418.396550606708, 2755.2797671434832
    ],
    stds=[
        410.28000498165164, 448.47107771197415, 540.9126273786808, 777.9434626297228, 638.0564464391689, 589.4139642468862
    ]
)  # change the mean and std of all the bands

bands = [0, 1, 2, 3, 4, 5]
tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
img_suffix = "_merged.tif"
seg_map_suffix = "_mask.tif"
ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "Prithvi_100M.pt"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = num_frames*embed_dim
max_intervals = 50000
evaluation_interval = 4000

# TO BE DEFINED BY USER: model path
experiment = "Solar_reduced_87_5_percent"
project_dir = "work_dir"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

save_path = work_dir
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
        channels_last=True
    ),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=False),
    dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=(tile_size, tile_size)),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(
            len(bands),
            num_frames,
            tile_size,
            tile_size
        )
    ),
    dict(
        type="Reshape",
        keys=["gt_semantic_seg"],
        new_shape=(1, tile_size, tile_size)
    ),
    dict(
        type="CastTensor",
        keys=["gt_semantic_seg"],
        new_type="torch.LongTensor"
    ),
    dict(type="Collect", keys=["img", "gt_semantic_seg"])
]
test_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
        channels_last=True
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up=dict({
            "2": 1,
            "3": 2
        })),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg"
        ]
    )
]

CLASSES = ("Land", "Solar Panels")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="training",
        ann_dir="training",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=-1),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="validation",
        ann_dir="validation",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=-1),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="testing",
        ann_dir="testing",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=-1
    )
)

optimizer = dict(type="AdamW", lr=4e-5, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook", by_epoch=False)
    ]
)

checkpoint_config = dict(
    by_epoch=True,
    interval=10,
    out_dir=save_path 
)

evaluation = dict(
    interval=evaluation_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=False
)
ce_weights = [0.2, 0.8]

loss_func = dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            avg_non_ignore=True
        )

runner = dict(type="IterBasedRunner", max_iters=max_intervals)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim*num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
gpu_ids = range(0, 1)
auto_resume = False
