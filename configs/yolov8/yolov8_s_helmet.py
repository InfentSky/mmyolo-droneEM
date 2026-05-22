# Copyright (c) InfantSky. All rights reserved.
# Author: InfantSky on 2026-4-25
# Description: a video demo for deployment.

# 1. 必须继承 DJI 指定的基准配置，确保算子适配
_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

load_from = 'work_dirs/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'

# ======================== 数据集配置 ======================
data_root = 'data/SHWD/'
num_classes = 2
metainfo = dict(classes=('hat', 'person'),
                palette=[(220, 20, 60), (0, 0, 142)])

train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'

# ======================== 硬件与训练参数 (RTX 3090) ======================
train_batch_size_per_gpu = 32     # 24GB 显存可承载
train_num_workers = 8             # 加速数据加载
persistent_workers = True         # 与基类保持一致

# 学习率: 线性缩放 (基类: 128总batch=0.01)
# 单卡32: 32/128 * 0.01 = 0.0025
base_lr = 0.0025

max_epochs = 200
close_mosaic_epochs = 20          # 最后20轮关闭Mosaic (10%)

# ======================== 模型修改 ======================
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes
        )
    )
)

# ======================== 训练与验证策略 ======================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)]  # Stage 2 每轮验证
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP'
    )
)

# ======================== 数据加载器 ======================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True)
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)
    )
)

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

# ======================== 优化器 ======================
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# ======================== 可视化 ======================
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)