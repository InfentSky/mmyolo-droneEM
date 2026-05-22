# Copyright (c) InfantSky. All rights reserved.
# Author: InfantSky on 2026-4-25
# Description: a video demo for deployment.

# 1. 必须继承 DJI 指定的基准配置，确保算子适配
_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

load_from = 'work_dirs/yolov8_s_person_car/epoch_300.pth'

# ======================== 基础参数修改 ======================
img_scale = (960, 960)
data_root = 'data/Drone-vision/' # 你的裂缝数据集根目录
class_name = ('bus','car', 'van', 'person')          # 根据你 JSON 里的分类填写
num_classes = 4
metainfo = dict(classes=('bus', 'car', 'van', 'person'),
                 palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32), (0, 165, 255)])

# ----- 路径映射 
train_ann_file = 'train/modified_annotations.coco.json'
train_data_prefix = 'train/'
val_ann_file = 'valid/modified_annotations.coco.json'
val_data_prefix = 'valid/'

# ======================== 针对裂缝识别的优化 ======================

# 1. 修改模型输出头
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

# 2. 硬件与学习率 (1660 Super)
train_batch_size_per_gpu = 16 # 裂缝检测建议 12 或 16
train_num_workers = 4
base_lr = 0.002 / (16/128)            # 线性缩放原理

# 3. 吸收猫咪教程的精华：自动保存最优模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10, 
        max_keep_ckpts=3, 
        save_best='coco/bbox_mAP' # 训练完会自动把最好的 pth 挑出来
    )
)

# 4. 训练周期控制
max_epochs = 150
train_cfg = dict(
    max_epochs=max_epochs, 
    val_interval=5)

# ======================== 数据加载器配置 ======================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
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
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

#----------------------------更新数据流----------------------------
train_pipeline = [
    dict(
        type='Mosaic', img_scale=img_scale),
    dict(
        type='YOLOv5RandomAffine',
        border=(-img_scale[0] // 2, -img_scale[1] // 2)
    )
]

train_pipeline_stage2 = [
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale)
]

test_pipeline = [
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale)
]

# ======================== 可视化配置 ======================

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),  # 保留本地图片可视化
        dict(type='TensorboardVisBackend')  # 开启 Tensorboard 支持
    ],
    name='visualizer'
)