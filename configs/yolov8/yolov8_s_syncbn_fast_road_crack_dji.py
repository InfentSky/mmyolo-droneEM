# 1. 必须继承 DJI 指定的基准配置，确保算子适配
_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

load_from = 'work_dirs/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'

# ======================== 基础参数修改 ======================
data_root = 'data/road-crack.v7i.coco/' # 你的裂缝数据集根目录
class_name = ('objects','Repair', 'fissure')          # 根据你 JSON 里的分类填写
num_classes = 3
metainfo = dict(classes=class_name,
                 palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]) # 裂缝通常用红色显示

# ----- 路径映射 (适配你那特殊的 train/vaild 结构) -----
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'
val_ann_file = 'valid/_annotations.coco.json'
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

# 2. 硬件与学习率 (4060)
train_batch_size_per_gpu = 16 # 裂缝检测建议 12 或 16
train_num_workers = 5
base_lr = 0.01 / (16/128)            # 线性缩放原理

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
max_epochs = 300
train_cfg = dict(
    max_epochs=max_epochs, 
    val_interval=8)
     # 每10轮验证一次

# ======================== 数据加载器配置 ======================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
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

# ======================== 可视化配置 ======================

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),  # 保留本地图片可视化
        dict(type='TensorboardVisBackend')  # 开启 Tensorboard 支持
    ],
    name='visualizer'
)
