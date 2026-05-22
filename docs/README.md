<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/27466624/222385101-516e551c-49f5-480d-a135-4b24ee6dc308.png"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmyolo)](https://pypi.org/project/mmyolo)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmyolo.readthedocs.io/en/latest/)
[![deploy](https://github.com/open-mmlab/mmyolo/workflows/deploy/badge.svg)](https://github.com/open-mmlab/mmyolo/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmyolo/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmyolo)
[![license](https://img.shields.io/github/license/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)

[📘Documentation](https://mmyolo.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html) |
[👀Model Zoo](https://mmyolo.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmyolo.readthedocs.io/en/latest/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmyolo/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
</div>

---

# 🎓 毕业设计项目归档说明

> **本项目为本人本科毕业设计的项目归档。**
>
> 基于 OpenMMLab 开源项目 [MMYOLO](https://github.com/open-mmlab/mmyolo) 进行改进，针对目标检测任务中的特定需求进行了算法优化与工程实现。

- **作者**：[InfantSky]
- **时间**：2026 年 5 月
- **原始项目**：[MMYOLO](https://github.com/open-mmlab/mmyolo) © OpenMMLab

---

## 🧩 主要修改内容

以下是对原版 MMYOLO 的主要修改项：

### 1. 数据预处理参数调整

- **文件**：`configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`
- **修改内容**：将数据预处理的 `mean/std` 从 `[0., 0., 0.] / [255., 255., 255.]` 调整为 `[128., 128., 128.] / [128., 128., 128.]`
- **目的**：适配特定场景下的数据归一化需求，将输入像素值映射至 `[-1, 1]` 范围，加速模型收敛

### 2. 激活函数替换（SiLU → ReLU）

- **文件**：`configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`
- **修改内容**：将 backbone 中的激活函数从 `SiLU`（默认）替换为 `ReLU`（`inplace=True`）
- **目的**：降低计算复杂度与推理延迟，ReLU 在边缘计算设备上具有更好的部署友好性

### 3. 新增反卷积上采样模块（DeconvUpsampling）

- **新增文件**：`mmyolo/utils/deconv_upsampling.py`
- **功能**：实现了基于转置卷积（Deconvolution / Transposed Convolution）的上采样模块，替代原始的最近邻插值或双线性插值上采样方式
- **目的**：通过学习式上采样提升对小目标的检测能力，增强特征金字塔的特征表示质量

### 4. YOLOv8 结构改进

- **修改文件**：
  - `mmyolo/models/backbones/csp_darknet.py` — backbone 结构调整
  - `mmyolo/models/necks/yolov8_pafpn.py` — neck 特征融合路径优化
  - `mmyolo/models/dense_heads/yolov8_head.py` — 检测头适配改进
  - `mmyolo/models/layers/yolo_bricks.py` — 基础构建块（bricks）扩展
- **目的**：协同反卷积上采样模块，构建更有效的特征提取与融合链路

### 5. 训练脚本增强

- **文件**：`tools/train.py`
- **修改内容**：在训练脚本中集成新的反卷积上采样模块注册机制，确保自定义层可被 MMEngine 正确识别与序列化

### 6. 数据集路径适配

- **文件**：`configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`
- **修改内容**：将 `data_root` 修改为自定义路径 `/mlcdev/nnsdk/data/coco/`
- **目的**：适配特定计算环境下的数据集存储路径

---

## 🛠️ 快速开始

### 环境安装

```bash
# 创建虚拟环境
conda create -n mmyolo python=3.8 -y
conda activate mmyolo

# 安装依赖
pip install -r requirements.txt

# 安装 mmyolo
pip install -v -e .
```

### 训练

```bash
python tools/train.py configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
```

### 测试

```bash
python tools/test.py configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py work_dirs/yolov8_s_xxx/epoch_xxx.pth --show
```

---

## 📄 许可证

本项目基于 [GNU General Public License v3.0](LICENSE) 开源协议发布。

原项目 [MMYOLO](https://github.com/open-mmlab/mmyolo) 版权所有 © OpenMMLab。  
本项目中所有修改内容均遵循 GPLv3 协议，并已标明修改范围与日期。

---

## 🙏 致谢

- [OpenMMLab](https://github.com/open-mmlab) — 提供优秀的计算机视觉开源框架
- [MMYOLO](https://github.com/open-mmlab/mmyolo) — 本项目的原始基础
- 导师在毕设期间的指导与建议
