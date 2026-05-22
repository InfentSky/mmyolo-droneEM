# Copyright (c) InfantSky. All rights reserved.
# Author: InfantSky on 2026-4-25
# Description: a video demo for deployment.

import argparse
import os
import random
import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config, ConfigDict
from mmyolo.utils import register_all_modules
from projects.easydeploy.model import TRTWrapper 

def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO EasyDeploy Video Inference')
    parser.add_argument('video', help='视频文件路径或摄像头索引 (如 0)')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='TensorRT engine file')
    parser.add_argument('--out', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show', action='store_true', help='是否实时显示窗口')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()
    return args

def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            x = x[None].float()
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x
    return PreProcess().eval()

def main():
    args = parse_args()
    register_all_modules()

    # 1. 加载模型
    model = TRTWrapper(args.checkpoint, args.device)
    model.to(args.device)

    cfg = Config.fromfile(args.config)
    class_names = cfg.get('class_name')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]

    # 2. 准备 Pipeline (将加载方式改为从数组读取)
    test_pipeline_cfg = get_test_pipeline_cfg(cfg)
    test_pipeline_cfg[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline_cfg)
    pre_pipeline = preprocess(cfg)

    # 3. 初始化视频流
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
    cap = cv2.VideoCapture(video_input)
    
    # 获取视频参数用于保存
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    print("开始推理，按 'q' 键退出...")

    frame_id = 0
    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        # --- 核心推理逻辑 (参考 image_demo) ---
        rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
        
        # 数据转换
        data, samples = test_pipeline(dict(img=rgb, img_id=frame_id)).values()
        
        # 提取缩放和填充参数用于坐标还原
        pad_param = samples.get('pad_param', np.array([0, 0, 0, 0], dtype=np.float32))
        h, w = samples.get('ori_shape', rgb.shape[:2])
        
        # 构造用于坐标修正的 tensor
        pad_param_t = torch.asarray(
            [pad_param[2], pad_param[0], pad_param[2], pad_param[0]],
            device=args.device)
        scale_factor = samples.get('scale_factor', [1., 1.])
        scale_factor_t = torch.asarray(scale_factor * 2, device=args.device)

        # 预处理并执行推理
        data = pre_pipeline(data).to(args.device)
        result = model(data)

        # 解析结果: num_dets, bboxes, scores, labels
        num_dets, bboxes, scores, labels = result
        num_dets = int(num_dets[0]) # 取第一张图的数量
        
        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        labels = labels[0, :num_dets]

        # 坐标还原逻辑
        bboxes -= pad_param_t
        bboxes /= scale_factor_t
        bboxes[:, 0::2].clamp_(0, w)
        bboxes[:, 1::2].clamp_(0, h)
        bboxes = bboxes.round().int()

        # --- 可视化 ---
        for (bbox, score, label) in zip(bboxes, scores, labels):
            if score < args.score_thr:
                continue
            
            bbox = bbox.tolist()
            label = int(label)
            color = colors[label]
            
            label_name = class_names[label] if class_names else str(label)
            text = f'{label_name}: {score:0.2f}'

            cv2.rectangle(bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(bgr, text, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if args.show:
            cv2.imshow('EasyDeploy Video Inference', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if video_writer:
            video_writer.write(bgr)
        
        frame_id += 1

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()