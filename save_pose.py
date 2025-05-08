#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_pose_disk.py

从磁盘读取：
  data_dir/
    ├── cam_K.txt
    ├── rgb/000000.png
    ├── depth/000000.png  # 16-bit PNG，单位 mm
    └── masks/000000.png  # 8-bit 二值 mask

运行环境：foundationpose conda env
"""

import argparse, time, logging
from pathlib import Path

import numpy as np
import cv2
import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',    required=True,
                    help='demo 数据目录，含 cam_K.txt, rgb/, depth/, masks/')
parser.add_argument('--mesh_file',   required=True,
                    help='物体 mesh 路径, e.g. rubic_cube.obj')
parser.add_argument('--out_file',    default='./pose_cam_obj.npz',
                    help='输出位姿 npz 路径')
parser.add_argument('--est_iter',    type=int, default=5,
                    help='FoundationPose 注册迭代次数')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def main():
    data_dir = Path(args.data_dir)
    # 1. 内参
    K = np.loadtxt(data_dir / 'cam_K.txt').astype(np.float64)

    # 2. 读 RGB（BGR→RGB）, Depth, Mask
    rgb_bgr = cv2.imread(str(data_dir/'rgb'/'000000.png'),
                        cv2.IMREAD_COLOR)
    assert rgb_bgr is not None, "无法读取 RGB"
    rgb = rgb_bgr[..., ::-1].copy()  # 转成 RGB

    depth_raw = cv2.imread(str(data_dir/'depth'/'000000.png'),
                           cv2.IMREAD_UNCHANGED)
    assert depth_raw is not None, "无法读取 Depth"
    # uint16 (mm) → float32 (m)
    depth = depth_raw.astype(np.float32) / 1000.0

    mask_raw = cv2.imread(str(data_dir/'masks'/'000000.png'),
                          cv2.IMREAD_UNCHANGED)
    assert mask_raw is not None, "无法读取 Mask"
    mask = (mask_raw > 0)  # bool mask

    # 检查尺寸一致
    H, W = depth.shape
    assert rgb.shape[0]==H and rgb.shape[1]==W and mask.shape==(H,W), \
        f"分辨率不匹配: rgb{rgb.shape}, depth{depth.shape}, mask{mask.shape}"

    # 3. 初始化 FoundationPose
    mesh   = trimesh.load(args.mesh_file)
#    # —— 强制将顶点和法向量转换为 float32 —— 
#     mesh.vertices     = mesh.vertices.astype(np.float32).copy()
#     mesh.vertex_normals = mesh.vertex_normals.astype(np.float32).copy()

    scorer = ScorePredictor()
    refiner= PoseRefinePredictor()
    est = FoundationPose(
        model_pts     = mesh.vertices,
        model_normals = mesh.vertex_normals,
        mesh          = mesh,
        scorer        = scorer,
        refiner       = refiner,
        debug_dir     = Path('./debug_fp'),
        debug         = 1,
        glctx         = dr.RasterizeCudaContext(),
    )  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

    # 4. 注册
    logging.info("开始注册 …")
    pose_cam_obj = est.register(K, rgb, depth, mask,
                                iteration=args.est_iter)
    logging.info(f"注册完成，位姿矩阵：\n{pose_cam_obj}")

    # 5. 保存结果
    np.savez_compressed(args.out_file,
                        pose      = pose_cam_obj,
                        timestamp = time.time())
    logging.info(f"已写入 {args.out_file}")

if __name__ == "__main__":
    main()

