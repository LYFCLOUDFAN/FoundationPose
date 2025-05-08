#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_to_pose.py

步骤
1. 读取深度图 + 掩码，生成相机系点云 → 变换到基座系
2. 求质心，构造竖直向下姿态并抬高 `approach_h`
3. 用 MPlib 规划关节轨迹
4. min-jerk 插值后，通过 XArm6RealWorld.set_joint_values_sequence() 下发
"""

import sys, argparse, time, numpy as np, cv2, open3d as o3d
sys.path += [
    '/home/xzx/Dro/third_party/xarm6',
    '/home/xzx/Dro/third_party/xarm6/xarm6_interface'
]
sys.path.append('/home/xzx/Dro/third_party/leaphand/LEAP_Hand_API/python')

from send_custom_pose import LeapSender 
from xarm6_interface.arm_rw import XArm6RealWorld
from xarm6_interface.arm_mplib import (
    XARM6Planner, XARM6PlannerCfg, min_jerk_interpolator_with_alpha
)

import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from xarm6_interface.arm_rw import XArm6RealWorld
from xarm6_interface.arm_mplib import XARM6Planner, XARM6PlannerCfg

def parse_args():
    parser = argparse.ArgumentParser(description="Move XArm above an object using point cloud")
    parser.add_argument("--intrinsic", required=True, type=str, help="Path to camera intrinsic matrix (3x3)")
    parser.add_argument("--extrinsic", required=True, type=str, help="Path to camera extrinsic matrix (4x4)")
    parser.add_argument("--depth", required=True, type=str, help="Path to depth image file")
    parser.add_argument("--mask", required=True, type=str, help="Path to mask image file")
    parser.add_argument("--ip", required=True, type=str, help="Robot IP address")
    parser.add_argument("--approach_height", type=float, default=-0.01, help="Approach height above object centroid (meters)")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for downsampling point cloud")
    parser.add_argument("--interp_timestep", type=float, default=0.05, help="Interpolation time step for planning (seconds)")
    return parser.parse_args()

def load_matrix(path):
    """Load a matrix from a text or .npy file."""
    try:
        mat = np.loadtxt(path)
    except Exception:
        mat = np.load(path)
    return mat

def main():
    args = parse_args()
    hand_pose_deg = [  # 16 个角度，单位=度
    3.124719, 3.1615343, 3.1523306, 3.1400588,
    3.1308548, 3.1553984, 3.1492627, 3.1477287,
    3.1308548, 3.1600003, 3.1538644, 3.1431267,
    3.1584663, 3.124719, 3.1369908, 3.1339228,
]
    leap_sender = LeapSender(port="/dev/ttyUSB0")  # 如端口不同请改
    leap_sender.send(hand_pose_deg)
    time.sleep(0.5)
    # arm = XArm6RealWorld(args.ip)

    # GRIP_CLOSE_POS = 850    # 0=全开，≈850=最紧，可按需调
    # GRIP_SPEED     = 1000     # 越大越快，≤5000
    # arm.arm.set_gripper_position(GRIP_CLOSE_POS, speed=GRIP_SPEED, wait=False)
    # print(">>> 轨迹执行完毕，夹爪已闭合")

    # time.sleep(0.2) 
    

    # Load camera parameters
    intrinsic = load_matrix(args.intrinsic)
    extrinsic = load_matrix(args.extrinsic)

    # Read depth and mask
    try:
        import imageio
        depth = imageio.imread(args.depth)
    except Exception:
        depth = np.load(args.depth)
    try:
        mask_img = imageio.imread(args.mask)
    except Exception:
        mask_img = np.load(args.mask)
    # Ensure mask is binary
    if mask_img.dtype != bool:
        mask = mask_img > 0
    else:
        mask = mask_img

    depth = depth.astype(np.float64)
    # Convert depth to meters if needed (assuming original in mm if large values)
    if depth.max() > 1000:
        depth = depth * 0.001

    h, w = depth.shape
    # Build point cloud in camera frame for object points (mask == True)
    u_coord, v_coord = np.meshgrid(np.arange(w), np.arange(h))
    mask_flat = mask.ravel()
    u_obj = u_coord.ravel()[mask_flat]
    v_obj = v_coord.ravel()[mask_flat]
    z_obj = depth.ravel()[mask_flat]
    fx = intrinsic[0, 0]; fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]; cy = intrinsic[1, 2]
    x_obj = (u_obj - cx) * z_obj / fx
    y_obj = (v_obj - cy) * z_obj / fy
    pts_cam_obj = np.vstack((x_obj, y_obj, z_obj, np.ones_like(z_obj))).T
    pts_world_obj_h = (extrinsic @ pts_cam_obj.T).T
    pts_world_obj = pts_world_obj_h[:, :3]

    if pts_world_obj.shape[0] == 0:
        print("No object points found in mask. Exiting.")
        return
    centroid = np.mean(pts_world_obj, axis=0)
    # import ipdb; ipdb.set_trace()
    target_pos = centroid.copy()

    target_pos[2] += args.approach_height
    
    # target_pos[0] -= 0.05
    # target_pos[1] -= 0.05
    # import ipdb; ipdb.set_trace()
    quat = R.from_euler('xyz', [0, 0, 0]).as_quat()  # identity orientation
    # import ipdb; ipdb.set_trace()
    T_target = np.eye(4)
    R1 = R.from_euler('x', 90, degrees=True).as_matrix()
    R2 = R.from_euler('z', -90, degrees=True).as_matrix()
    R3 = R.from_euler('y', 90, degrees=True).as_matrix()
    R_all = R1 @ R2 @ R3

    # 构造目标位姿（相对于物体质心）
    delta_xyz = np.array([0.03138719,-0.13473819,-0.0649106])     # 平移偏移（m）
    delta_rpy = np.array([-0.9488417,1.0036732,2.626805])       # 旋转偏移（rad）
    T_target[:3, :3] = R.from_euler('xyz', delta_rpy).as_matrix() @ R_all
    T_target[:3, 3] = target_pos + delta_xyz

    # Build environment point cloud (mask == False) and downsample
    mask_inv = ~mask_flat
    u_env = u_coord.ravel()[mask_inv]
    v_env = v_coord.ravel()[mask_inv]
    z_env = depth.ravel()[mask_inv]
    x_env = (u_env - cx) * z_env / fx
    y_env = (v_env - cy) * z_env / fy
    pts_cam_env = np.vstack((x_env, y_env, z_env, np.ones_like(z_env))).T
    pts_world_env = (extrinsic @ pts_cam_env.T).T[:, :3]
    pcd_env = o3d.geometry.PointCloud()
    pcd_env.points = o3d.utility.Vector3dVector(pts_world_env)
    pcd_env = pcd_env.voxel_down_sample(voxel_size=args.voxel_size)
    pts_env_down = np.asarray(pcd_env.points)

    # Visualize point clouds and target frame
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(pts_world_obj)
    pcd_obj.paint_uniform_color([1.0, 0.0, 0.0])  # red for object
    pcd_env_vis = o3d.geometry.PointCloud()
    pcd_env_vis.points = o3d.utility.Vector3dVector(pts_env_down)
    pcd_env_vis.paint_uniform_color([0.7, 0.7, 0.7])  # gray for environment
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=target_pos)
    o3d.visualization.draw_geometries([pcd_env_vis, pcd_obj, frame],
                                      window_name="Point Cloud and Target Frame")

    # Set up planner and add environment
    planner_cfg = XARM6PlannerCfg(vis=False)
    xarm_planner = XARM6Planner(planner_cfg)
    # xarm_planner.mplib_add_point_cloud(pts_env_down)


    # Initialize robot wrapper and get current joint values
    arm = XArm6RealWorld(args.ip)
    current_qpos = arm.default_joint_values

    # Plan motion to target pose
    planning_result = xarm_planner.mplib_plan_screw(current_qpos, T_target)
    # import ipdb; ipdb.set_trace()
    if planning_result is None or planning_result.get("status", None) != "Success":
        print("Planning failed or no path found.")
        return

    waypoints = np.array(planning_result["position"])  # (N,6) array of joint waypoints
    # import ipdb; ipdb.set_trace()

    # Execute trajectory using wrapper (min-jerk interpolation)
    arm.set_joint_values_sequence(waypoints, planning_timestep=args.interp_timestep)
    # hand_pose_deg = [  # 16 个角度，单位=度
    #     3.1273, 2.8100, 4.7512, 3.8787,
    #     3.7473, 3.7872, 4.1290, 2.9599,
    #     4.3990, 3.9145, 4.7462, 2.7700,
    #     4.5365, -2.4671, 3.3332, 4.1665,
    # ]cylinder

    hand_pose_deg  = [
    3.11971569, 3.64052045, 3.75250040, 4.07596701,
    3.10568031, 3.66771221, 3.74179123, 4.06587338,
    3.78007387, 3.54894070, 4.80152910, 3.20546638,
    4.07135641, 2.83584486, 3.03249112, 4.06005710
]

    leap_sender = LeapSender(port="/dev/ttyUSB0")  # 如端口不同请改
    leap_sender.send(hand_pose_deg)
    time.sleep(0.5)
    # GRIP_CLOSE_POS = 400      # 0=全开，≈850=最紧，可按需调
    # GRIP_SPEED     = 1000     # 越大越快，≤5000
    # arm.arm.set_gripper_position(GRIP_CLOSE_POS, speed=GRIP_SPEED, wait=False)
    # print(">>> 轨迹执行完毕，夹爪已闭合")
 

    target_pos[2] += 0.2
    T_target[:3, 3] = target_pos

    planning_result2 = xarm_planner.mplib_plan_screw(waypoints[-1], T_target)
    # import ipdb; ipdb.set_trace()
    if planning_result2 is None or planning_result.get("status", None) != "Success":
        print("Planning failed or no path found.")
        return
    waypoints2 = np.array(planning_result2["position"])  # (N,6) array of joint waypoints
    arm.set_joint_values_sequence(waypoints2, planning_timestep=args.interp_timestep)

    hand_pose_deg = [
    3.11971569, 3.64052045, 3.75250040, 4.07596701,
    3.10568031, 3.66771221, 3.74179123, 4.06587338,
    3.78007387, 3.54894070, 4.80152910, 3.20546638,
    4.07135641, 2.83584486, 3.03249112, 4.06005710
]
    leap_sender = LeapSender(port="/dev/ttyUSB0")  # 如端口不同请改
    leap_sender.send(hand_pose_deg)
    time.sleep(50)


# target_pos = centroid.copy()
#     target_pos[2] += args.approach_height
#     target_pos[0] -= 0.05
#     # target_pos[1] -= 0.05
#     # import ipdb; ipdb.set_trace()
#     quat = R.from_euler('xyz', [0, 0, 0]).as_quat()  # identity orientation
#     # import ipdb; ipdb.set_trace()
#     T_target = np.eye(4)
#     T_target[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix()
#     T_target[:3, 3] = target_pos
  

if __name__ == "__main__":
    main()

