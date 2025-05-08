#!/usr/bin/env python3
import sys
sys.path.append('/home/xzx/Dro/third_party/xarm6')
sys.path.append('/home/xzx/Dro/third_party/xarm6/xarm_interface')
import time
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 控制接口
from xarm6_interface.arm_rw import XArm6RealWorld
from xarm6_interface.arm_mplib import XARM6Planner, XARM6PlannerCfg
from leap_hand_utils.dynamixel_client import DynamixelClient

from urdf_mapping import map_urdf_to_real

class LeapSender:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 4_000_000,
        kp: int = 600,
        kd: int = 200,
        ki: int = 0,
        curr_lim: int = 350,
    ):
        self.ids = list(range(16))
        self.dxl = DynamixelClient(self.ids, port, baud)
        self.dxl.connect()

        self.dxl.sync_write(self.ids, np.ones(16) * 5, 11, 1)
        self.dxl.set_torque_enabled(self.ids, True)
        self.dxl.sync_write(self.ids, np.ones(16) * kp, 84, 2)
        self.dxl.sync_write([0, 4, 8], np.ones(3) * kp * 0.75, 84, 2)
        self.dxl.sync_write(self.ids, np.ones(16) * kd, 80, 2)
        self.dxl.sync_write([0, 4, 8], np.ones(3) * kd * 0.75, 80, 2)
        self.dxl.sync_write(self.ids, np.ones(16) * ki, 82, 2)
        self.dxl.sync_write(self.ids, np.ones(16) * curr_lim, 102, 2)

    def send(self, pose16):
        self.dxl.write_desired_pos(self.ids, np.asarray(pose16, dtype=float))


def main():
    pkl_path = Path("/home/xzx/Dro/FoundationPose/res_diffuser_dro_3hand_leaphand_epoch_200.pkl")

    # === 读取 predict_q ===
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    predict_q = data['results'][0]['predict_q']  # shape (22,)
    xyz = predict_q[:3]
    rpy = predict_q[3:6]  # in degrees
    leaphand_urdf = predict_q[6:]
    leaphand_real = map_urdf_to_real(leaphand_urdf)

    # === 初始化机器人和手 ===
    arm = XArm6RealWorld("192.168.1.230")  # 根据你的实际 IP 设置
    planner = XARM6Planner(XARM6PlannerCfg(vis=False))
    hand = LeapSender(port="/dev/ttyUSB0")

    # === 先发送手部动作，避免延迟 ===
    hand.send(leaphand_real)
    time.sleep(0.3)

    # === 构建目标姿态变换 ===
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy, degrees=True).as_matrix()
    T[:3, 3] = xyz

    # === 获取当前关节角并规划 ===
    q_cur = arm.get_joint_values()
    plan = planner.mplib_plan_screw(q_cur, T)
    if plan is None or plan.get("status") != "Success":
        print("[ERROR] Failed to plan trajectory.")
        return

    # === 执行轨迹并发送手部动作 ===
    waypoints = np.asarray(plan["position"])
    arm.set_joint_values_sequence(waypoints, planning_timestep=0.05)
    hand.send(leaphand_real)

    print("[DONE] Executed predict_q successfully.")


if __name__ == "__main__":
    main()
