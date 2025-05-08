import pyrealsense2 as rs, cv2, numpy as np, os, argparse
from segment_anything import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--n_frames", type=int, default=200)
parser.add_argument("--ckpt", default="sam_vit_b.pth")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/rgb", exist_ok=True)
os.makedirs(f"{args.out_dir}/depth", exist_ok=True)
os.makedirs(f"{args.out_dir}/masks", exist_ok=True)

# ---------------- 相机初始化 ----------------
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
profile = pipe.start(cfg)
align   = rs.align(rs.stream.color)

# ---------------- 保存内参 ----------------
K_PATH = "/home/xzx/Dro/third_party/xarm6/xarm6_interface/calib/calib_imgs/K.npy"

assert os.path.isfile(K_PATH), f"找不到标定内参文件: {K_PATH}"
K = np.load(K_PATH).astype(np.float32)           # 直接加载 3×3 numpy 数组
np.savetxt(f"{args.out_dir}/cam_K.txt", K)       # 仍按原流程保存到 out_dir

print(f"[INFO] 已使用离线标定内参 K: \n{K}")

  # 丢掉几帧让自动曝光／白平衡收敛
for _ in range(60):
    pipe.wait_for_frames()


# ---------------- 捕获首帧 ----------------
frames  = align.process(pipe.wait_for_frames())
color   = np.asanyarray(frames.get_color_frame().get_data())
depth0  = np.asanyarray(frames.get_depth_frame().get_data())

# ---------------- 加载 SAM ----------------
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(device)
predictor = SamPredictor(sam)
predictor.set_image(color[..., ::-1])        # BGR→RGB

# ---------------- 交互点选 ----------------
pts, labels = [], []   # 1=前景, 0=背景

def on_mouse(event, x, y, flags, _):
    global pts, labels, color
    if event == cv2.EVENT_LBUTTONDOWN:   # 前景点
        pts.append([x, y]); labels.append(1)
    if event == cv2.EVENT_RBUTTONDOWN:   # 背景点
        pts.append([x, y]); labels.append(0)

cv2.namedWindow("SAM"); cv2.setMouseCallback("SAM", on_mouse)

mask_vis = color.copy()
while True:
    disp = mask_vis.copy()
    for (x, y), lab in zip(pts, labels):
        c = (0,255,0) if lab==1 else (0,0,255)
        cv2.circle(disp, (x,y), 4, c, -1)
    cv2.imshow("SAM", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):                 # 重置
        pts, labels, mask_vis = [], [], color.copy()
        predictor.reset_image()
        predictor.set_image(color[..., ::-1])
    if k == ord('p') and pts:         # 预测
        masks, _, _ = predictor.predict(
            np.array(pts), np.array(labels), multimask_output=False)
        mask = masks[0].astype(np.uint8)*255
        mask_vis = color.copy()
        mask_vis[mask==0] = mask_vis[mask==0]*0.3
    if k == 32 and 'mask' in locals():  # Space 保存
        break
    if k == 27:  # ESC 退出
        pipe.stop(); exit()

# ---------------- 保存首帧 ----------------
cv2.imwrite(f"{args.out_dir}/masks/000000.png", mask)
cv2.imwrite(f"{args.out_dir}/rgb/000000.png", color)
cv2.imwrite(f"{args.out_dir}/depth/000000.png", depth0)
print("首帧保存完成，开始连续录制…  (ESC 停止)")

# ---------------- 录制余下帧 ----------------
idx = 1
while idx < args.n_frames:
    frames = align.process(pipe.wait_for_frames())
    color  = np.asanyarray(frames.get_color_frame().get_data())
    depth  = np.asanyarray(frames.get_depth_frame().get_data())
    cv2.imwrite(f"{args.out_dir}/rgb/{idx:06}.png", color)
    cv2.imwrite(f"{args.out_dir}/depth/{idx:06}.png", depth)
    cv2.imshow("RGB", color)
    if cv2.waitKey(1)==27: break
    idx += 1

pipe.stop()
print("Finished!")
