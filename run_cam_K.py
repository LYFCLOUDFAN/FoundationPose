import numpy as np, os
fx, fy, cx, cy = 605.5433349609375, 604.828125, 331.627685546875, 243.877685546875   # 改成你的数值
K = np.array([[fx, 0, cx],
              [0,  fy, cy],
              [0,  0,  1]], dtype=np.float32)
np.savetxt("/home/xzx/Dro/realworld/realworld-main/demo_rubic_cube/cam_K.txt", K)
