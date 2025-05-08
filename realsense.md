第一步：git clone https://github.com/NVlabs/FoundationPose #follow the github instructions.
第二步：Download all network weights from here and put them under the folder weights/. For the refiner, you will need 2023-10-28-18-33-37. For scorer, you will need 2024-01-11-20-02-45.

Download demo data and extract them under the folder demo_data/
# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"#这一步骤注意要找到eigen3 path 另外需要在FoundationPose/bundlesdf/mycuda/setup.py line37下面 添加"/home/xzx/anaconda3/envs/foundationpose/include/eigen3",

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh


#下载sam_vit_b
https://github.com/facebookresearch/segment-anything

pip install opencv-python==4.8.1.78 git+https://github.com/facebookresearch/segment-anything.git

#下载realworld文件夹
https://github.com/Robot-Experiment-Lab/realworld

#realsense
pip install realsense2
https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide

#xarm
https://github.com/xArm-Developer/xArm-Python-SDK

获得depth,masks,rgb
python save_realsense_sequence.py   --out_dir /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube   --n_frames 300 #更改为自己的路径
python save_realsense_sequence.py   --out_dir /home/xzx/Dro/realworld/realworld-main/demo_cylinder_bottle   --n_frames 300 #更改为自己的路径
python save_realsense_sequence.py   --out_dir /home/xzx/Dro/realworld/realworld-main/demo_mug   --n_frames 300


python run_demo.py   --mesh_file /home/xzx/Dro/realworld/realworld-main/object_mesh/rubic_cube/rubic_cube.obj   --test_scene_dir /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube   --est_refine_iter 5   --track_refine_iter 2 --debug 2   --debug_dir /home/xzx/Dro/FoundationPose/output#更改为自己的路径

conda activate foundationpose
python save_pose.py \
  --data_dir /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube \
  --mesh_file /home/xzx/Dro/realworld/realworld-main/object_mesh/rubic_cube/rubic_cube.obj \
  --out_file   /home/xzx/Dro/pose/pose_cam_obj.npz

python save_pose.py \
  --data_dir /home/xzx/Dro/realworld/realworld-main/demo_cylinder_bottle \
  --mesh_file /home/xzx/Dro/realworld/realworld-main/Obj_mesh/cylinder_bottle/cylinder_bottle.obj \
  --out_file   /home/xzx/Dro/pose/pose_cam_cylinder_bottle.npz

python save_pose.py \
  --data_dir /home/xzx/Dro/realworld/realworld-main/demo_mug \
  --mesh_file /home/xzx/Dro/realworld/realworld-main/Obj_mesh/mug/mug.obj \
  --out_file   /home/xzx/Dro/pose/pose_cam_mug.npz


conda activate xarm

python move_to_pose.py   --intrinsic  /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube/cam_K.txt   --extrinsic  /home/xzx/Dro/third_party/xarm6/data/camera/317222073552/0307_excalib_capture00/optimized_X_BaseCamera.npy   --depth      /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube/depth/000000.png   --mask       /home/xzx/Dro/realworld/realworld-main/demo_rubic_cube/masks/000000.png   --ip   192.168.1.208 

python move_to_pose.py   --intrinsic  /home/xzx/Dro/realworld/realworld-main/demo_cylinder_bottle/cam_K.txt   --extrinsic  /home/xzx/Dro/third_party/xarm6/data/camera/317222073552/0307_excalib_capture00/optimized_X_BaseCamera.npy   --depth      /home/xzx/Dro/realworld/realworld-main/demo_cylinder_bottle/depth/000000.png   --mask       /home/xzx/Dro/realworld/realworld-main/demo_cylinder_bottle/masks/000000.png   --ip   192.168.1.208 

python move_to_pose.py   --intrinsic  /home/xzx/Dro/realworld/realworld-main/demo_mug/cam_K.txt   --extrinsic  /home/xzx/Dro/third_party/xarm6/data/camera/317222073552/0307_excalib_capture00/optimized_X_BaseCamera.npy   --depth      /home/xzx/Dro/realworld/realworld-main/demo_mug/depth/000000.png   --mask       /home/xzx/Dro/realworld/realworld-main/demo_mug/masks/000000.png   --ip   192.168.1.208 