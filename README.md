# Dangerous Driving Scenario Reconstruction using Carla and 3D Gaussian Splatting (3DGS)


This project aims to simulate dangerous driving scenarios in the Carla simulator, such as pedestrian avoidance, overtaking, and lane-cutting. The collected multi-view data, including RGB camera and LiDAR sensor data, will be used to reconstruct these scenarios in 3D using S3Gaussian, with the objective of extending S3Gaussian capabilities to work with Carla datasets, enhancing the robustness of autonomous driving systems.

### Demo

### Installation
#### Clone this repository

```
git clone https://github.com/ZyuanW/carla_gs.git --recursive
```
#### Environmental Setup

```
# Set conda environment
conda create -n carla-s3 python=3.8
conda activate carla-s3

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### Preparing Carla Dataset
Check /carla_gs/carla_data_collection   
You can also download from [Here](https://drive.google.com/drive/folders/1rkUZolT7OMYxU50GeOyr6BK7Q7omFd_j?usp=sharing) for a quick start


### Quick Start
Using /carla_gs/S3_carla_train.ipynb for quick start

### Training
For training first clip (eg. 0-50 frames), run 

```
python train.py -s $data_dir --port 6017 --expname "carla" --model_path $model_path 
```
If you want to try novel view  synthesis, use 
```
--configs "arguments/nvs.py"
```

### Evaluation and Visualization

You can visualize and eval a checkpoints follow:
```python
python train.py -s $data_dir --port 6017 --expname "carla" --start_checkpoint "$ckpt_dir/chkpnt_fine_50000.pth" --model_path $model_path --eval_only
```
 
## Acknowledgments
Our code is based on [S3Gaussian](https://github.com/nnanhuang/S3Gaussian) and [StreetGaussians](https://github.com/zju3dv/street_gaussians)