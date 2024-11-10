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

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### Collecting Carla Dataset



 
