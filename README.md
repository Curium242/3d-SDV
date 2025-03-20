# KITTI 3D Pipeline: Object Detection and Point Cloud Reconstruction

This integrated pipeline combines object detection and 3D scene reconstruction capabilities to process KITTI autonomous driving datasets. The pipeline provides a unified interface for both raw KITTI files and the PyKITTI loader, with optimizations for edge device deployment.

## Features

- **Object Detection**: Uses YOLOv5 to detect objects in camera images
- **3D Localization**: Projects LiDAR data onto images and associates depth with detected objects
- **Coordinate Transformation**: Converts between multiple coordinate frames (camera, LiDAR, IMU, geodetic)
- **Point Cloud Generation**: Creates colored 3D point clouds from camera and LiDAR data
- **Trajectory Visualization**: Plots vehicle trajectory from OXTS/IMU data
- **Map Integration**: Visualizes detected objects on interactive maps
- **Edge-Device Optimization**: Optimized for deployment on resource-constrained edge computing devices

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/kitti-3d-pipeline.git
cd kitti-3d-pipeline
```

2. Install dependencies:
```bash
pip install numpy opencv-python matplotlib torch torchvision pillow folium pymap3d pykitti scikit-learn plyfile
```

3. Install YOLOv5:
```bash
pip install -U ultralytics
```

## File Structure

- `kitti_3d_pipeline.py`: Main pipeline class that integrates all functionality
- `kitti_utils.py`: Utility functions for KITTI data processing
- `kitti_demo.py`: Demonstration script showing how to use the pipeline
- `README.md`: Documentation

## Usage

### Basic Usage

```python
from kitti_3d_pipeline import KITTI3DPipeline

# Initialize the pipeline with raw KITTI files
pipeline = KITTI3DPipeline(
    data_path='/path/to/kitti/data',
    date='2011_09_26',
    drive='0009',
    use_raw_files=True  # Set to False to use pykitti loader
)

# Process a single frame
frame_idx = 10
outputs = pipeline.process_frame(frame_idx)

# Build a point cloud from multiple frames
pipeline.build_point_cloud(start_frame=0, end_frame=20)
pipeline.save_point_cloud('scene.ply')

# Visualize the vehicle trajectory
pipeline.visualize_trajectory()
```

### Run Demo Script

```bash
python kitti_demo.py
```

## Components

### 1. Object Detection Module

The object detection component uses YOLOv5 to identify objects in camera images. The pipeline supports dynamic loading of different YOLOv5 models based on resource constraints, making it suitable for edge deployment.

```python
# Using a lighter model for edge devices
pipeline = KITTI3DPipeline(yolo_model='yolov5s', conf_thres=0.25)

# For more accuracy when resources permit
pipeline = KITTI3DPipeline(yolo_model='yolov5x', conf_thres=0.4)
```

### 2. LiDAR Projection

The pipeline projects LiDAR points onto camera images to visualize depth information and associate it with detected objects.

```python
# Process a frame with LiDAR projection
outputs = pipeline.process_frame(frame_idx, draw_lidar=True)

# Access the LiDAR projection
lidar_image = outputs['image_with_velo']
```

### 3. Point Cloud Reconstruction

Building a colored 3D point cloud by fusing camera and LiDAR data across multiple frames.

```python
# Build point cloud from 20 frames, filtering out points closer than 5 meters
pipeline.build_point_cloud(start_frame=0, end_frame=20, filter_min_x=5.0)

# Save the point cloud
pipeline.save_point_cloud('output.ply')
```

### 4. Coordinate Transformation

The pipeline handles multiple coordinate transformations:
- Camera to LiDAR
- LiDAR to IMU
- IMU to Geodetic (latitude, longitude, altitude)

```python
# Get geodetic coordinates of detected objects
frame_data = pipeline.process_frame(frame_idx)
geodetic_positions = frame_data['geodetic']
```

## KITTI Dataset Structure

The pipeline supports two methods of accessing KITTI data:

### Raw File Structure

```
/path/to/kitti/data/
├── 2011_09_26/
│   ├── 2011_09_26_drive_0009_sync/
│   │   ├── image_02/data/*.png
│   │   ├── image_03/data/*.png
│   │   ├── velodyne_points/data/*.bin
│   │   └── oxts/data/*.txt
│   ├── calib_cam_to_cam.txt
│   ├── calib_velo_to_cam.txt
│   └── calib_imu_to_velo.txt
```

### PyKITTI Loader

The pipeline can also use the pykitti library to load data:

```python
pipeline = KITTI3DPipeline(
    data_path='/path/to/kitti',
    date='2011_09_26',
    drive='0009',
    use_raw_files=False  # Use pykitti loader
)
```

## Integration with Existing Code

This pipeline integrates two previously separate workflows:
1. YOLO-based object detection with LiDAR projection
2. 3D scene reconstruction from RGB-LiDAR fusion

The integration allows for a more streamlined workflow that combines the strengths of both approaches.

## Performance Considerations

- The pipeline includes options for downsampling LiDAR points to improve performance on edge devices
- Ground plane removal is optional and can be disabled for faster processing
- Multi-threaded file loading is used to speed up I/O operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This pipeline uses the KITTI dataset: http://www.cvlibs.net/datasets/kitti/
- YOLOv5 implementation from Ultralytics: https://github.com/ultralytics/yolov5
- PyKITTI library: https://github.com/utiasSTARS/pykitti
