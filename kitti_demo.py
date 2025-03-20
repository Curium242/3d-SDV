import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from kitti_3d_pipeline import KITTI3DPipeline

# Demo settings
DATA_PATH = '/content/3d-SDV/KITTI_SAMPLE/RAW'  # Adjust to your data path
USE_PYKITTI = False  # Set to True to use pykitti loader instead of raw files

def demo_object_detection():
    """Demo the object detection and LiDAR projection capabilities"""
    print("Initializing KITTI 3D pipeline for object detection...")

    if USE_PYKITTI:
        # Using pykitti loader
        pipeline = KITTI3DPipeline(
            data_path='/content',  # Base path for pykitti
            date='2011_10_03',
            drive='0047',
            use_raw_files=False    # Use pykitti loader
        )
    else:
        # Using raw files
        pipeline = KITTI3DPipeline(
            data_path=DATA_PATH,
            date='2011_09_26',
            drive='0009',
            use_raw_files=True     # Use raw file structure
        )

    # Process a few frames
    print("Processing frames for object detection...")
    for frame_idx in range(0, 30, 10):
        print(f"\nProcessing frame {frame_idx}...")
        fig, drive_map = pipeline.visualize_frame(frame_idx)

        # Display the results
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Frame {frame_idx}")
        plt.subplot(121)
        plt.imshow(plt.imread(f"frame_{frame_idx}_detection.png"))
        plt.title("Object Detection with Depth")

        plt.subplot(122)
        plt.imshow(plt.imread(f"frame_{frame_idx}_lidar.png"))
        plt.title("LiDAR Projection")

        plt.tight_layout()
        plt.show()

        if drive_map:
            # Display the map if available
            drive_map.save(f"frame_{frame_idx}_map.html")
            print(f"Map saved to frame_{frame_idx}_map.html")

def demo_point_cloud_reconstruction():
    """Demo the point cloud reconstruction capabilities"""
    print("\nInitializing KITTI 3D pipeline for point cloud reconstruction...")

    if USE_PYKITTI:
        # Using pykitti loader
        pipeline = KITTI3DPipeline(
            data_path='/content',  # Base path for pykitti
            date='2011_10_03',
            drive='0047',
            use_raw_files=False    # Use pykitti loader
        )

        # Display the trajectory
        print("Visualizing vehicle trajectory...")
        fig = pipeline.visualize_trajectory(start_frame=0, end_frame=50)
        plt.show()

        # Build and save the point cloud
        print("Building point cloud from 50 frames...")
        pipeline.build_point_cloud(start_frame=0, end_frame=50)
        ply_path = pipeline.save_point_cloud('pykitti_scene.ply')
    else:
        # Using raw files
        pipeline = KITTI3DPipeline(
            data_path=DATA_PATH,
            date='2011_09_26',
            drive='0009',
            use_raw_files=True     # Use raw file structure
        )

        # Display the trajectory
        print("Visualizing vehicle trajectory...")
        fig = pipeline.visualize_trajectory(start_frame=0, end_frame=20)
        plt.show()

        # Build and save the point cloud
        print("Building point cloud from 20 frames...")
        pipeline.build_point_cloud(start_frame=0, end_frame=20)
        ply_path = pipeline.save_point_cloud('raw_scene.ply')

    print(f"Point cloud saved to {ply_path}")
    print("You can view this PLY file with any 3D viewer or tool that supports PLY format.")

def demo_integrated_pipeline():
    """Demo the complete integrated pipeline"""
    print("\nRunning the complete integrated pipeline...")

    # Initialize pipeline
    pipeline = KITTI3DPipeline(
        data_path=DATA_PATH,
        date='2011_09_26',
        drive='0009',
        use_raw_files=True
    )

    # 1. Process a specific frame with object detection and LiDAR projection
    frame_idx = 10
    print(f"\nStep 1: Processing frame {frame_idx} with object detection...")
    frame_data = pipeline.process_frame(frame_idx)

    # Display the results
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(frame_data['image_with_bboxes'])
    ax[0].set_title("Object Detection with Depth")
    ax[1].imshow(frame_data['image_with_velo'])
    ax[1].set_title("LiDAR Projection")
    plt.tight_layout()
    plt.show()

    # 2. Show 3D positions of detected objects
    print("\nStep 2: Displaying detected objects with 3D positions...")

    # Extract bounding box information (class, confidence, bbox, uvz)
    bboxes = frame_data['bboxes']
    for i, bbox in enumerate(bboxes):
        class_id = int(bbox[5])
        conf = bbox[4]
        uvz = bbox[-3:]
        print(f"Object {i+1}: Class {class_id}, Confidence {conf:.2f}, Depth {uvz[2]:.2f} m")

    # 3. Visualize trajectory
    print("\nStep 3: Visualizing vehicle trajectory...")
    fig = pipeline.visualize_trajectory(start_frame=0, end_frame=20)
    plt.show()

    # 4. Build and save the point cloud
    print("\nStep 4: Building colored 3D point cloud from multiple frames...")
    pipeline.build_point_cloud(start_frame=0, end_frame=20)
    ply_path = pipeline.save_point_cloud('integrated_scene.ply')
    print(f"Point cloud saved to {ply_path}")

    print("\nIntegrated pipeline demo completed successfully!")

if __name__ == "__main__":
    # Uncomment the demo you want to run
    #demo_object_detection()
    #demo_point_cloud_reconstruction()
    demo_integrated_pipeline()

    print("\nAll demos completed!")
