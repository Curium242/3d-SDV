import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import folium
import pymap3d as pm
import pykitti
from PIL import Image as PILImage
import plyfile as ply
from glob import glob
from time import time
from concurrent.futures import ThreadPoolExecutor

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["image.interpolation"] = 'nearest'

# Import utils (we'll create these files)
from kitti_utils import (get_rigid_transformation, get_oxts, project_velobin2uvz,
                        transform_uvz, draw_velo_on_image)

class KITTI3DPipeline:
    """
    A comprehensive pipeline for KITTI dataset processing that combines:
    - Object detection with YOLO
    - LiDAR projection and depth association
    - 3D point cloud reconstruction and colorization
    - Coordinate transformations and visualization
    """

    def __init__(self, data_path=None, date=None, drive=None,
                 yolo_model='yolov5s', conf_thres=0.25, iou_thres=0.25,
                 use_raw_files=True):
        """
        Initialize the KITTI processing pipeline.

        Args:
            data_path: Path to KITTI dataset
            date: Date string for KITTI raw dataset (e.g., '2011_09_26')
            drive: Drive string for KITTI raw dataset (e.g., '0009')
            yolo_model: YOLO model variant to use
            conf_thres: Confidence threshold for YOLO detection
            iou_thres: IoU threshold for YOLO detection
            use_raw_files: Whether to use raw file structure (True) or pykitti loader (False)
        """
        self.data_path = data_path
        self.date = date
        self.drive = drive
        self.use_raw_files = use_raw_files
        self.calibration = {}
        self.paths = {}

        # Initialize the pipeline
        if data_path:
            self._setup_data_paths()
            self._load_calibration()

        # Load YOLO model
        self.detector = self._load_yolo_model(yolo_model, conf_thres, iou_thres)

        # For storing processed results
        self.detections = []
        self.point_cloud = None
        self.trajectory = None

    def _setup_data_paths(self):
        """Set up file paths based on the dataset structure."""
        if self.use_raw_files:
            # Raw file structure paths
            drive_path = f"{self.data_path}/{self.date}/{self.date}_drive_{self.drive}_sync"
            self.paths = self._load_raw_paths(drive_path)
            # Calibration file paths
            self.calib_paths = {
                'cam_to_cam': f"{self.data_path}/{self.date}/calib_cam_to_cam.txt",
                'velo_to_cam': f"{self.data_path}/{self.date}/calib_velo_to_cam.txt",
                'imu_to_velo': f"{self.data_path}/{self.date}/calib_imu_to_velo.txt"
            }
        else:
            # Using pykitti for data loading
            self.kitti_data = pykitti.raw(self.data_path, self.date, self.drive)
            # We'll handle pykitti paths later as needed

    def _load_raw_paths(self, drive_path):
        """Load file paths concurrently to speed up I/O operations on edge devices."""
        with ThreadPoolExecutor() as executor:
            futures = {
                "left_images": executor.submit(sorted, glob(os.path.join(drive_path, 'image_02/data/*.png'))),
                "right_images": executor.submit(sorted, glob(os.path.join(drive_path, 'image_03/data/*.png'))),
                "lidar": executor.submit(sorted, glob(os.path.join(drive_path, 'velodyne_points/data/*.bin'))),
                "oxts": executor.submit(sorted, glob(os.path.join(drive_path, 'oxts/data/*.txt')))
            }
            return {key: future.result() for key, future in futures.items()}

    def _load_calibration(self):
        """Load and process calibration data."""
        if self.use_raw_files:
            self.calibration = self._parse_camera_calib(
                self.calib_paths['cam_to_cam'],
                self.calib_paths['velo_to_cam'],
                self.calib_paths['imu_to_velo']
            )
        else:
            # Get calibration from pykitti
            calib = self.kitti_data.calib
            # Process calibration from pykitti (implementation to follow)
            self.calibration = self._process_pykitti_calib(calib)

    def _parse_camera_calib(self, cam_to_cam_path, velo_to_cam_path, imu_to_velo_path):
        """
        Parse and compute required transformation matrices from KITTI calibration files.
        """
        with open(cam_to_cam_path, 'r') as f:
            calib = f.readlines()

        # Projection matrix for rectified camera 2 (3x4)
        P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3, 4))

        # Rectification from camera 0 -> rectified reference (convert 3x3 to 4x4)
        R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3))
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0, 0, 0], axis=0)
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0, 0, 0, 1], axis=1)

        # Rigid transform from camera 0 -> camera 2 (build a proper 4x4 matrix)
        R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3, 3))
        t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3, 1))
        T_ref0_ref2 = np.vstack((np.hstack((R_2, t_2)), np.array([0, 0, 0, 1])))

        # Read external transformations
        T_velo_ref0 = get_rigid_transformation(velo_to_cam_path)
        T_imu_velo = get_rigid_transformation(imu_to_velo_path)

        # Compute transforms
        T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0
        T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0, 0, 0, 1], axis=0))
        T_imu_cam2 = T_velo_cam2 @ T_imu_velo
        T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0, 0, 0, 1], axis=0))

        return {
            'P_rect2_cam2': P_rect2_cam2,
            'R_ref0_rect2': R_ref0_rect2,
            'T_ref0_ref2': T_ref0_ref2,
            'T_velo_ref0': T_velo_ref0,
            'T_velo_cam2': T_velo_cam2,
            'T_cam2_velo': T_cam2_velo,
            'T_imu_cam2': T_imu_cam2,
            'T_cam2_imu': T_cam2_imu
        }

    def _process_pykitti_calib(self, calib):
        """Process calibration data from pykitti."""
        # Extract needed calibration matrices from pykitti
        P_rect2_cam2 = calib.P_rect_20  # Projection matrix for camera 2
        T_velo_cam2 = calib.P_rect_20 @ calib.T_cam2_velo  # LiDAR to camera 2 projection
        T_cam2_velo = calib.T_cam2_velo  # Camera 2 to LiDAR
        T_imu_velo = np.linalg.inv(calib.T_velo_imu)  # IMU to LiDAR
        T_imu_cam2 = T_velo_cam2 @ T_imu_velo  # IMU to camera 2
        T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0, 0, 0, 1], axis=0))  # Camera 2 to IMU

        return {
            'P_rect2_cam2': P_rect2_cam2,
            'T_velo_cam2': T_velo_cam2,
            'T_cam2_velo': T_cam2_velo,
            'T_imu_velo': T_imu_velo,
            'T_imu_cam2': T_imu_cam2,
            'T_cam2_imu': T_cam2_imu,
            'K_cam2': calib.K_cam2  # Intrinsic camera matrix
        }

    def _load_yolo_model(self, model_name='yolov5s', conf_thres=0.25, iou_thres=0.25):
        """
        Loads a lightweight YOLO model for edge inference.
        YOLOv5s is chosen for its speed and small footprint.
        """
        model = torch.hub.load('ultralytics/yolov5', model_name, device='cpu')
        model.conf = conf_thres
        model.iou = iou_thres
        return model

    def get_uvz_centers(self, image, velo_uvz, bboxes, draw=True):
        """
        For each bounding box, find its center (u,v) in the image,
        then associate it to the nearest LiDAR point (using its depth value).
        Appends (u, v, depth) to each bounding box.
        """
        u, v, z = velo_uvz  # (N,)

        bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
        bboxes_out[:, :bboxes.shape[1]] = bboxes

        for i, bbox in enumerate(bboxes):
            pt1 = torch.round(bbox[0:2]).to(torch.int).numpy()  # (x1, y1)
            pt2 = torch.round(bbox[2:4]).to(torch.int).numpy()  # (x2, y2)
            obj_u_center = (pt1[0] + pt2[0]) / 2
            obj_v_center = (pt1[1] + pt2[1]) / 2

            # Find nearest LiDAR point (using Euclidean distance in image plane)
            delta = np.array([v - obj_v_center, u - obj_u_center])
            dist = np.linalg.norm(delta, axis=0)
            min_loc = np.argmin(dist)

            velo_depth = z[min_loc]
            uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])
            bboxes_out[i, -3:] = uvz_location

            if draw:
                center_pt = (int(np.round(obj_u_center)), int(np.round(obj_v_center)))
                cv2.putText(image,
                            f'{velo_depth:.2f} m',
                            center_pt,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 2, cv2.LINE_AA)
        return bboxes_out

    def imu_to_geodetic(self, x, y, z, lat0, lon0, alt0, heading0):
        """
        Convert local IMU coordinates to geodetic coordinates using an initial reference.
        """
        rng = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
        el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90.0
        lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)
        return np.column_stack((lla[0], lla[1], lla[2]))

    def detect_objects(self, image, bin_path, draw_boxes=True, draw_depth=True):
        """
        Run YOLO detection, project LiDAR to image, and associate object centers with depth.
        Returns bounding boxes appended with (u_center, v_center, depth) and the LiDAR projection.
        """
        detections = self.detector(image)
        if draw_boxes:
            detections.render()  # in-place drawing

        bboxes = detections.xyxy[0].cpu().clone()
        velo_uvz = project_velobin2uvz(bin_path, self.calibration['T_velo_cam2'], image, remove_plane=True)
        bboxes_out = self.get_uvz_centers(image, velo_uvz, bboxes, draw=draw_depth)
        return bboxes_out, velo_uvz

    def process_frame(self, frame_idx, draw_boxes=True, draw_depth=True, draw_lidar=True):
        """
        Process a single frame with object detection and LiDAR projection.
        Returns the frame data, detected objects, and visualization images.
        """
        # Load frame data
        if self.use_raw_files:
            left_image = cv2.cvtColor(cv2.imread(self.paths['left_images'][frame_idx]), cv2.COLOR_BGR2RGB)
            bin_path = self.paths['lidar'][frame_idx]
            oxts_frame = get_oxts(self.paths['oxts'][frame_idx])
        else:
            left_image = np.array(self.kitti_data.get_cam2(frame_idx))
            # Convert to RGB if needed
            if len(left_image.shape) == 3 and left_image.shape[2] == 3:
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            oxts_frame = self.kitti_data.oxts[frame_idx]
            bin_data = self.kitti_data.get_velo(frame_idx)  # This returns numpy array
            # Save to a temporary bin file to use with project_velobin2uvz
            bin_path = f"temp_lidar_{frame_idx}.bin"
            with open(bin_path, 'wb') as f:
                bin_data.astype(np.float32).tofile(f)

        # Detect objects and get LiDAR projection
        bboxes, velo_uvz = self.detect_objects(left_image, bin_path, draw_boxes, draw_depth)

        # Transform bounding-box centers from camera to IMU to geodetic coordinates
        uvz_cam = bboxes[:, -3:]
        imu_xyz = transform_uvz(uvz_cam, self.calibration['T_cam2_imu'])  # Nx3 in IMU coords

        lat0, lon0, alt0 = oxts_frame[0], oxts_frame[1], oxts_frame[2]
        heading0 = oxts_frame[5]  # yaw in radians
        lla = self.imu_to_geodetic(imu_xyz[:, 0], imu_xyz[:, 1], imu_xyz[:, 2],
                              lat0, lon0, alt0, heading0)

        # Create output images
        outputs = {
            'image': left_image,
            'image_with_bboxes': left_image,  # Already modified by YOLO render if draw_boxes=True
            'image_with_velo': left_image.copy(),
            'bboxes': bboxes,
            'velo_uvz': velo_uvz,
            'geodetic': lla,
            'oxts': oxts_frame
        }

        # Draw LiDAR points if requested
        if draw_lidar:
            outputs['image_with_velo'] = draw_velo_on_image(velo_uvz, outputs['image_with_velo'])

        # Remove temporary bin file if created
        if not self.use_raw_files and os.path.exists(bin_path):
            os.remove(bin_path)

        return outputs

    def build_point_cloud(self, start_frame=0, end_frame=None, filter_min_x=5.0):
        """
        Build a colored point cloud from multiple frames by projecting LiDAR points
        into world coordinates and extracting colors from camera images.
        """
        if not self.use_raw_files:
            # Use pykitti data
            return self._build_point_cloud_pykitti(start_frame, end_frame, filter_min_x)
        else:
            # Use raw files
            return self._build_point_cloud_raw(start_frame, end_frame, filter_min_x)

    def _build_point_cloud_pykitti(self, start_frame=0, end_frame=None, filter_min_x=5.0):
        """Build point cloud using pykitti data."""
        if end_frame is None:
            end_frame = len(self.kitti_data.timestamps)

        # Store all points and colors
        all_inside_points = []

        # Get calibration
        calibre = np.linalg.inv(self.kitti_data.calib.T_velo_imu)  # IMU to LiDAR
        calibre_camera = self.kitti_data.calib.T_cam2_velo  # LiDAR to camera
        Kw = self.kitti_data.calib.K_cam2  # Camera intrinsic matrix

        # Extract position matrices (IMU poses)
        pos_matrices = [pose[1] for pose in self.kitti_data.oxts[start_frame:end_frame]]

        # Process each frame
        for i in range(start_frame, end_frame):
            rel_idx = i - start_frame

            # Get LiDAR points
            lidar_pts = np.array(self.kitti_data.get_velo(i))

            # Filter points
            lidar_pts = lidar_pts[(lidar_pts[:, 0] >= filter_min_x)]  # Filter points with x >= filter_min_x
            if len(lidar_pts) == 0:
                continue

            # Convert to homogeneous coordinates
            lidar_pts_hom = np.hstack((lidar_pts[:, :3], np.ones((lidar_pts.shape[0], 1))))

            # Transform LiDAR points into world coordinates
            new_points = np.array((pos_matrices[rel_idx] @ (calibre @ lidar_pts_hom.T)))
            new_points[3, :] = 1  # Normalize homogeneous coordinates

            # Project LiDAR points into the camera image plane
            X_2D = np.array((Kw @ (calibre_camera[0:3, :] @ lidar_pts_hom.T)))

            # Transpose for compatibility
            new_points = new_points.T

            # Normalize by depth (z-axis)
            X_2D = X_2D / X_2D[2, :]
            X_2D = X_2D[0:2].T

            # Load the corresponding image as a NumPy array
            Image = np.array(self.kitti_data.get_cam2(i))

            # Mask points that fall within the image bounds
            mask = ((X_2D[:, 0] > 0) & (X_2D[:, 0] < Image.shape[1]) &
                    (X_2D[:, 1] > 0) & (X_2D[:, 1] < Image.shape[0]))

            # Filter points and corresponding 2D positions
            new_points_inside = new_points[mask]
            X_2D_inside = X_2D[mask]

            # Convert 2D positions to integer pixel indices
            pixel_position = X_2D_inside.astype(int)

            # Retrieve RGB colors for each point
            color_rgb = []
            for j in range(len(pixel_position)):
                color_rgb.append(Image[pixel_position[j, 1], pixel_position[j, 0]])

            if len(color_rgb) > 0:
                color_rgb = np.vstack(color_rgb)

                # Store x, y, z coordinates and RGB colors
                new_points_inside = new_points_inside[:, :-1]  # Remove homogeneous component
                new_points_position_color = np.concatenate((new_points_inside, color_rgb), axis=1)

                # Append to the global list
                all_inside_points.append(new_points_position_color)

        # Combine all points
        if len(all_inside_points) > 0:
            self.point_cloud = np.vstack(all_inside_points)
            return self.point_cloud
        else:
            self.point_cloud = np.array([])
            return self.point_cloud

    def _build_point_cloud_raw(self, start_frame=0, end_frame=None, filter_min_x=5.0):
        """Build point cloud using raw files."""
        if end_frame is None:
            end_frame = len(self.paths['lidar'])

        # Store all points and colors
        all_inside_points = []

        # Get transformation matrices from calibration
        T_cam2_velo = self.calibration['T_cam2_velo']
        T_velo_imu = np.linalg.inv(get_rigid_transformation(self.calib_paths['imu_to_velo']))
        calibre = np.linalg.inv(T_velo_imu)  # IMU to LiDAR
        calibre_camera = T_cam2_velo[:3, :]  # LiDAR to camera

        # Get intrinsic camera matrix from calibration
        with open(self.calib_paths['cam_to_cam'], 'r') as f:
            calib = f.readlines()
        K_cam2 = np.array([float(x) for x in calib[5].strip().split(' ')[1:]]).reshape((3, 3))

        # Process each frame
        for i in range(start_frame, end_frame):
            try:
                # Load oxts data for this frame
                oxts_frame = get_oxts(self.paths['oxts'][i])

                # Create position matrix from oxts data
                roll, pitch, yaw = oxts_frame[3], oxts_frame[4], oxts_frame[5]
                pos_matrix = np.eye(4)

                # Simple approximation of the pose matrix
                # For more accurate pose, use the actual IMU data transformations
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
                cos_roll, sin_roll = np.cos(roll), np.sin(roll)

                # Rotation matrix
                R_z = np.array([[cos_yaw, -sin_yaw, 0],
                               [sin_yaw, cos_yaw, 0],
                               [0, 0, 1]])

                R_y = np.array([[cos_pitch, 0, sin_pitch],
                               [0, 1, 0],
                               [-sin_pitch, 0, cos_pitch]])

                R_x = np.array([[1, 0, 0],
                               [0, cos_roll, -sin_roll],
                               [0, sin_roll, cos_roll]])

                R = R_z @ R_y @ R_x
                pos_matrix[:3, :3] = R

                # Read in the lidar point cloud
                bin_file = self.paths['lidar'][i]
                lidar_pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

                # Filter points
                lidar_pts = lidar_pts[(lidar_pts[:, 0] >= filter_min_x)]  # Filter points with x >= filter_min_x
                if len(lidar_pts) == 0:
                    continue

                # Convert to homogeneous coordinates
                lidar_pts_hom = np.hstack((lidar_pts[:, :3], np.ones((lidar_pts.shape[0], 1))))

                # Transform LiDAR points into world coordinates
                new_points = np.array((pos_matrix @ (calibre @ lidar_pts_hom.T)))
                new_points[3, :] = 1  # Normalize homogeneous coordinates

                # Project LiDAR points into the camera image plane
                X_2D = np.array((K_cam2 @ (calibre_camera @ lidar_pts_hom.T)))

                # Transpose for compatibility
                new_points = new_points.T

                # Normalize by depth (z-axis)
                X_2D = X_2D / X_2D[2, :]
                X_2D = X_2D[0:2].T

                # Load the corresponding image as a NumPy array
                Image = cv2.cvtColor(cv2.imread(self.paths['left_images'][i]), cv2.COLOR_BGR2RGB)

                # Mask points that fall within the image bounds
                mask = ((X_2D[:, 0] > 0) & (X_2D[:, 0] < Image.shape[1]) &
                        (X_2D[:, 1] > 0) & (X_2D[:, 1] < Image.shape[0]))

                # Filter points and corresponding 2D positions
                new_points_inside = new_points[mask]
                X_2D_inside = X_2D[mask]

                # Convert 2D positions to integer pixel indices
                pixel_position = X_2D_inside.astype(int)

                # Retrieve RGB colors for each point
                color_rgb = []
                for j in range(len(pixel_position)):
                    color_rgb.append(Image[pixel_position[j, 1], pixel_position[j, 0]])

                if len(color_rgb) > 0:
                    color_rgb = np.vstack(color_rgb)

                    # Store x, y, z coordinates and RGB colors
                    new_points_inside = new_points_inside[:, :-1]  # Remove homogeneous component
                    new_points_position_color = np.concatenate((new_points_inside, color_rgb), axis=1)

                    # Append to the global list
                    all_inside_points.append(new_points_position_color)

            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue

        # Combine all points
        if len(all_inside_points) > 0:
            self.point_cloud = np.vstack(all_inside_points)
            return self.point_cloud
        else:
            self.point_cloud = np.array([])
            return self.point_cloud

    def save_point_cloud(self, output_path='point_cloud.ply'):
        """Save the reconstructed scene as a .ply file"""
        if self.point_cloud is None or len(self.point_cloud) == 0:
            print("No point cloud data available. Run build_point_cloud() first.")
            return

        # Convert to structured array for PLY export
        pos_3D_tuple = list(map(tuple, self.point_cloud))
        vertex = np.array(pos_3D_tuple, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = ply.PlyElement.describe(vertex, 'vertex')
        ply.PlyData([el]).write(output_path)
        print(f"3D scene reconstruction saved as '{output_path}'")
        return output_path

    # Visualization methods
    def visualize_frame(self, frame_idx, show_map=True):
        """
        Visualize a processed frame with object detection, LiDAR projection,
        and optionally the detected objects on a map.
        """
        # Process the frame
        outputs = self.process_frame(frame_idx)

        # Display results side-by-side
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(outputs['image_with_bboxes'])
        ax[0].set_title("YOLO Detections (with Depth)")
        ax[1].imshow(outputs['image_with_velo'])
        ax[1].set_title("Camera + Projected LiDAR")

        # Add map visualization if requested
        if show_map and len(outputs['geodetic']) > 0:
            oxts = outputs['oxts']
            lat0, lon0, alt0 = oxts[0], oxts[1], oxts[2]

            drive_map = folium.Map(location=(lat0, lon0), zoom_start=18)
            folium.CircleMarker(location=(lat0, lon0), radius=4, weight=5, color='red',
                                tooltip="Ego Vehicle").add_to(drive_map)

            for pos in outputs['geodetic']:
                folium.CircleMarker(location=(pos[0], pos[1]), radius=3, weight=3, color='blue',
                                    tooltip="Detected Object").add_to(drive_map)

            # Can't directly display Folium map in matplotlib, so return it for display
            return fig, drive_map
        else:
            return fig, None

    def visualize_trajectory(self, start_frame=0, end_frame=None):
        """
        Visualize the vehicle trajectory from OXTS data.
        """
        if end_frame is None:
            if self.use_raw_files:
                end_frame = len(self.paths['oxts'])
            else:
                end_frame = len(self.kitti_data.timestamps)

        # Extract trajectory data
        Xs, Ys, Zs = [], [], []

        if self.use_raw_files:
            for i in range(start_frame, end_frame):
                try:
                    oxts_frame = get_oxts(self.paths['oxts'][i])
                    # Use IMU position directly if available, otherwise convert from lat/lon
                    # Here we just use a simple approximation based on the first frame's position
                    if i == start_frame:
                        X0, Y0, Z0 = 0, 0, 0
                    else:
                        # Simplified relative position (in reality, use proper coordinate transformations)
                        lat, lon = oxts_frame[0], oxts_frame[1]
                        X0, Y0 = pm.geodetic2enu(lat, lon, 0, oxts_frame[0], oxts_frame[1], 0)[:2]
                        Z0 = 0

                    Xs.append(X0)
                    Ys.append(Y0)
                    Zs.append(Z0)
                except Exception as e:
                    print(f"Error processing trajectory frame {i}: {e}")
                    continue
        else:
            # Using pykitti data
            pos_matrix = self.kitti_data.oxts[start_frame:end_frame]
            for i, frame in enumerate(pos_matrix):
                Xs.append(frame[1][0][3])  # X
                Ys.append(frame[1][1][3])  # Y
                Zs.append(frame[1][2][3])  # Z

        # Store trajectory
        self.trajectory = np.column_stack((Xs, Ys, Zs))

        # Plot X, Y, Z trajectories
        fig, axis = plt.subplots(1, 3, figsize=(18, 5))
        axis[0].plot(Xs, Ys)
        axis[0].set(xlabel='X values', ylabel='Y values')
        axis[0].axis('equal')

        axis[1].plot(Ys, Zs)
        axis[1].set(xlabel='Y values', ylabel='Z values')
        axis[1].axis('equal')

        axis[2].plot(Xs, Zs)
        axis[2].set(xlabel='X values', ylabel='Z values')
        axis[2].axis('equal')

        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Example 1: Using raw files
    DATA_PATH = '/content/3d-SDV/KITTI_SAMPLE/RAW'
    DATE = '2011_09_26'
    DRIVE = '0009'

    pipeline = KITTI3DPipeline(data_path=DATA_PATH, date=DATE, drive=DRIVE, use_raw_files=True)

    # Process a single frame
    frame_idx = 10
    fig, drive_map = pipeline.visualize_frame(frame_idx)
    plt.show()

    # Build and save point cloud
    pipeline.build_point_cloud(start_frame=0, end_frame=20)
    pipeline.save_point_cloud('scene_reconstruction.ply')

    # Example 2: Using pykitti
    # pipeline2 = KITTI3DPipeline(data_path='/content', date='2011_10_03', drive='0047', use_raw_files=False)
    # pipeline2.visualize_trajectory()
    # pipeline2.build_point_cloud(start_frame=0, end_frame=50)
    # pipeline2.save_point_cloud('pykitti_scene.ply')
