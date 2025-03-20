import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

def get_rigid_transformation(file_path):
    """
    Read a KITTI calibration file and return the 4x4 rigid transformation matrix.
    """
    with open(file_path, 'r') as f:
        calib = f.readlines()

    # Extract rotation matrix (3x3)
    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))

    # Extract translation vector (3x1)
    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]]).reshape((3, 1))

    # Combine into a 4x4 transformation matrix
    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

    return T

def get_oxts(file_path):
    """
    Read KITTI oxts data from file and return as a tuple containing:
    (lat, lon, alt, roll, pitch, yaw)
    """
    with open(file_path, 'r') as f:
        line = f.readline().strip().split(' ')

    # Extract relevant fields
    lat, lon, alt = float(line[0]), float(line[1]), float(line[2])
    roll, pitch, yaw = float(line[3]), float(line[4]), float(line[5])

    return (lat, lon, alt, roll, pitch, yaw)

def project_velobin2uvz(bin_path, T_velo_cam, image, remove_plane=False,
                       min_dist=0, max_dist=100, visualize=False, downsample=1):
    """
    Project a LiDAR point cloud (from .bin file) to image coordinates and return (u, v, z) arrays

    Args:
        bin_path: Path to KITTI .bin LiDAR file
        T_velo_cam: 3x4 projection matrix from LiDAR to camera
        image: Image to project onto (for size reference)
        remove_plane: Whether to remove ground plane
        min_dist, max_dist: Min/max depth range for points
        visualize: Whether to show the projection (for debugging)
        downsample: Downsampling factor for LiDAR points

    Returns:
        Tuple (u, v, z) of image coordinates and depths
    """
    # Read bin file
    velo_points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    velo_points = velo_points[::downsample]  # downsample for speed

    # Filter points by distance
    distances = np.sqrt(np.sum(velo_points[:, :3] ** 2, axis=1))
    velo_points = velo_points[(distances >= min_dist) & (distances <= max_dist)]

    # Remove ground plane if requested
    if remove_plane:
        velo_points = remove_ground_plane(velo_points)

    # Convert to homogeneous coordinates
    velo_points_hom = np.hstack((velo_points[:, :3], np.ones((velo_points.shape[0], 1))))

    # Project to image plane
    uvz = T_velo_cam @ velo_points_hom.T  # 3x4 @ 4xN = 3xN

    # Filter points in front of camera (z > 0)
    z = uvz[2, :]
    mask = z > 0
    uvz = uvz[:, mask]

    # Normalize u, v coordinates by z
    uv = uvz[:2, :] / uvz[2, :]
    z = uvz[2, :]

    # Filter points within image bounds
    img_h, img_w = image.shape[:2]
    mask = (uv[0, :] >= 0) & (uv[0, :] < img_w) & (uv[1, :] >= 0) & (uv[1, :] < img_h)
    u, v = uv[0, mask], uv[1, mask]
    z = z[mask]

    if visualize:
        # Create a copy of the image for visualization
        img_vis = image.copy()
        for i in range(len(u)):
            px = int(round(u[i]))
            py = int(round(v[i]))
            cv2.circle(img_vis, (px, py), 2, (0, 255, 0), -1)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_vis)
        plt.title('LiDAR points projected to image')
        plt.show()

    return (u, v, z)

def remove_ground_plane(velo_points, height_threshold=0.6, n_clusters=1):
    """
    Remove ground plane points from a LiDAR point cloud using RANSAC.

    Args:
        velo_points: Nx4 array of LiDAR points
        height_threshold: Height threshold for ground plane (meters)
        n_clusters: Number of ground plane clusters to extract

    Returns:
        Filtered point cloud with ground plane removed
    """
    try:
        # Prepare data
        XYZ = velo_points[:, :3]

        # Polynomial features for RANSAC (we model the ground as a polynomial surface)
        poly = PolynomialFeatures(degree=2)
        XY = poly.fit_transform(XYZ[:, :2])

        # Fit RANSAC model to find the ground plane
        ransac = RANSACRegressor(max_trials=100, residual_threshold=0.1)
        ransac.fit(XY, XYZ[:, 2])

        # Predict the height of the ground plane
        z_ground = ransac.predict(XY)

        # Identify points above the ground plane
        inliers = XYZ[:, 2] - z_ground > height_threshold

        # Return filtered points
        return velo_points[inliers]

    except Exception as e:
        print(f"Error in ground plane removal: {e}")
        return velo_points  # Return original points if the algorithm fails

def transform_uvz(uvz, T_src_dst):
    """
    Transform (u, v, z) points from one reference frame to another using a 4x4 transformation matrix.

    Args:
        uvz: Nx3 array of (u, v, z) points
        T_src_dst: 4x4 transformation matrix from source to destination frame

    Returns:
        Nx3 array of transformed (x, y, z) points
    """
    # Get camera coordinates (un-project from image to camera)
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]

    # Simple pin-hole camera model un-projection
    # Note: for a real pinhole camera, this should use calibration parameters
    # This is a simplification assuming a unit focal length centered camera
    x = np.multiply(u, z)
    y = np.multiply(v, z)

    # Create 3D points in camera coordinate system
    points_src = np.column_stack([x, y, z, np.ones(len(z))])

    # Transform to destination coordinate system
    points_dst = np.dot(points_src, T_src_dst.T)

    return points_dst[:, :3]

def draw_velo_on_image(velo_uvz, image, point_size=2, color_by_depth=True):
    """
    Draw LiDAR points on an image.

    Args:
        velo_uvz: Tuple of (u, v, z) arrays for LiDAR points
        image: Image to draw on
        point_size: Size of drawn points
        color_by_depth: Whether to color points by depth

    Returns:
        Image with projected LiDAR points
    """
    img_vis = image.copy()
    u, v, z = velo_uvz

    if color_by_depth:
        # Create a colormap for depth
        z_min, z_max = np.min(z), np.max(z)
        z_range = z_max - z_min

        for i in range(len(u)):
            px = int(round(u[i]))
            py = int(round(v[i]))

            # Normalize depth to 0-1
            depth_norm = (z[i] - z_min) / z_range if z_range > 0 else 0.5

            # Create color using HSV (hue based on depth) and convert to BGR
            hue = int(240 * (1 - depth_norm))  # Blue (240) for far, Red (0) for near
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
            color = (int(color[0]), int(color[1]), int(color[2]))  # Convert to int tuple

            cv2.circle(img_vis, (px, py), point_size, color, -1)
    else:
        # Single color for all points
        for i in range(len(u)):
            px = int(round(u[i]))
            py = int(round(v[i]))
            cv2.circle(img_vis, (px, py), point_size, (0, 255, 0), -1)

    return img_vis
