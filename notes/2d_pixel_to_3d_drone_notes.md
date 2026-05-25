# 2D Pixel To 3D Drone Model Notes

This note records the dataset choice and the basic geometry needed to project a labeled 2D image point, such as a logo center `(u, v)`, onto a reconstructed 3D drone model.

## Recommended Dataset

Use **UAVScenes**:

- GitHub: <https://github.com/sijieaaa/UAVScenes>
- Hugging Face files: <https://huggingface.co/datasets/sijieaaa/UAVScenes/tree/main>

This dataset is a good fit for the project because it includes:

- drone camera frames
- LiDAR point clouds
- accurate 6-DoF poses
- camera/LiDAR calibration
- reconstructed DJI Terra point-cloud and mesh maps
- `Mesh.ply`, `cloud_merged.ply`, and block-level Terra PLY files
- `sampleinfos_interpolated.json` for camera-to-3D-map alignment
- `calibration_results.py` for camera intrinsics, distortion, and camera/LiDAR extrinsics

The dataset is not tiny overall, but you can practice with one extracted scene and one frame.

## Local Files In This Repo

Relevant local files currently present:

- `interval5_HKairport03/sampleinfos_interpolated.json`
- `interval5_HKairport03/rtk_positions_raw.csv`
- `interval5_HKairport03/interval5_LIDAR/`
- `terra_3dmap_pointcloud_mesh/HKairport_GNSS_Evening/cloud_merged.ply`
- `terra_3dmap_pointcloud_mesh/AMtown/Mesh.ply`
- `terra_3dmap_pointcloud_mesh/AMtown/cloud_merged.ply`

For the first implementation, use one frame from `interval5_HKairport03/sampleinfos_interpolated.json` and one corresponding LiDAR/map point cloud.

## What Drone Metadata Usually Provides

Useful metadata can include:

- image timestamp
- image filename or frame id
- camera intrinsics: `fx`, `fy`, `cx`, `cy`
- distortion coefficients: `k1`, `k2`, `p1`, `p2`, `k3`
- camera-to-body, camera-to-LiDAR, or camera-to-world extrinsic transform
- GPS or RTK position
- IMU orientation
- gimbal yaw, pitch, and roll
- LiDAR point cloud timestamp
- optimized 6-DoF camera pose from SLAM, photogrammetry, or bundle adjustment

GPS alone is not enough for accurate pixel-to-model projection. GPS gives approximate camera position, while IMU/gimbal gives approximate orientation. For precise projection, use the optimized camera pose in the 3D map coordinate system whenever it is available.

## Camera Metadata Shape

UAVScenes-style calibration data conceptually looks like this:

```python
camera_intrinsic = [
    fx, 0,  cx,
    0,  fy, cy,
    0,  0,  1,
]

camera_dist_coeffs = [k1, k2, p1, p2, k3]

camera_ext_R = [
    [r00, r01, r02],
    [r10, r11, r12],
    [r20, r21, r22],
]

camera_ext_t = [tx, ty, tz]
```

The intrinsic matrix is:

```python
K = [
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1],
]
```

## Core Problem

Given one labeled point:

```text
frame_t, u, v
```

where `(u, v)` is the pixel coordinate of the center of a logo bounding box, compute the corresponding 3D point on the model.

Required inputs:

```text
K              camera intrinsic matrix
D              lens distortion coefficients
T_world_cam    camera pose in the 3D map/model coordinate system
Mesh.ply       or point cloud/depth data
```

## Projection Pipeline

### 1. Undistort The Pixel

Lens distortion means the raw pixel is not exactly on the ideal pinhole camera ray. First undistort it:

```python
u_undist, v_undist = undistort_pixel(u, v, K, D)
```

With OpenCV this is typically done with `cv2.undistortPoints`.

### 2. Convert Pixel To Camera Ray

Convert the undistorted pixel into a ray in the camera coordinate system:

```python
pixel_h = np.array([u_undist, v_undist, 1.0])
ray_cam = np.linalg.inv(K) @ pixel_h
ray_cam = ray_cam / np.linalg.norm(ray_cam)
```

This gives a direction vector from the camera center through the selected pixel.

### 3. Transform Ray To World/Map Coordinates

If `T_world_cam` is a camera-to-world transform:

```python
R_world_cam = T_world_cam[:3, :3]
C_world = T_world_cam[:3, 3]

ray_world = R_world_cam @ ray_cam
ray_world = ray_world / np.linalg.norm(ray_world)
```

If the pose is stored as world-to-camera, as in COLMAP-style output:

```python
C_world = -R_world_cam.T @ t_world_cam
ray_world = R_world_cam.T @ ray_cam
```

Always verify the convention of the specific metadata file before coding the final transform.

### 4. Intersect Ray With The 3D Model

The 3D point lies somewhere on this ray:

```python
P_world = C_world + depth * ray_world
```

To find `depth`, intersect the ray with:

- a mesh, for example `Mesh.ply`
- a point cloud, for example `cloud_merged.ply`
- a LiDAR frame transformed into the map coordinate system
- a depth map, if available

For a mesh, use ray-triangle intersection. Libraries such as `trimesh` can do this directly:

```python
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=[C_world],
    ray_directions=[ray_world],
)
```

The closest positive intersection along the ray is the projected 3D point.

## Alternative Point Cloud Method

If using a point cloud instead of a mesh:

1. Transform 3D points into the camera coordinate frame.
2. Project each 3D point into the image with `K`.
3. Keep points whose projected pixel is near `(u, v)`.
4. Choose the nearest valid depth or average a small neighborhood.

Projection from camera coordinates to pixel coordinates:

```python
x = X_cam / Z_cam
y = Y_cam / Z_cam

u = fx * x + cx
v = fy * y + cy
```

This is often easier to debug than ray-mesh intersection because you can draw projected point-cloud pixels back onto the image.

## First Coding Milestone

For this repo, a good first milestone is:

1. Load one record from `interval5_HKairport03/sampleinfos_interpolated.json`.
2. Extract or construct the camera pose for that frame.
3. Load the matching image and choose one manual `(u, v)` point.
4. Build `K` and distortion coefficients.
5. Convert `(u, v)` into a world ray.
6. Load a local `.ply` mesh or point cloud.
7. Compute the closest ray intersection or nearest projected point.
8. Visualize the selected 3D point.

## Useful References

- UAVScenes: <https://github.com/sijieaaa/UAVScenes>
- UAVScenes files: <https://huggingface.co/datasets/sijieaaa/UAVScenes/tree/main>
- COLMAP output format: <https://colmap.github.io/format.html>
- COLMAP camera models: <https://colmap.github.io/cameras.html>
- MARS-LVIG dataset page: <https://mars.hku.hk/dataset.html>
- AeroGrid100 fallback dataset: <https://huggingface.co/datasets/bbz4021/AeroGrid100>

