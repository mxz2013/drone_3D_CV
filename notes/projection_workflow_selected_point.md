# Selected Pixel To 3D Model Workflow

This file explains the first complete 2D-to-3D experiment in this repo.

Selected input:

```text
image = interval5_HKairport03/interval5_CAM/1671607414.199796915.jpg
u = 1406
v = 1493
```

The code is in:

```text
projection_pipeline.py
```

Run it from the repo root:

```bash
uv run python projection_pipeline.py
```

The outputs are written to:

```text
outputs/projection_1671607414_199796915_u1406_v1493/
```

## What We Are Solving

You clicked one 2D image pixel. A single pixel does not directly give a 3D point. It gives a **3D ray** leaving the camera:

```text
camera center -> through pixel (u, v) -> out into the world
```

To get a 3D point, we intersect that ray with the reconstructed 3D mesh:

```text
3D point = ray intersected with Mesh.ply
```

## Inputs Used

From `sampleinfos_interpolated.json`, the selected frame provides:

```text
OriginalImageName = 1671607414.199796915.jpg
SortedImageID = 220
P3x3 = camera intrinsic matrix
T4x4 = camera pose transform
K1, K2, K3, P1, P2 = distortion coefficients
Width = 2448
Height = 2048
```

The script uses:

```text
Mesh.ply
```

for the actual ray-surface intersection.

It also uses:

```text
cloud_merged.ply
```

for a visual point-cloud overlay verification.

## Step 1: Load Frame Metadata

The function:

```python
load_frame_metadata(...)
```

finds the JSON record where:

```python
record["OriginalImageName"] == "1671607414.199796915.jpg"
```

It extracts:

```python
K = record["P3x3"]
D = [K1, K2, P1, P2, K3]
T = record["T4x4"]
```

For this frame:

```text
fx = 1471.0653
fy = 1471.0653
cx = 1172.3577
cy = 1046.3674
```

The distortion terms are all zero, so the first version does not need to bend the ray for lens distortion.

## Step 2: Pixel To Normalized Camera Coordinates

The selected pixel is:

```text
u = 1406
v = 1493
```

The intrinsic matrix maps between camera rays and pixels. To go backward from pixel to ray, remove the principal point and focal length:

```python
x = (u - cx) / fx
y = (v - cy) / fy
```

This gives normalized camera coordinates:

```text
x = 0.1588 approximately
y = 0.3036 approximately
```

The corresponding camera-space ray is:

```python
ray_camera = normalize([x, y, 1])
```

The actual value written by the script is:

```text
ray_camera = [0.1502498944, 0.2872189263, 0.9460075357]
```

## Step 3: Camera Ray To World Ray

The camera ray is only meaningful inside the camera coordinate system. To intersect the 3D map, it must be rotated into the map/world coordinate system.

The code supports two pose conventions:

```text
camera_to_world
world_to_camera
```

This matters because datasets do not always name transforms consistently.

For this run, the default is:

```text
camera_to_world
```

So the camera center is:

```python
camera_center_world = T4x4[:3, 3]
```

and the world ray direction is:

```python
ray_world = T4x4[:3, :3] @ ray_camera
```

The script computed:

```text
camera_center_world = [-120.32487605, -19.68809542, -5.92145123]
ray_direction_world = [0.34591117, -0.02151563, -0.93802055]
```

## Step 4: Ray-Mesh Intersection

The ray equation is:

```text
P(depth) = camera_center_world + depth * ray_direction_world
```

The script loads:

```text
terra_3dmap_pointcloud_mesh/HKairport/Mesh.ply
```

Then Open3D's raycasting scene finds the first triangle hit by the ray.

The result was:

```text
hit_point_world = [-93.03814542, -21.38532670, -79.91592292]
hit_distance = 78.883636
hit_mesh_triangle_id = 5755287
```

That 3D point is the map/model point corresponding to your selected image pixel.

## Verification 1: Same-Frame Reprojection

Output:

```text
01_same_frame_reprojection.png
```

This projects the 3D hit point back into the original image.

Expected result:

```text
the cyan reprojected point should land on the red selected point
```

The measured error was:

```text
0.000000 px
```

This proves the backprojection and reprojection math are internally consistent.

Important limitation: this check alone does not prove the pose convention is globally correct. If you use the same wrong convention in both directions, this check can still look perfect. That is why the next checks exist.

## Verification 2: Point Cloud Overlay

Outputs:

```text
02_point_cloud_overlay_camera_to_world.png
02_point_cloud_overlay_world_to_camera.png
```

This check samples points from:

```text
cloud_merged.ply
```

Then it projects those 3D map points into the selected image. The projected map points are drawn as bright green pixels.

Use this to answer:

```text
Do projected 3D map points align with visible roads, markings, buildings, and objects?
```

The score in `projection_result.json` says:

```text
camera_to_world: 27869 / 250000 sampled points inside image
world_to_camera: 13395 / 250000 sampled points inside image
```

The `camera_to_world` convention has more projected map support for this frame, so it is the better default for the ray intersection.

## Verification 3: 3D Ray And Hit Plot

Output:

```text
03_3d_ray_mesh_hit.png
```

This image shows:

```text
blue point = camera center
orange line = selected pixel ray
red point = mesh hit
gray points = sampled 3D map points near the hit
```

Use this to verify:

```text
the ray starts at the camera, travels toward the scene, and hits the surface near nearby map points
```

This check helps catch sign errors, axis flips, and impossible geometry.

## Verification 4: MeshLab Marker Overlay

Output:

```text
04_meshlab_debug_markers.ply
```

This file is a small colored mesh that can be loaded on top of the large
HKairport reconstruction in MeshLab.

It contains:

```text
red sphere = projected 3D hit point
blue sphere = camera center
orange cylinder = ray from camera to hit point
```

In MeshLab:

1. Open the main reconstruction:

   ```text
   terra_3dmap_pointcloud_mesh/HKairport/Mesh.ply
   ```

2. Import the marker layer:

   ```text
   outputs/projection_1671607414_199796915_u1406_v1493/04_meshlab_debug_markers.ply
   ```

3. Open the layer panel with:

   ```text
   View -> Show Layer Dialog
   ```

4. Keep both layers visible:

   ```text
   Mesh
   04_meshlab_debug_markers.ply
   ```

5. Use MeshLab's navigation tools to zoom toward the red sphere. The projected
   point should sit on the same physical surface you clicked in the image.

If the marker is hidden by the mesh surface, try one of these:

```text
Render -> Show Wireframe
Render -> Back-Face Culling
Filters -> Normals, Curvatures and Orientation -> Invert Faces Orientation
```

The most useful quick trick is to temporarily hide the main `Mesh` layer, locate
the red marker and orange ray, then show the main mesh again.

## Result Summary

The selected point:

```text
image = 1671607414.199796915.jpg
u = 1406
v = 1493
```

projected to:

```text
world X = -93.03814542
world Y = -21.38532670
world Z = -79.91592292
```

The full numeric result is stored in:

```text
outputs/projection_1671607414_199796915_u1406_v1493/projection_result.json
```

## How To Try Another Point

Use a different pixel:

```bash
uv run python projection_pipeline.py --u 1200 --v 900
```

Use a different image:

```bash
uv run python projection_pipeline.py \
  --image-name 1671607415.199801922.jpg \
  --u 1200 \
  --v 900
```

Use the alternative pose convention:

```bash
uv run python projection_pipeline.py --pose-convention world_to_camera
```

For a real annotation workflow, choose a physically precise point, not just a bounding-box center. Good points are:

```text
painted circle center
lane marking corner
car wheel center
logo center
building corner
road marking intersection
```
