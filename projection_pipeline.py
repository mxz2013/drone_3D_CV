"""Project one labeled image pixel into the UAVScenes 3D model.

This script is intentionally written as a learning tool.  Each function maps to
one concept in the 2D-to-3D workflow:

1. Read one frame's camera metadata.
2. Convert the selected image pixel into a camera ray.
3. Convert that camera ray into the 3D map/world coordinate frame.
4. Intersect the world ray with the reconstructed mesh.
5. Verify the result with three visual checks.

Run the default example from the repo root:

    uv run python projection_pipeline.py

The default point is the one selected in the prompt:

    image = 1671607414.199796915.jpg
    u = 1406
    v = 1493
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)


DEFAULT_IMAGE_NAME = "1671607414.199796915.jpg"
DEFAULT_U = 1406.0
DEFAULT_V = 1493.0


@dataclass(frozen=True)
class FrameMetadata:
    """Camera data for one image frame."""

    image_name: str
    image_path: Path
    sorted_image_id: int
    width: int
    height: int
    K: np.ndarray
    distortion: np.ndarray
    T4x4: np.ndarray


@dataclass(frozen=True)
class PoseConvention:
    """One possible interpretation of the metadata transform."""

    name: str
    description: str


@dataclass(frozen=True)
class RayHit:
    """The 3D result produced by ray-mesh intersection."""

    camera_center_world: np.ndarray
    ray_direction_world: np.ndarray
    hit_point_world: np.ndarray
    hit_distance: float
    mesh_triangle_id: int


CAMERA_TO_WORLD = PoseConvention(
    name="camera_to_world",
    description="Treat T4x4 as camera coordinates -> world/map coordinates.",
)

WORLD_TO_CAMERA = PoseConvention(
    name="world_to_camera",
    description="Treat T4x4 as world/map coordinates -> camera coordinates.",
)


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return the same vector with length 1."""

    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / length


def load_frame_metadata(
    sampleinfos_path: Path,
    image_dir: Path,
    image_name: str,
) -> FrameMetadata:
    """Find one image record in sampleinfos_interpolated.json."""

    records = json.loads(sampleinfos_path.read_text())
    matching_records = [
        record for record in records if record["OriginalImageName"] == image_name
    ]
    if not matching_records:
        raise FileNotFoundError(f"No metadata found for image {image_name!r}")

    record = matching_records[0]
    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image exists in metadata but not on disk: {image_path}")

    return FrameMetadata(
        image_name=image_name,
        image_path=image_path,
        sorted_image_id=int(record["SortedImageID"]),
        width=int(record["Width"]),
        height=int(record["Height"]),
        K=np.asarray(record["P3x3"], dtype=np.float64),
        distortion=np.asarray(
            [
                record["K1"],
                record["K2"],
                record["P1"],
                record["P2"],
                record["K3"],
            ],
            dtype=np.float64,
        ),
        T4x4=np.asarray(record["T4x4"], dtype=np.float64),
    )


def undistort_pixel_to_normalized_camera(
    u: float,
    v: float,
    K: np.ndarray,
    distortion: np.ndarray,
    iterations: int = 8,
) -> np.ndarray:
    """Convert a raw pixel into normalized pinhole-camera coordinates.

    Pixel coordinates live in image units:

        [u, v]

    Normalized camera coordinates remove focal length and principal point:

        x = (u - cx) / fx
        y = (v - cy) / fy

    If distortion coefficients are non-zero, this function iteratively removes
    Brown-Conrady radial/tangential distortion.  In this selected UAVScenes
    frame all distortion terms are zero, but keeping the math here makes the
    pipeline complete.
    """

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x_distorted = (u - cx) / fx
    y_distorted = (v - cy) / fy

    if np.allclose(distortion, 0.0):
        return np.array([x_distorted, y_distorted], dtype=np.float64)

    k1, k2, p1, p2, k3 = distortion
    x = x_distorted
    y = y_distorted

    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        tangential_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        tangential_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_distorted - tangential_x) / radial
        y = (y_distorted - tangential_y) / radial

    return np.array([x, y], dtype=np.float64)


def pixel_to_camera_ray(
    u: float,
    v: float,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Convert the selected pixel into a 3D ray in camera coordinates."""

    normalized_xy = undistort_pixel_to_normalized_camera(u, v, K, distortion)
    ray_camera = np.array([normalized_xy[0], normalized_xy[1], 1.0], dtype=np.float64)
    return normalize(ray_camera)


def camera_center_from_pose(T4x4: np.ndarray, convention: PoseConvention) -> np.ndarray:
    """Return the camera center expressed in world/map coordinates."""

    R = T4x4[:3, :3]
    t = T4x4[:3, 3]

    if convention == CAMERA_TO_WORLD:
        return t

    if convention == WORLD_TO_CAMERA:
        return -R.T @ t

    raise ValueError(f"Unknown convention: {convention}")


def camera_ray_to_world(
    ray_camera: np.ndarray,
    T4x4: np.ndarray,
    convention: PoseConvention,
) -> np.ndarray:
    """Rotate a camera-space ray into world/map coordinates."""

    R = T4x4[:3, :3]

    if convention == CAMERA_TO_WORLD:
        return normalize(R @ ray_camera)

    if convention == WORLD_TO_CAMERA:
        return normalize(R.T @ ray_camera)

    raise ValueError(f"Unknown convention: {convention}")


def world_points_to_camera(
    points_world: np.ndarray,
    T4x4: np.ndarray,
    convention: PoseConvention,
) -> np.ndarray:
    """Transform world/map points into the selected camera's coordinate frame."""

    R = T4x4[:3, :3]
    t = T4x4[:3, 3]

    if convention == CAMERA_TO_WORLD:
        camera_center = t
        return (R.T @ (points_world - camera_center).T).T

    if convention == WORLD_TO_CAMERA:
        return (R @ points_world.T).T + t

    raise ValueError(f"Unknown convention: {convention}")


def distort_normalized_points(
    xy_normalized: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Apply Brown-Conrady distortion to normalized camera coordinates."""

    if np.allclose(distortion, 0.0):
        return xy_normalized

    k1, k2, p1, p2, k3 = distortion
    x = xy_normalized[:, 0]
    y = xy_normalized[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([x_distorted, y_distorted])


def project_camera_points_to_pixels(
    points_camera: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Project camera-coordinate 3D points into image pixel coordinates."""

    z = points_camera[:, 2]
    xy_normalized = points_camera[:, :2] / z[:, None]
    xy_distorted = distort_normalized_points(xy_normalized, distortion)

    u = K[0, 0] * xy_distorted[:, 0] + K[0, 2]
    v = K[1, 1] * xy_distorted[:, 1] + K[1, 2]
    return np.column_stack([u, v])


def project_world_points_to_pixels(
    points_world: np.ndarray,
    metadata: FrameMetadata,
    convention: PoseConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform world points to camera space, then project them to pixels."""

    points_camera = world_points_to_camera(points_world, metadata.T4x4, convention)
    pixels = project_camera_points_to_pixels(
        points_camera,
        metadata.K,
        metadata.distortion,
    )
    return pixels, points_camera[:, 2]


def intersect_ray_with_mesh(
    mesh_path: Path,
    camera_center_world: np.ndarray,
    ray_direction_world: np.ndarray,
) -> RayHit:
    """Intersect a world-space camera ray with a triangle mesh."""

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no usable geometry: {mesh_path}")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ray = np.concatenate([camera_center_world, ray_direction_world])
    rays = o3d.core.Tensor([ray], dtype=o3d.core.Dtype.Float32)
    result = scene.cast_rays(rays)

    hit_distance = float(result["t_hit"].numpy()[0])
    if not math.isfinite(hit_distance):
        raise RuntimeError("The selected pixel ray did not hit the mesh.")

    triangle_id = int(result["primitive_ids"].numpy()[0])
    hit_point_world = camera_center_world + hit_distance * ray_direction_world

    return RayHit(
        camera_center_world=camera_center_world,
        ray_direction_world=ray_direction_world,
        hit_point_world=hit_point_world,
        hit_distance=hit_distance,
        mesh_triangle_id=triangle_id,
    )


def parse_binary_xyzrgb_ply_header(ply_path: Path) -> tuple[int, int]:
    """Return (vertex_count, byte_offset) for the UAVScenes binary point cloud."""

    vertex_count = None

    with ply_path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {ply_path}")

            text = line.decode("ascii", errors="replace").strip()

            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])

            if text == "end_header":
                if vertex_count is None:
                    raise ValueError(f"PLY header did not include vertex count: {ply_path}")
                return vertex_count, file.tell()


def sample_binary_xyzrgb_point_cloud(
    ply_path: Path,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a large binary point cloud without loading every point into RAM."""

    vertex_count, data_offset = parse_binary_xyzrgb_ply_header(ply_path)

    point_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    point_cloud = np.memmap(
        ply_path,
        dtype=point_dtype,
        mode="r",
        offset=data_offset,
        shape=(vertex_count,),
    )

    sample_count = min(max_points, vertex_count)
    sample_indices = np.linspace(0, vertex_count - 1, sample_count, dtype=np.int64)

    points = np.column_stack(
        [
            point_cloud["x"][sample_indices],
            point_cloud["y"][sample_indices],
            point_cloud["z"][sample_indices],
        ]
    ).astype(np.float64)

    colors = np.column_stack(
        [
            point_cloud["red"][sample_indices],
            point_cloud["green"][sample_indices],
            point_cloud["blue"][sample_indices],
        ]
    ).astype(np.uint8)

    return points, colors


def in_image_mask(
    pixels: np.ndarray,
    depth: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Select projected points that are in front of the camera and inside image bounds."""

    u = pixels[:, 0]
    v = pixels[:, 1]
    return (
        (depth > 0.0)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )


def save_reprojection_check(
    image_path: Path,
    selected_pixel: np.ndarray,
    reprojected_pixel: np.ndarray,
    output_path: Path,
) -> float:
    """Draw the clicked pixel and the reprojected 3D hit point on the image."""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    selected_u, selected_v = selected_pixel
    reproj_u, reproj_v = reprojected_pixel
    error_pixels = float(np.linalg.norm(reprojected_pixel - selected_pixel))

    draw_cross(draw, selected_u, selected_v, color=(255, 40, 40), radius=18, width=5)
    draw_cross(draw, reproj_u, reproj_v, color=(0, 220, 255), radius=26, width=4)
    draw.line(
        [(selected_u, selected_v), (reproj_u, reproj_v)],
        fill=(255, 255, 255),
        width=3,
    )

    label = (
        f"red = selected ({selected_u:.1f}, {selected_v:.1f}); "
        f"cyan = reprojected hit; error = {error_pixels:.3f}px"
    )
    draw.rectangle((20, 20, 1180, 88), fill=(0, 0, 0))
    draw.text((34, 42), label, fill=(255, 255, 255))

    image.save(output_path)
    return error_pixels


def draw_cross(
    draw: ImageDraw.ImageDraw,
    u: float,
    v: float,
    color: tuple[int, int, int],
    radius: int,
    width: int,
) -> None:
    """Draw a cross centered on one image pixel."""

    draw.line([(u - radius, v), (u + radius, v)], fill=color, width=width)
    draw.line([(u, v - radius), (u, v + radius)], fill=color, width=width)


def save_point_cloud_overlay(
    image_path: Path,
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    metadata: FrameMetadata,
    convention: PoseConvention,
    selected_pixel: np.ndarray,
    output_path: Path,
) -> dict[str, float | int | str]:
    """Project sampled map points into the image and save an overlay."""

    pixels, depth = project_world_points_to_pixels(points_world, metadata, convention)
    valid = in_image_mask(pixels, depth, metadata.width, metadata.height)

    valid_pixels = pixels[valid]
    valid_depth = depth[valid]
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    if len(valid_pixels) > 0:
        u_int = np.round(valid_pixels[:, 0]).astype(np.int64)
        v_int = np.round(valid_pixels[:, 1]).astype(np.int64)
        inside_after_round = (
            (u_int >= 0)
            & (u_int < metadata.width)
            & (v_int >= 0)
            & (v_int < metadata.height)
        )

        u_int = u_int[inside_after_round]
        v_int = v_int[inside_after_round]
        valid_depth = valid_depth[inside_after_round]

        pixel_id = v_int * metadata.width + u_int
        sort_order = np.lexsort((valid_depth, pixel_id))
        pixel_id = pixel_id[sort_order]
        u_int = u_int[sort_order]
        v_int = v_int[sort_order]

        first_for_pixel = np.r_[True, pixel_id[1:] != pixel_id[:-1]]
        u_int = u_int[first_for_pixel]
        v_int = v_int[first_for_pixel]

        # Use a bright overlay color instead of the point cloud's original RGB.
        # The goal of this verification image is alignment, so high contrast is
        # more useful than photorealistic coloring.
        overlay_color = np.array([0, 255, 80], dtype=np.float32)
        blended = (
            0.35 * image_array[v_int, u_int].astype(np.float32)
            + 0.65 * overlay_color[None, :]
        ).astype(np.uint8)
        image_array[v_int, u_int] = blended

    overlay = Image.fromarray(image_array)
    draw = ImageDraw.Draw(overlay)
    draw_cross(
        draw,
        selected_pixel[0],
        selected_pixel[1],
        color=(255, 40, 40),
        radius=20,
        width=5,
    )

    label = (
        f"{convention.name}: {int(valid.sum())} sampled map points project inside image"
    )
    draw.rectangle((20, 20, 980, 86), fill=(0, 0, 0))
    draw.text((34, 42), label, fill=(255, 255, 255))
    overlay.save(output_path)

    return {
        "convention": convention.name,
        "sampled_points": int(len(points_world)),
        "in_front_and_in_image": int(valid.sum()),
        "fraction_in_image": float(valid.sum() / max(len(points_world), 1)),
    }


def save_3d_ray_check(
    points_world: np.ndarray,
    ray_hit: RayHit,
    output_path: Path,
) -> None:
    """Save a 3D plot with sampled map points, camera center, ray, and hit point."""

    camera = ray_hit.camera_center_world
    hit = ray_hit.hit_point_world

    distances_to_hit = np.linalg.norm(points_world - hit[None, :], axis=1)
    nearby = points_world[distances_to_hit < 55.0]
    if len(nearby) > 20_000:
        nearby = nearby[np.linspace(0, len(nearby) - 1, 20_000, dtype=np.int64)]

    ray_line = np.vstack([camera, hit])

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    if len(nearby) > 0:
        axis.scatter(
            nearby[:, 0],
            nearby[:, 1],
            nearby[:, 2],
            s=1,
            c="lightgray",
            alpha=0.35,
            label="sampled map points near hit",
        )

    axis.plot(
        ray_line[:, 0],
        ray_line[:, 1],
        ray_line[:, 2],
        color="tab:orange",
        linewidth=3,
        label="camera ray",
    )
    axis.scatter(
        [camera[0]],
        [camera[1]],
        [camera[2]],
        c="tab:blue",
        s=80,
        label="camera center",
    )
    axis.scatter(
        [hit[0]],
        [hit[1]],
        [hit[2]],
        c="tab:red",
        s=100,
        label="mesh hit point",
    )

    axis.set_xlabel("world X")
    axis.set_ylabel("world Y")
    axis.set_zlabel("world Z")
    axis.legend(loc="upper left")
    axis.set_title("Verification 3: camera ray intersects the 3D map surface")
    set_axes_equal(axis, np.vstack([nearby, ray_line]) if len(nearby) else ray_line)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def rotation_matrix_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that rotates one unit vector into another."""

    source = normalize(source)
    target = normalize(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))

    if np.isclose(dot, 1.0):
        return np.eye(3)

    if np.isclose(dot, -1.0):
        if abs(source[0]) < 0.9:
            axis = normalize(np.cross(source, np.array([1.0, 0.0, 0.0])))
        else:
            axis = normalize(np.cross(source, np.array([0.0, 1.0, 0.0])))
        skew_axis = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + 2.0 * skew_axis @ skew_axis

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def save_meshlab_debug_markers(ray_hit: RayHit, output_path: Path) -> None:
    """Write colored 3D helper geometry that can be imported into MeshLab.

    MeshLab can render a single point, but it is easy to miss inside a large
    reconstruction.  A small red sphere and an orange cylinder are much easier
    to inspect.
    """

    camera = ray_hit.camera_center_world
    hit = ray_hit.hit_point_world
    ray_vector = hit - camera
    ray_length = float(np.linalg.norm(ray_vector))
    ray_direction = normalize(ray_vector)

    hit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5, resolution=24)
    hit_sphere.translate(hit)
    hit_sphere.paint_uniform_color([1.0, 0.0, 0.0])

    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.8, resolution=24)
    camera_sphere.translate(camera)
    camera_sphere.paint_uniform_color([0.0, 0.2, 1.0])

    ray_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.25,
        height=ray_length,
        resolution=24,
    )
    rotation = rotation_matrix_between_vectors(
        source=np.array([0.0, 0.0, 1.0]),
        target=ray_direction,
    )
    ray_cylinder.rotate(rotation, center=np.array([0.0, 0.0, 0.0]))
    ray_cylinder.translate((camera + hit) / 2.0)
    ray_cylinder.paint_uniform_color([1.0, 0.55, 0.0])

    marker_mesh = hit_sphere + camera_sphere + ray_cylinder
    marker_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), marker_mesh, write_ascii=True)


def set_axes_equal(axis: plt.Axes, points: np.ndarray) -> None:
    """Make a 3D matplotlib plot use the same scale on X, Y, and Z."""

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1.0)

    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


def write_report(
    output_path: Path,
    metadata: FrameMetadata,
    selected_pixel: np.ndarray,
    convention: PoseConvention,
    ray_camera: np.ndarray,
    ray_hit: RayHit,
    reprojected_pixel: np.ndarray,
    reprojection_error_pixels: float,
    overlay_scores: Iterable[dict[str, float | int | str]],
) -> None:
    """Write a machine-readable summary of the run."""

    report = {
        "image_name": metadata.image_name,
        "sorted_image_id": metadata.sorted_image_id,
        "selected_pixel_uv": selected_pixel.tolist(),
        "chosen_pose_convention": convention.name,
        "camera_intrinsic_K": metadata.K.tolist(),
        "distortion_k1_k2_p1_p2_k3": metadata.distortion.tolist(),
        "T4x4": metadata.T4x4.tolist(),
        "ray_camera": ray_camera.tolist(),
        "camera_center_world": ray_hit.camera_center_world.tolist(),
        "ray_direction_world": ray_hit.ray_direction_world.tolist(),
        "hit_point_world": ray_hit.hit_point_world.tolist(),
        "hit_distance_from_camera": ray_hit.hit_distance,
        "hit_mesh_triangle_id": ray_hit.mesh_triangle_id,
        "reprojected_pixel_uv": reprojected_pixel.tolist(),
        "same_frame_reprojection_error_pixels": reprojection_error_pixels,
        "point_cloud_overlay_scores": list(overlay_scores),
    }
    output_path.write_text(json.dumps(report, indent=2))


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete 2D-pixel to 3D-model experiment."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_frame_metadata(
        sampleinfos_path=args.sampleinfos,
        image_dir=args.image_dir,
        image_name=args.image_name,
    )
    selected_pixel = np.array([args.u, args.v], dtype=np.float64)

    ray_camera = pixel_to_camera_ray(
        u=args.u,
        v=args.v,
        K=metadata.K,
        distortion=metadata.distortion,
    )

    convention = CAMERA_TO_WORLD if args.pose_convention == "camera_to_world" else WORLD_TO_CAMERA
    camera_center_world = camera_center_from_pose(metadata.T4x4, convention)
    ray_direction_world = camera_ray_to_world(ray_camera, metadata.T4x4, convention)

    ray_hit = intersect_ray_with_mesh(
        mesh_path=args.mesh,
        camera_center_world=camera_center_world,
        ray_direction_world=ray_direction_world,
    )

    hit_pixels, hit_depth = project_world_points_to_pixels(
        ray_hit.hit_point_world[None, :],
        metadata,
        convention,
    )
    if hit_depth[0] <= 0:
        raise RuntimeError("The hit point projects behind the selected camera.")

    reprojection_error_pixels = save_reprojection_check(
        image_path=metadata.image_path,
        selected_pixel=selected_pixel,
        reprojected_pixel=hit_pixels[0],
        output_path=output_dir / "01_same_frame_reprojection.png",
    )

    sampled_points, sampled_colors = sample_binary_xyzrgb_point_cloud(
        ply_path=args.point_cloud,
        max_points=args.point_cloud_samples,
    )

    overlay_scores = []
    for overlay_convention in (CAMERA_TO_WORLD, WORLD_TO_CAMERA):
        overlay_scores.append(
            save_point_cloud_overlay(
                image_path=metadata.image_path,
                points_world=sampled_points,
                colors_rgb=sampled_colors,
                metadata=metadata,
                convention=overlay_convention,
                selected_pixel=selected_pixel,
                output_path=output_dir
                / f"02_point_cloud_overlay_{overlay_convention.name}.png",
            )
        )

    save_3d_ray_check(
        points_world=sampled_points,
        ray_hit=ray_hit,
        output_path=output_dir / "03_3d_ray_mesh_hit.png",
    )
    save_meshlab_debug_markers(
        ray_hit=ray_hit,
        output_path=output_dir / "04_meshlab_debug_markers.ply",
    )

    write_report(
        output_path=output_dir / "projection_result.json",
        metadata=metadata,
        selected_pixel=selected_pixel,
        convention=convention,
        ray_camera=ray_camera,
        ray_hit=ray_hit,
        reprojected_pixel=hit_pixels[0],
        reprojection_error_pixels=reprojection_error_pixels,
        overlay_scores=overlay_scores,
    )

    print(f"Image: {metadata.image_name}")
    print(f"Selected pixel: u={args.u}, v={args.v}")
    print(f"Pose convention used for ray intersection: {convention.name}")
    print(f"Camera center world: {ray_hit.camera_center_world}")
    print(f"Ray direction world: {ray_hit.ray_direction_world}")
    print(f"Hit point world: {ray_hit.hit_point_world}")
    print(f"Hit distance: {ray_hit.hit_distance:.3f}")
    print(f"Same-frame reprojection error: {reprojection_error_pixels:.6f} px")
    print(f"Wrote outputs to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the learning script."""

    parser = argparse.ArgumentParser(
        description="Project one selected UAVScenes image pixel into the 3D mesh.",
    )
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--u", type=float, default=DEFAULT_U)
    parser.add_argument("--v", type=float, default=DEFAULT_V)
    parser.add_argument(
        "--sampleinfos",
        type=Path,
        default=Path("interval5_HKairport03/sampleinfos_interpolated.json"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("interval5_HKairport03/interval5_CAM"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("terra_3dmap_pointcloud_mesh/HKairport/Mesh.ply"),
    )
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=Path("terra_3dmap_pointcloud_mesh/HKairport/cloud_merged.ply"),
    )
    parser.add_argument(
        "--pose-convention",
        choices=["camera_to_world", "world_to_camera"],
        default="camera_to_world",
        help="How to interpret T4x4 when computing the ray-mesh hit.",
    )
    parser.add_argument(
        "--point-cloud-samples",
        type=int,
        default=250_000,
        help="How many cloud_merged.ply points to sample for visual overlays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/projection_1671607414_199796915_u1406_v1493"),
    )
    return parser


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
