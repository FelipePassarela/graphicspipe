import sys

import numpy as np
from numba import njit


@njit(cache=True)
def render_faces(
    viewport: np.ndarray,
    z_buffer: np.ndarray,
    faces: np.ndarray,
    clip_coords: np.ndarray,
    normals: np.ndarray,
    light_dir: np.ndarray,
    screen_w: int,
    screen_h: int,
) -> None:
    for i in range(len(faces)):
        face = faces[i]
        v1 = clip_coords[face[0, 0]].copy()
        v2 = clip_coords[face[1, 0]].copy()
        v3 = clip_coords[face[2, 0]].copy()

        v1_w = v1[3]
        v2_w = v2[3]
        v3_w = v3[3]

        # Reject vertices behind the camera
        if v1_w <= 0 or v2_w <= 0 or v3_w <= 0:
            continue

        # Frustum clipping (in clip space: -w <= x,y,z <= w)
        if not (
            -v1_w <= v1[0] <= v1_w and -v2_w <= v2[0] <= v2_w and -v3_w <= v3[0] <= v3_w
        ):
            continue
        if not (
            -v1_w <= v1[1] <= v1_w and -v2_w <= v2[1] <= v2_w and -v3_w <= v3[1] <= v3_w
        ):
            continue
        if not (
            -v1_w <= v1[2] <= v1_w and -v2_w <= v2[2] <= v2_w and -v3_w <= v3[2] <= v3_w
        ):
            continue

        v1 /= v1_w
        v2 /= v2_w
        v3 /= v3_w

        # Screen coordinates
        s1 = np.array(
            [(v1[0] + 1.0) * 0.5 * screen_w, (1.0 - v1[1]) * 0.5 * screen_h, v1[2]]
        )
        s2 = np.array(
            [(v2[0] + 1.0) * 0.5 * screen_w, (1.0 - v2[1]) * 0.5 * screen_h, v2[2]]
        )
        s3 = np.array(
            [(v3[0] + 1.0) * 0.5 * screen_w, (1.0 - v3[1]) * 0.5 * screen_h, v3[2]]
        )

        # Backface culling
        area = edge_function(s1, s3, s2)
        if area <= 0:
            continue
        inv_area = 1.0 / area

        # Light intensity
        n1 = normals[face[0, 1]]
        n2 = normals[face[1, 1]]
        n3 = normals[face[2, 1]]
        mean_normal = (n1 + n2 + n3) / 3.0

        intensity = np.dot(mean_normal, -light_dir)
        intensity = max(0.0, min(1.0, intensity))
        shade_idx = int(intensity * 4)

        # Triangle bounding box
        min_x = max(0, int(min(s1[0], s2[0], s3[0])))
        max_x = min(screen_w - 1, int(max(s1[0], s2[0], s3[0])))
        min_y = max(0, int(min(s1[1], s2[1], s3[1])))
        max_y = min(screen_h - 1, int(max(s1[1], s2[1], s3[1])))

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                pixel = px + 0.5, py + 0.5
                w0 = edge_function(s3, s2, pixel)
                w1 = edge_function(s1, s3, pixel)
                w2 = edge_function(s2, s1, pixel)

                inside_triangle = w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0
                if inside_triangle:
                    w0 *= inv_area
                    w1 *= inv_area
                    w2 *= inv_area

                    z = w0 * s1[2] + w1 * s2[2] + w2 * s3[2]
                    if z < z_buffer[py, px]:
                        z_buffer[py, px] = z
                        viewport[py, px] = shade_idx


@njit(cache=True)
def edge_function(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def display(
    viewport: np.ndarray,
    screen_w: int,
    screen_h: int,
    camera: dict = None,
    dt: float = None,
) -> None:
    shade_chars = np.array(list(" ░▒▓█"))
    viewport_chars = shade_chars[viewport]
    lines = ["".join(row) for row in viewport_chars]

    message = "Graphicspipe - Press ESC to quit"
    lines[screen_h - 1] = message.ljust(screen_w)

    debug_info = ""
    if camera is not None:
        camera_x = camera["eye"][0]
        camera_y = camera["eye"][1]
        camera_z = camera["eye"][2]
        debug_info += (
            f"Camera Position: ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f}) "
            f"Yaw: {camera['yaw']:.2f} Pitch: {camera['pitch']:.2f} "
            f"FOV: {camera['fov']:.2f} "
        )
    if dt is not None:
        fps = 1.0 / dt if dt > 0 else 0.0
        debug_info += f"FPS: {fps:.2f}"
    lines[0] = debug_info.ljust(screen_w)

    frame = "\n".join(lines)
    sys.stdout.write("\x1b[H" + frame)
    sys.stdout.flush()
