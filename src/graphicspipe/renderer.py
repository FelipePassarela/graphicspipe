import sys
from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def reder_faces(
    faces: np.ndarray,
    clip_coords: np.ndarray,
    normals: np.ndarray,
    light_dir: np.ndarray,
    screen_w: int,
    screen_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_faces = len(faces)
    sx_arr = np.empty(max_faces, dtype=np.int32)
    sy_arr = np.empty(max_faces, dtype=np.int32)
    shade_arr = np.empty(max_faces, dtype=np.int32)
    count = 0

    num_clip_coords = len(clip_coords)

    for i in range(len(faces)):
        face = faces[i]
        v1_idx = face[0, 0]
        v2_idx = face[1, 0]
        v3_idx = face[2, 0]

        if (
            v1_idx >= num_clip_coords
            or v2_idx >= num_clip_coords
            or v3_idx >= num_clip_coords
        ):
            continue

        v1 = clip_coords[v1_idx]
        v2 = clip_coords[v2_idx]
        v3 = clip_coords[v3_idx]

        # Backface culling
        # This is currently disabled because, for now, only small vertices on the face
        # are rendered, so backface culling would make the scene feeling empty.
        # area = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
        # if area <= 0:
        #     continue

        center_x = (v1[0] + v2[0] + v3[0]) / 3.0
        center_y = (v1[1] + v2[1] + v3[1]) / 3.0
        sx = int((center_x + 1) * screen_w / 2.0)
        sy = int((1 - center_y) * screen_h / 2.0)

        if not (0 <= sx < screen_w and 0 <= sy < screen_h):
            continue

        n1_idx = face[0, 1]
        n2_idx = face[1, 1]
        n3_idx = face[2, 1]

        n1 = normals[n1_idx]
        n2 = normals[n2_idx]
        n3 = normals[n3_idx]
        n1_intensity = np.dot(n1, light_dir)
        n2_intensity = np.dot(n2, light_dir)
        n3_intensity = np.dot(n3, light_dir)

        intensity = (n1_intensity + n2_intensity + n3_intensity) / 3.0
        intensity = max(0.0, intensity)
        shade_index = min(int(5 * intensity), 4)

        sx_arr[count] = sx
        sy_arr[count] = sy
        shade_arr[count] = shade_index
        count += 1

    return sx_arr[:count], sy_arr[:count], shade_arr[:count]


def display(
    viewport: np.ndarray,
    screen_w: int,
    screen_h: int,
    camera: dict = None,
    dt: float = None,
) -> None:
    lines = ["".join(c for c in row) for row in viewport]
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
