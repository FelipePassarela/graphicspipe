import os
import sys
import time

import numpy as np
from pynput import keyboard

from graphicspipe import math, mesh
from graphicspipe.input_state import InputState

SCREEN_W = 120
SCREEN_H = 40
TARGET_FRAME_TIME = 1.0 / 60.0

MOVE_SPEED = 2.0
CAMERA_ROTATION_SPEED = 45.0
FOV_SPEED = 20.0


def main() -> None:
    model_mesh, normals, faces = mesh.parse("assets/plane.obj")

    model = {
        "mesh": model_mesh,
        "normals": normals,
        "faces": faces,
        "translation": np.array([0.0, 0.0, 0.0]),
        "scale": np.array([1.0, 1.0, 1.0]),
        "rotation": np.array([90.0, 0.0, 0.0]),
    }
    model["scale"] /= np.max(np.abs(model["mesh"][:, :3]))  # normalize size

    camera = {
        "yaw": -75.0,
        "pitch": 10.0,
        "eye": np.array([-1.2, 0.0, -0.3]),  # move back to see the object
        "up": np.array([0.0, 1.0, 0.0]),
        "near": 0.1,
        "far": 100.0,
        "fov": 60.0,
    }

    light_source = {
        "position": np.array([-5.0, 5.0, -5.0]),
        "direction": np.array([0.0, 0.0, 1.0]),
    }
    light_source["direction"] /= np.array([0.0, 0.0, 0.0]) - light_source["position"]
    light_source["direction"] /= np.linalg.norm(light_source["direction"])

    input_state = InputState()
    key_listener = keyboard.Listener(
        on_press=input_state.on_press, on_release=input_state.on_release
    )
    key_listener.start()

    os.system("cls" if os.name == "nt" else "clear")
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        dt = min(dt, 0.1)  # avoid large jumps

        forward = math.forward(np.radians(camera["yaw"]), np.radians(camera["pitch"]))
        right = np.cross(camera["up"], forward)
        right /= np.linalg.norm(right)

        if input_state.is_pressed("w"):
            camera["eye"] += forward * MOVE_SPEED * dt
        if input_state.is_pressed("s"):
            camera["eye"] -= forward * MOVE_SPEED * dt
        if input_state.is_pressed("a"):
            camera["eye"] -= right * MOVE_SPEED * dt
        if input_state.is_pressed("d"):
            camera["eye"] += right * MOVE_SPEED * dt

        if input_state.is_pressed(keyboard.Key.left):
            camera["yaw"] += CAMERA_ROTATION_SPEED * dt
        if input_state.is_pressed(keyboard.Key.right):
            camera["yaw"] -= CAMERA_ROTATION_SPEED * dt
        if input_state.is_pressed(keyboard.Key.up):
            camera["pitch"] += CAMERA_ROTATION_SPEED * dt
        if input_state.is_pressed(keyboard.Key.down):
            camera["pitch"] -= CAMERA_ROTATION_SPEED * dt
        if input_state.is_pressed("+"):
            camera["fov"] = min(150.0, camera["fov"] + FOV_SPEED * dt)
        if input_state.is_pressed("-"):
            camera["fov"] = max(60.0, camera["fov"] - FOV_SPEED * dt)

        if input_state.is_pressed(keyboard.Key.esc):
            key_listener.stop()
            exit()

        model["rotation"][1] += 180.0 * dt

        world_matrix = math.compose(
            translations=model["translation"],
            rotations=model["rotation"],
            scales=model["scale"],
        )
        world_coords = model["mesh"] @ world_matrix

        # view transformation
        camera["pitch"] = np.clip(camera["pitch"], -89, 89)
        view_matrix = math.fps_view(
            camera["eye"], np.radians(camera["yaw"]), np.radians(camera["pitch"])
        )
        # terminal characters are not square, so adjust aspect ratio to compensate
        view_matrix = view_matrix @ math.scaling(2.0, 1.0, 1.0)
        view_coords = world_coords @ view_matrix

        z = view_coords[:, 2]
        clipping_mask = (z > camera["near"]) & (z < camera["far"])
        view_coords = view_coords[clipping_mask]
        z = z[clipping_mask]

        # perspective projection
        proj_matrix = math.perspective(
            fov=np.radians(camera["fov"]),
            near=camera["near"],
            far=camera["far"],
            aspect=SCREEN_W / SCREEN_H,
        )
        clip_coords = view_coords @ proj_matrix
        clip_coords = clip_coords[:, :3] / clip_coords[:, 3:4]

        # viewport transformation
        viewport = np.full((SCREEN_H, SCREEN_W), " ")

        for face in model["faces"]:
            try:
                v1_idx, v2_idx, v3_idx = face[:, 0]
                v1 = clip_coords[v1_idx]
                v2 = clip_coords[v2_idx]
                v3 = clip_coords[v3_idx]
            except IndexError:
                continue

            center = (v1 + v2 + v3) / 3.0
            sx = int((center[0] + 1) * SCREEN_W / 2.0)
            sy = int((1 - center[1]) * SCREEN_H / 2.0)
            if not (0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H):
                continue

            n1_idx, n2_idx, n3_idx = face[:, 1]
            n1_intesity = np.dot(model["normals"][n1_idx], light_source["direction"])
            n2_intesity = np.dot(model["normals"][n2_idx], light_source["direction"])
            n3_intesity = np.dot(model["normals"][n3_idx], light_source["direction"])
            intensity = (n1_intesity + n2_intesity + n3_intesity) / 3.0
            if intensity < 0:  # backface culling
                continue

            shade_chars = " ░▒▓█"
            shade_index = min(int(len(shade_chars) * intensity), len(shade_chars) - 1)
            viewport[sy, sx] = shade_chars[shade_index]

        display(viewport, camera=camera)

        elapsed = time.time() - now
        sleep_time = TARGET_FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def display(viewport: np.ndarray, camera: dict = None, dt: float = None) -> None:
    lines = ["".join(c for c in row) for row in viewport]
    message = "Graphicspipe - Press ESC to quit"
    lines[SCREEN_H - 1] = message.ljust(SCREEN_W)

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
    lines[0] = debug_info.ljust(SCREEN_W)

    frame = "\n".join(lines)
    sys.stdout.write("\x1b[H" + frame)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
