import os
import sys
import time

import numpy as np
from pynput import keyboard

from graphicspipe import math, mesh
from graphicspipe.input_state import InputState
from graphicspipe.torus_controller import TorusController

SCREEN_W = 120
SCREEN_H = 40
FRAME_DELAY = 1 / 60.0

MOVE_SPEED = 2.0
CAMERA_ROTATION_SPEED = 45.0


def main() -> None:
    torus_controller = TorusController(interval=3)

    model = {
        "mesh": mesh.parse("assets/plane.obj"),
        "translation": np.array([0.0, 0.0, 1.8]),
        "scale": np.array([1.0, 1.0, 1.0]),
        "rotation": np.array([90.0, 0.0, 0.0]),
    }
    model["scale"] /= np.max(np.abs(model["mesh"][:, :3]))  # normalize size

    camera = {
        "yaw": 0.0,
        "pitch": 0.0,
        "eye": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
        "near": 0.1,
        "far": 100.0,
        "fov": 60.0,
    }

    input_state = InputState()
    key_listener = keyboard.Listener(
        on_press=input_state.on_press, on_release=input_state.on_release
    )
    key_listener.start()

    os.system("cls" if os.name == "nt" else "clear")

    while True:
        # model["rotation"][1] +=wswsws np.radians(180) * FRAME_DELAY

        forward = math.forward(np.radians(camera["yaw"]), np.radians(camera["pitch"]))
        right = np.cross(camera["up"], forward)
        right /= np.linalg.norm(right)

        if input_state.is_pressed("w"):
            camera["eye"] += forward * MOVE_SPEED * FRAME_DELAY
        if input_state.is_pressed("s"):
            camera["eye"] -= forward * MOVE_SPEED * FRAME_DELAY
        if input_state.is_pressed("a"):
            camera["eye"] -= right * MOVE_SPEED * FRAME_DELAY
        if input_state.is_pressed("d"):
            camera["eye"] += right * MOVE_SPEED * FRAME_DELAY

        if input_state.is_pressed(keyboard.Key.left):
            camera["yaw"] += CAMERA_ROTATION_SPEED * FRAME_DELAY
        if input_state.is_pressed(keyboard.Key.right):
            camera["yaw"] -= CAMERA_ROTATION_SPEED * FRAME_DELAY
        if input_state.is_pressed(keyboard.Key.up):
            camera["pitch"] += CAMERA_ROTATION_SPEED * FRAME_DELAY
        if input_state.is_pressed(keyboard.Key.down):
            camera["pitch"] -= CAMERA_ROTATION_SPEED * FRAME_DELAY
        if input_state.is_pressed("+"):
            camera["fov"] = min(150.0, camera["fov"] + 20.0 * FRAME_DELAY)
        if input_state.is_pressed("-"):
            camera["fov"] = max(60.0, camera["fov"] - 20.0 * FRAME_DELAY)

        if input_state.is_pressed(keyboard.Key.esc):
            key_listener.stop()
            exit()

        # model["mesh"] = torus_controller.update(model["mesh"])

        world_matrix = math.compose(
            translations=model["translation"],
            rotations=model["rotation"],
            scales=model["scale"],
        )
        world_coords = model["mesh"] @ world_matrix  # objects in row vector format

        # view transformation
        camera["pitch"] = np.clip(camera["pitch"], -89, 89)
        view_matrix = math.fps_view(
            camera["eye"], np.radians(camera["yaw"]), np.radians(camera["pitch"])
        )
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
        sx = (clip_coords[:, 0] + 1) * SCREEN_W / 2.0
        sy = (1 - clip_coords[:, 1]) * SCREEN_H / 2.0
        screen_coords = np.stack([sx, sy], axis=1)
        screen_coords = np.round(screen_coords).astype(np.int32)

        viewport = np.full((SCREEN_H, SCREEN_W), ord(" "), dtype=np.uint8)
        xs = screen_coords[:, 0]
        ys = screen_coords[:, 1]
        mask = (xs >= 0) & (xs < SCREEN_W) & (ys >= 0) & (ys < SCREEN_H)
        viewport[ys[mask], xs[mask]] = ord("*")

        display(viewport, camera=camera)

        time.sleep(FRAME_DELAY)


def display(viewport: np.ndarray, camera: dict = None):
    lines = [bytes(row).decode("ascii") for row in viewport]
    if camera is not None:
        camera_x = camera["eye"][0]
        camera_y = camera["eye"][1]
        camera_z = camera["eye"][2]
        lines.insert(
            0,
            f"Camera Position: ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f}) "
            f"Yaw: {camera['yaw']:.2f} Pitch: {camera['pitch']:.2f} "
            f"FOV: {camera['fov']:.2f}",
        )
    frame = "\n".join(lines)
    sys.stdout.write("\x1b[H" + frame)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
