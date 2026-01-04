import os
import time

import numpy as np
from pynput import keyboard

from graphicspipe import math, mesh
from graphicspipe.input_state import InputState
from graphicspipe.renderer import display, reder_faces

SCREEN_W = 120
SCREEN_H = 40

MOVE_SPEED = 2.0
CAMERA_ROTATION_SPEED = 45.0
FOV_SPEED = 20.0


def main() -> None:
    model_mesh, normals, faces = mesh.parse("assets/plane.obj")

    # TODO: make these dicts be classes
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

        camera["pitch"] = np.clip(camera["pitch"], -89, 89)

        model_matrix = math.compose(
            translations=model["translation"],
            rotations=[np.radians(a) for a in model["rotation"]],
            scales=model["scale"],
        )
        view_matrix = math.fps_view(
            camera["eye"], np.radians(camera["yaw"]), np.radians(camera["pitch"])
        )
        # terminal characters are not square, so adjust aspect ratio to compensate
        view_matrix = view_matrix @ math.scaling(2.0, 1.0, 1.0)
        proj_matrix = math.perspective(
            fov=np.radians(camera["fov"]),
            near=camera["near"],
            far=camera["far"],
            aspect=SCREEN_W / SCREEN_H,
        )
        mvp_matrix = model_matrix @ view_matrix @ proj_matrix

        clip_coords = model["mesh"] @ mvp_matrix

        viewport = np.full((SCREEN_H, SCREEN_W), " ")
        shade_chars = np.array(list(" ░▒▓█"))
        sx_arr, sy_arr, shade_arr = reder_faces(
            model["faces"],
            clip_coords,
            model["normals"],
            light_source["direction"],
            SCREEN_W,
            SCREEN_H,
        )
        viewport[sy_arr, sx_arr] = shade_chars[shade_arr]

        display(viewport, SCREEN_W, SCREEN_H, camera=camera, dt=dt)


if __name__ == "__main__":
    main()
