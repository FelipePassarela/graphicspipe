import os
import time

import numpy as np
from pynput import keyboard

from graphicspipe import math, mesh
from graphicspipe.input_state import InputState
from graphicspipe.renderer import display, render_faces

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
        "yaw": 0.0,
        "pitch": -15.0,
        "eye": np.array([0.0, 0.4, -0.7]),  # move back to see the object
        "up": np.array([0.0, 1.0, 0.0]),
        "near": 0.1,
        "far": 100.0,
        "fov": 60.0,
    }

    light_source = {
        "position": np.array([-1.0, -5.0, 5.0]),
        "direction": np.array([0.0, 0.0, 1.0]),
    }
    light_source["direction"] = np.array([0.0, 0.0, 0.0]) - light_source["position"]
    light_source["direction"] /= np.linalg.norm(light_source["direction"])

    input_state = InputState()
    key_listener = keyboard.Listener(
        on_press=input_state.on_press, on_release=input_state.on_release
    )
    key_listener.start()

    os.system("cls" if os.name == "nt" else "clear")
    last_time = time.time()

    viewport = np.full((SCREEN_H, SCREEN_W), 0)
    z_buffer = np.full((SCREEN_H, SCREEN_W), np.inf)

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

        model["rotation"][1] += 45.0 * dt  # auto-rotate model
        camera["pitch"] = np.clip(camera["pitch"], -89, 89)

        # Compute matrices
        model_matrix = math.compose(
            translations=model["translation"],
            rotations=[np.radians(a) for a in model["rotation"]],
            scales=model["scale"],
        )
        view_matrix = math.fps_view(
            eye=camera["eye"],
            yaw=np.radians(camera["yaw"]),
            pitch=np.radians(camera["pitch"]),
        )
        proj_matrix = math.perspective(
            fov=np.radians(camera["fov"]),
            near=camera["near"],
            far=camera["far"],
            # terminal characters are not square, so adjust aspect ratio to compensate
            aspect=SCREEN_W / (2.0 * SCREEN_H),
        )
        mvp_matrix = model_matrix @ view_matrix @ proj_matrix

        clip_coords = model["mesh"] @ mvp_matrix

        normals = (
            model["normals"]
            @ math.rotation_x(np.radians(model["rotation"][0]))
            @ math.rotation_y(np.radians(model["rotation"][1]))
            @ math.rotation_z(np.radians(model["rotation"][2]))
        )

        viewport.fill(0)
        z_buffer.fill(np.inf)
        render_faces(
            viewport,
            z_buffer,
            model["faces"],
            clip_coords,
            normals,
            light_source["direction"],
            SCREEN_W,
            SCREEN_H,
        )

        display(viewport, SCREEN_W, SCREEN_H, camera=camera, dt=dt)


if __name__ == "__main__":
    main()
