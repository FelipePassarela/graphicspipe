import os
import sys
import time

import numpy as np

from graphicspipe import math
from graphicspipe.torus_controller import TorusController

SCREEN_W = 120
SCREEN_H = 40
FRAME_DELAY = 1 / 60.0


def main() -> None:
    torus_controller = TorusController(interval=3)
    local_coords = torus_controller.create_torus()

    translation = (0, 0, 1.5)
    scale = 1.0 / np.max(np.abs(local_coords))
    rotation_y = 0.0

    os.system("cls" if os.name == "nt" else "clear")
    while True:
        rotation_y += np.radians(180) * FRAME_DELAY

        local_coords = torus_controller.update(local_coords)

        world_matrix = math.compose(
            translations=translation,
            rotations=(0, rotation_y, 0),
            scales=(scale, scale, scale),
        )
        world_coords = local_coords @ world_matrix  # objects in row vector format

        # perspective projection
        view_coords = world_coords.copy()
        z = np.expand_dims(view_coords[:, 2], axis=1)
        xy = view_coords[:, :2]
        projected = xy / z

        # viewport transformation
        sx = (projected[:, 0] + 1) * SCREEN_W / 2.0
        sy = (1 - projected[:, 1]) * SCREEN_H / 2.0
        screen_coords = np.stack([sx, sy], axis=1)
        screen_coords = np.round(screen_coords).astype(np.int32)

        viewport = np.full((SCREEN_H, SCREEN_W), ord(" "), dtype=np.uint8)
        xs = screen_coords[:, 0]
        ys = screen_coords[:, 1]
        mask = (xs >= 0) & (xs < SCREEN_W) & (ys >= 0) & (ys < SCREEN_H)
        viewport[ys[mask], xs[mask]] = ord("*")

        display(viewport)

        time.sleep(FRAME_DELAY)


def display(viewport):
    lines = [bytes(row).decode("ascii") for row in viewport]
    frame = "\n".join(lines)
    sys.stdout.write("\x1b[H" + frame)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
