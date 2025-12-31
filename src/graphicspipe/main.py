import os
import time

import numpy as np

from graphicspipe import math, mesh

SCREEN_W = 120
SCREEN_H = 40
FRAME_DELAY = 1 / 60.0


def main() -> None:
    local_coords = mesh.torus(1, 0.3, nrings=15, tube_vertices=30)

    translation = (0, 0, 1.5)
    scale = 1.0 / np.max(np.abs(local_coords))
    rotation_y = 0.0

    while True:
        rotation_y += np.radians(180) * FRAME_DELAY

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

        viewport = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
        viewport[:] = ord(" ")
        for x, y in screen_coords:
            if 0 <= x < SCREEN_W and 0 <= y < SCREEN_H:
                viewport[y, x] = ord("*")

        display(viewport)

        time.sleep(FRAME_DELAY)


def display(viewport):
    os.system("cls" if os.name == "nt" else "clear")
    for row in viewport:
        for c in row:
            print(chr(c), end="")
        print()


if __name__ == "__main__":
    main()
