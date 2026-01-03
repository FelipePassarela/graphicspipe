import numpy as np


def rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1],
        ]
    )


def scaling(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ]
    )


def compose(
    translations: tuple = (0, 0, 0),
    rotations: tuple = (0, 0, 0),
    scales: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    T = translation(*translations)
    Rx = rotation_x(rotations[0])
    Ry = rotation_y(rotations[1])
    Rz = rotation_z(rotations[2])
    S = scaling(*scales)
    return S @ Rx @ Ry @ Rz @ T


def fps_view(eye: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    T = translation(*-eye)
    Rx = rotation_x(-pitch)
    Ry = rotation_y(-yaw)
    return T @ Ry @ Rx


def forward(yaw: float, pitch: float) -> np.ndarray:
    forward = np.array([0.0, 0.0, 1.0, 0.0])
    forward = forward @ rotation_x(pitch) @ rotation_y(yaw)
    forward = forward[:-1] / np.linalg.norm(forward)
    return forward
