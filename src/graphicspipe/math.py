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


def compose(translations: tuple, rotations: tuple, scales: tuple) -> np.ndarray:
    T = translation(*translations)
    Rx = rotation_x(rotations[0])
    Ry = rotation_y(rotations[1])
    Rz = rotation_z(rotations[2])
    S = scaling(*scales)
    return S @ Rx @ Ry @ Rz @ T


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.astype(eye, float)
    target = np.astype(target, float)
    up = np.astype(up, float)

    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    view_matrix = np.array(
        [
            [x[0], y[0], z[0], 0],
            [x[1], y[1], z[1], 0],
            [x[2], y[2], z[2], 0],
            [-np.dot(x, eye), -np.dot(y, eye), -np.dot(z, eye), 1],
        ]
    )
    return view_matrix


def forward(yaw: float, pitch: float) -> np.ndarray:
    return np.array(
        [
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        ]
    )
