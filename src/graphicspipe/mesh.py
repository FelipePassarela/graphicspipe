import numpy as np


def parse(filename: str) -> np.ndarray:
    vertices = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z, 1])
    return np.array(vertices)


def torus(R: float, r: float, nrings: int = 15, tube_vertices: int = 30) -> np.ndarray:
    u_thetas = np.linspace(0, 2 * np.pi, nrings)
    v_thetas = np.linspace(0, 2 * np.pi, tube_vertices)
    vertices = []
    for u in u_thetas:
        for v in v_thetas:
            cos_u, cos_v = np.cos(u), np.cos(v)
            sin_u, sin_v = np.sin(u), np.sin(v)
            x = (R + r * cos_v) * cos_u
            y = (R + r * cos_v) * sin_u
            z = r * sin_v
            vertices.append([x, y, z, 1])
    return np.array(vertices)
