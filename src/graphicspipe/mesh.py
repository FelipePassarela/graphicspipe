import numpy as np


def parse(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    normals = []
    faces = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z, 1])
            elif line.startswith("vn "):
                parts = line.strip().split()
                nx, ny, nz = map(float, parts[1:4])
                normals.append([nx, ny, nz, 0])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    vals = part.split("/")
                    v_idx = int(vals[0]) - 1
                    n_idx = int(vals[2]) - 1 if len(vals) >= 3 and vals[2] else -1
                    face.append([v_idx, n_idx])
                # triangulate if face has more than 3 vertices
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])

    vertices = np.array(vertices, dtype=float)
    normals = np.array(normals, dtype=float)
    faces = np.array(faces, dtype=int)
    return vertices, normals, faces


def torus(R: float, r: float, nrings: int = 15, tube_vertices: int = 30) -> np.ndarray:
    u_thetas = np.linspace(0, 2 * np.pi, nrings + 1)
    v_thetas = np.linspace(0, 2 * np.pi, tube_vertices + 1)
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
