import time

import numpy as np

from graphicspipe import mesh


class TorusController:
    def __init__(self, interval: float):
        self._interval = interval
        self._previous_toggle = time.time()
        self._states = {
            "min": {"nrings": 10, "tube_vertices": 100, "next": "mid"},
            "mid": {"nrings": 20, "tube_vertices": 100, "next": "max"},
            "max": {"nrings": 300, "tube_vertices": 100, "next": "min_stripped"},
            "min_stripped": {
                "nrings": 300,
                "tube_vertices": 8,
                "next": "max_stripped",
            },
            "max_stripped": {"nrings": 300, "tube_vertices": 3, "next": "min"},
        }
        self._current_state = "min"

    def update(self, local_coords: np.ndarray) -> np.ndarray:
        now = time.time()
        if now - self._previous_toggle >= self._interval:
            self._previous_toggle = now
            self._current_state = self._next_state()
            return self.create_torus()
        return local_coords

    def _next_state(self) -> str:
        return self._states[self._current_state]["next"]

    def create_torus(self) -> np.ndarray:
        nrings = self._states[self._current_state]["nrings"]
        tube_vertices = self._states[self._current_state]["tube_vertices"]
        return mesh.torus(1, 0.3, nrings=nrings, tube_vertices=tube_vertices)
