from typing import Tuple, Callable

import numpy as np

Ray = Tuple[np.ndarray, np.ndarray]


class SurfaceData:
    def __init__(self, position, normal, intersection):
        self.position = position
        self.normal = normal
        self.intersection = intersection


Scene = Callable[[float, float, Ray], SurfaceData]
Camera = Callable[[Scene], np.ndarray]
CameraRayGenerator = Callable[[np.ndarray], Ray]
Sampler = Tuple[Callable[[float, float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
