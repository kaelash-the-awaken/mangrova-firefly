from typing import Tuple, Callable, List

import numpy as np

Ray = Tuple[np.ndarray, np.ndarray]
Integrator = Callable[[Ray, int], np.ndarray]
Material = Tuple[
    Callable[[np.ndarray, np.ndarray, Integrator, int], np.ndarray],
    Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
]


class SurfaceData:
    def __init__(self, position, normal, material_id, material_list: List[Material], intersection):
        self.position = position
        self.normal = normal
        self.material_id = material_id
        self.intersection = intersection
        self.material_list = material_list


Scene = Callable[[float, float, Ray], SurfaceData]
Camera = Callable[[Integrator], np.ndarray]
CameraRayGenerator = Callable[[np.ndarray], Ray]
Sampler = Callable[[float, float, np.ndarray], np.ndarray]
