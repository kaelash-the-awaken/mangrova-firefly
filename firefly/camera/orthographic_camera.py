import math

import numpy as np

from firefly.type import Camera
from .abstract_camera import random_sampler, _generic_camera


def orthographic_camera(width: int, height: int, focal: float, position=None, direction=None) -> Camera:
    direction = np.array([0, 0, 1] if direction is None else direction)
    position = np.array([0, 0, 0] if position is None else position)

    def __compute_ray(uv_screen_pos: np.matrix):
        screen_width = math.tan(90 / 2) * focal * 2.0
        screen_height = screen_width * float(height / width)

        # We compute each pixel position from the screen UV to world space
        side = np.cross([0, 1, 0], direction)
        up = np.cross(direction, side)
        screen_matrix = np.vstack((side * screen_width, up * screen_height, direction)).T
        screen_origin = position + (direction * focal) - 0.5 * (side * screen_width + up * screen_height)
        camera_px_pos = screen_origin + uv_screen_pos @ screen_matrix

        # Now we can compute the direction of each ray
        ray_direction = camera_px_pos - position
        ray_direction /= np.expand_dims(np.linalg.norm(ray_direction, axis=1), axis=1)
        return np.ones_like(ray_direction) * position, ray_direction

    return _generic_camera(width, height, random_sampler(20), __compute_ray)
