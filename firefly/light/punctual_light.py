import numpy as np

from firefly.type import Scene


def directional_light(position: np.ndarray, scene: Scene):
    visibility_ray = (position, np.full_like(position, [0, 1, 0]))
    ray_test = scene(0, 5000, visibility_ray)
    color = np.where(np.isfinite(ray_test.intersection)[:, np.newaxis], [0, 0, 0], [1, 1, 1])
    return color
