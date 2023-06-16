import time

import numpy as np

from firefly.type import Sampler, CameraRayGenerator, Camera, Integrator


def random_sampler() -> Sampler:
    def __sample(pixel_width: float, pixel_height: float, uv_screen_pos: np.ndarray) -> np.ndarray:
        random_vectors = np.random.random((uv_screen_pos.shape[0], 3)) - 0.5
        return uv_screen_pos + [pixel_width, pixel_height, 0] * random_vectors

    return __sample


def _generic_camera(width: int, height: int, sampler: Sampler, generator: CameraRayGenerator) -> Camera:
    def __generate_image(integrator: Integrator):
        # First we compute the center of each pixel on the screen
        half_pix_width = np.array([1 / width * 0.5, 0, 0])
        u = np.linspace([1, 0, 0] - half_pix_width, half_pix_width, width)
        half_pix_height = np.array([0, 1 / height * 0.5, 0])
        v = np.linspace([0, 1, 0] - half_pix_height, half_pix_height, height)
        uv_screen_pos = np.tile(u, (height, 1)) + np.repeat(v, width, axis=0)

        samples = [integrator(generator(sampler(1 / width, 1 / height, uv_screen_pos)), 0) for _ in range(10)]
        result = np.sum(samples, axis=0) * 1 / 10
        return (result * 255).astype(int)

    return __generate_image
