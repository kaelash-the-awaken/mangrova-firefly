import numpy as np

from firefly.type import Sampler, Scene, CameraRayGenerator, Camera
from firefly.integrator import integrator


def random_sampler(nb_sample: int) -> Sampler:
    def __sample(pixel_width: float, pixel_height: float, uv_screen_pos: np.ndarray) -> np.ndarray:
        random_arrays = [np.random.random((uv_screen_pos.shape[0], 3)) - 0.5 for _ in range(nb_sample)]
        pixel_size = [pixel_width, pixel_height, 0]

        resampled_arrays = tuple(uv_screen_pos + pixel_size * random_arrays[i] for i in range(nb_sample))
        resampled_arrays = np.stack(resampled_arrays, axis=1)

        return resampled_arrays.reshape(-1, 3)

    def __recompose(image_data: np.ndarray) -> np.ndarray:
        return np.mean(image_data.reshape((-1, nb_sample, 3)), axis=1)

    return __sample, __recompose


def _generic_camera(width: int, height: int, sampler: Sampler, generator: CameraRayGenerator) -> Camera:
    def __generate_image(scene: Scene):
        # First we compute the center of each pixel on the screen
        half_pix_width = np.array([1 / width * 0.5, 0, 0])
        u = np.linspace([1, 0, 0] - half_pix_width, half_pix_width, width)
        half_pix_height = np.array([0, 1 / height * 0.5, 0])
        v = np.linspace([0, 1, 0] - half_pix_height, half_pix_height, height)
        uv_screen_pos = np.tile(u, (height, 1)) + np.repeat(v, width, axis=0)

        pixel_sample = sampler[0](1 / width, 1 / height, uv_screen_pos)
        camera_rays = generator(pixel_sample)
        received_lights = integrator(camera_rays, scene)

        received_lights[np.isnan(received_lights)] = 0.0

        return (sampler[1](received_lights) * 255).astype(int)

    return __generate_image
