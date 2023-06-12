import math

import numpy as np

from type import Sampler, CameraRayGenerator, Camera, Scene, Ray, SurfaceData


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


def __sample_diffuse(normal):
    def compute_random_vector(size):
        theta = np.random.random(size) * 2.0 * math.pi
        phi = np.random.random(size) * math.pi
        radius = np.expand_dims(np.random.random(size), axis=1)

        sin_theta = np.sin(theta)
        return np.vstack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta))).T * radius

    secondary_direction = normal + compute_random_vector(normal.shape[0])
    return secondary_direction / np.expand_dims(np.linalg.norm(secondary_direction, axis=1), axis=1)


def integrator(ray: Ray, scene: Scene, __step=0) -> 'Light':
    def __reflected_radiance(surface_data: SurfaceData, out_direction) -> 'Light':
        cumulated_light = np.zeros_like(surface_data.position)

        # We compute the emitted light for each point and add them
        emitted_light = np.zeros_like(surface_data.position)
        cumulated_light += emitted_light

        # Then we compute secondary ray and cumulate them
        mask = np.isfinite(surface_data.intersection)

        secondary_ray_position = surface_data.position[mask]
        brdf = np.ones_like(secondary_ray_position) / math.pi

        cumulated_light[mask] = [1,0,0] * brdf * \
                                np.expand_dims(np.clip(np.dot(surface_data.normal[mask], [0, 1, 0]), 0, None), axis=1)

        secondary_ray = (secondary_ray_position, __sample_diffuse(surface_data.normal[mask]))
        blop = integrator(secondary_ray, scene, __step=__step + 1) * brdf
        cumulated_light[mask] += blop

        cumulated_light[~mask] = 1.0
        return cumulated_light

    if __step > 2:
        return np.zeros_like(ray[1])

    environment = 1.0
    return __reflected_radiance(scene(0, 5000, ray), -ray[1]) * environment


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
