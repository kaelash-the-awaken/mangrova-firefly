import math

import numpy as np

from firefly.type import Ray, Scene, SurfaceData


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

        cumulated_light[mask] = [1, 0, 0] * brdf * \
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
