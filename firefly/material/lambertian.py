import math

import numpy as np


def __sample_diffuse(normal):
    def compute_random_vector(size):
        theta = np.random.random(size) * 2.0 * math.pi
        phi = np.random.random(size) * math.pi
        radius = np.expand_dims(np.random.random(size), axis=1)

        sin_theta = np.sin(theta)
        return np.vstack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta))).T * radius

    secondary_direction = normal + compute_random_vector(normal.shape[0])
    return secondary_direction / np.clip(np.linalg.norm(secondary_direction, axis=1)[:, np.newaxis], 0.001, None)


def lambertian_material(color, nb_sample=1):
    def __rho(position, normal, wi, integrator, __step):
        def __compute_light():
            directions = __sample_diffuse(normal)
            return integrator((position, directions), __step=__step + 1)

        return np.sum((__compute_light() for _ in range(nb_sample)), axis=0) * color * (1 / nb_sample)

    def __rho_sample(position, normal, direction):
        return color * np.clip(np.einsum("ij,ij->i", normal, direction), 0, None)[:, np.newaxis] / math.pi

    return __rho, __rho_sample
