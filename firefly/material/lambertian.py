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


def lambertian_material(color):
    def __rho(position, normal, integrator, __step):
        return color * integrator((position, __sample_diffuse(normal)), __step=__step + 1)

    def __rho_sample(position, normal, direction):
        return color * np.clip(np.einsum("ij,ij->i", normal, direction), 0, None)[:, np.newaxis] / math.pi

    return __rho, __rho_sample
