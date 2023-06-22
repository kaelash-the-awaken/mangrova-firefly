import math

import numpy as np

from firefly import Material


def compute_random_vector(size):
    theta = np.random.random(size) * 2.0 * math.pi
    phi = np.random.random(size) * math.pi
    radius = np.expand_dims(np.random.random(size), axis=1)

    sin_theta = np.sin(theta)
    return np.vstack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta))).T * radius

def transparent_material(color, ior, nb_sample=1) -> Material:
    def __rho(position, normal, wi, integrator, __step):
        def __compute_light_contribution():
            cos_theta_i = np.einsum("ij,ij->i", wi, normal)
            sin_theta_t = (1.0 / ior) * np.sin(np.arccos(cos_theta_i))

            r0 = ((1 - ior) / (1 + ior)) ** 2
            schlick_approx = r0 + (1 - r0) * pow((1 + cos_theta_i), 5)

            refracted_ray = (ior * wi) + ((ior * cos_theta_i - np.sqrt(1.0 - sin_theta_t ** 2))[:, None] * normal)
            refracted_ray /= np.linalg.norm(refracted_ray, axis=1)[:, np.newaxis]

            reflected_ray = wi - 2 * cos_theta_i[:, np.newaxis] * normal
            reflected_ray /= np.linalg.norm(reflected_ray, axis=1)[:, np.newaxis]

            mask = (sin_theta_t > 1.0) | (schlick_approx > np.random.random(cos_theta_i.shape))
            secondary_ray = np.where(mask[:, None], reflected_ray, refracted_ray)
            return integrator((position, secondary_ray), __step=__step + 1)

        return np.sum((__compute_light_contribution() for _ in range(nb_sample)), axis=0) * color * (1 / nb_sample)

    def __rho_sample(position, normal, wo):
        return np.full_like(position, 0)

    return __rho, __rho_sample
