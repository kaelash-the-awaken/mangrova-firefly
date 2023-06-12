import math

import numpy as np

from type import Scene, SurfaceData, Ray


def sphere(center, radius) -> Scene:
    def __compute_sphere_intersection(t_min: float, t_max: float, ray: Ray):
        origin, direction = ray

        oc = origin - center
        a = np.einsum("ij,ij->i", direction, direction)
        half_b = np.einsum("ij,ij->i", direction, oc)
        c = np.einsum("ij,ij->i", oc, oc) - radius * radius
        discriminant = half_b * half_b - a * c

        discriminant[discriminant < 0] = math.nan

        intersection = (-half_b - np.sqrt(discriminant)) / a
        intersection[(t_min >= intersection) | (intersection >= t_max) | np.isnan(intersection)] = np.inf

        position = origin + np.expand_dims(intersection, axis=1) * direction

        normal = (position - center) / radius
        invert_normal = np.einsum("ij, ij->i", normal, direction) > 0
        normal[invert_normal] = -normal[invert_normal]

        return SurfaceData(position, normal, intersection)

    return __compute_sphere_intersection
