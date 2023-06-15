import math

import numpy as np

from firefly.material import lambertian_material
from firefly.type import Scene, SurfaceData, Ray


def sphere(center, radius, color=[0.5, 0.5, 0.5]) -> Scene:
    material = lambertian_material(color)

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

        position = origin + np.where(np.isfinite(intersection), intersection, 0.0)[:, np.newaxis] * direction

        normal = (position - center) / radius
        invert_normal = np.einsum("ij, ij->i", normal, direction) > 0
        normal[invert_normal] = -normal[invert_normal]

        return SurfaceData(position, normal, np.full_like(intersection, 0), [material], intersection)

    return __compute_sphere_intersection
