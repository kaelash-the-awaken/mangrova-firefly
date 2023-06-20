import numpy as np

from firefly.type import Scene, Material, Light


def PointLight(light_position, light_color=None) -> Light:
    if light_color is None:
        light_color = [1, 1, 1]

    def __compute_light_contribution(position: np.ndarray, normal: np.ndarray, material: Material, scene: Scene):
        wi = light_position - position
        wi /= np.linalg.norm(wi, axis=1)[:, np.newaxis]

        ray_test = scene(0, 5000, (position, wi))
        color = np.where(np.isfinite(ray_test.intersection)[:, np.newaxis], [0, 0, 0], light_color)
        return material[1](position, normal, wi) * color

    return __compute_light_contribution


def DirectionalLight(direction, light_color=None) -> Light:
    if light_color is None:
        light_color = [1, 1, 1]

    def __compute_light_contribution(position: np.ndarray, normal: np.ndarray, material: Material, scene: Scene):
        wi = np.full_like(position, direction)
        ray_test = scene(0, 5000, (position, wi))
        color = np.where(np.isfinite(ray_test.intersection)[:, np.newaxis], [0, 0, 0], light_color)
        return material[1](position, normal, wi) * color

    return __compute_light_contribution
