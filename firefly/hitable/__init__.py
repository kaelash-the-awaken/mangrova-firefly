from firefly.type import Scene, Ray

from .sphere import *


def merge_scene_content(main_scene: Scene, other: Scene):
    def __compute_intersection(t_min: float, t_max: float, ray: Ray):
        surface_data_scene = main_scene(t_min, t_max, ray)
        surface_data_object = other(t_min, t_max, ray)

        visible_mask = surface_data_object.intersection < surface_data_scene.intersection
        surface_data_scene.intersection[visible_mask] = surface_data_object.intersection[visible_mask]
        surface_data_scene.position[visible_mask] = surface_data_object.position[visible_mask]
        surface_data_scene.normal[visible_mask] = surface_data_object.normal[visible_mask]

        return surface_data_scene

    return __compute_intersection
