import functools
from typing import List

import numpy as np

import firefly.renderer.integrator.simple_path as simple_path
from firefly.type import Scene, Camera, SurfaceData, Ray, Hittable, Material, Light


def aggregate_lights(light_list: List[Light]) -> Light:
    def __compute_light_contribution(position, normal, material: Material, scene: Scene):
        return np.sum((light(position, normal, material, scene) for light in light_list), axis=0)

    return __compute_light_contribution


def create_scene(scene_content: List[Hittable]) -> Scene:
    def __test_ray_intersection(t_min: float, t_max: float, ray: Ray):
        def __reduce_object_array(surface_data: SurfaceData, element: Hittable):
            surface_data_object = element(t_min, t_max, ray)

            if not surface_data:
                return surface_data_object

            visible_mask = surface_data_object.intersection < surface_data.intersection
            surface_data.intersection[visible_mask] = surface_data_object.intersection[visible_mask]
            surface_data.position[visible_mask] = surface_data_object.position[visible_mask]
            surface_data.normal[visible_mask] = surface_data_object.normal[visible_mask]

            nb_material = len(surface_data.material_list)
            surface_data.material_id[visible_mask] = surface_data_object.material_id[visible_mask] + nb_material
            surface_data.material_list += surface_data_object.material_list
            return surface_data

        init = SurfaceData(
            np.zeros_like(ray[0]), np.zeros_like(ray[0]),
            np.zeros(ray[0].shape[0]), [],
            np.full(ray[0].shape[0], np.inf)
        )
        return functools.reduce(__reduce_object_array, scene_content, init)

    return __test_ray_intersection


def Renderer(background_color=[0, 0, 0]):
    def __internal_function(camera: Camera, scene_content: List[Hittable], light_content: List[Light] = []):
        scene = create_scene(scene_content)
        light = aggregate_lights(light_content)
        return camera(simple_path.Integrator(scene, light))

    return __internal_function
