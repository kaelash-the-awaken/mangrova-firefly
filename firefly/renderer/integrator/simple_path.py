import numpy as np

from firefly.type import Scene, Integrator, SurfaceData, Ray, Light


def Integrator(scene: Scene, light: Light) -> Integrator:
    def compute_material(surface_data: SurfaceData, mask: np.ndarray, integrator: Integrator, __step=0):
        position = surface_data.position[mask]
        normal = surface_data.normal[mask]

        if not np.any(position):
            return 0.0

        cumulated_light = np.zeros((position.shape[0], 3))
        for material_id, (rho, _) in enumerate(surface_data.material_list):
            material_mask = np.equal(surface_data.material_id[mask], material_id)
            cumulated_light[material_mask] += rho(position[material_mask], normal[material_mask], integrator, __step) \
                if np.any(material_mask) else 0.0

        return cumulated_light

    def compute_material_reaction_light(surface_data: SurfaceData, mask: np.ndarray):
        position = surface_data.position[mask]
        normal = surface_data.normal[mask]

        if not np.any(position):
            return 0.0

        cumulated_light = np.zeros((position.shape[0], 3))
        for material_id, material in enumerate(surface_data.material_list):
            mat_mask = np.equal(surface_data.material_id[mask], material_id)
            cumulated_light[mat_mask] += light(position[mat_mask], normal[mat_mask], material, scene)

        return cumulated_light

    def __integrate(ray: Ray, __step=0):
        if __step > 2:
            return np.zeros_like(ray[1])

        surface_data = scene(0, 5000, ray)
        cumulated_light = np.full_like(surface_data.position, [0.1, 0.1, 0.1])

        mask = np.isfinite(surface_data.intersection)
        cumulated_light[mask] = compute_material(surface_data, mask, __integrate, __step)
        cumulated_light[mask] += compute_material_reaction_light(surface_data, mask)

        environment = 1.0
        return cumulated_light * environment

    return __integrate
