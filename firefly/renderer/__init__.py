import firefly.renderer.integrator.simple_path as simple_path
from firefly.type import Scene, Camera


def Renderer(background_color=[0, 0, 0]):
    def __internal_function(camera: Camera, scene: Scene):
        return camera(simple_path.integrator(scene))

    return __internal_function
