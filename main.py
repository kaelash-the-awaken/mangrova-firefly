from PIL import Image

from camera.orthographic_camera import orthographic_camera
from hitable import merge_scene_content
from hitable.sphere import sphere
from type import Camera, Scene


def render(width: int, height: int):
    camera: Camera = orthographic_camera(width, height, 1, position=[0, 0, 2], direction=[0, 0, -1])
    scene: Scene = merge_scene_content(sphere([0, 0, 0], 1), sphere([0, -50, 0], 49))
    image_data_int = camera(scene)

    result_image = Image.new("RGB", (width, height))
    result_image.putdata(list(zip(image_data_int[:, 0], image_data_int[:, 1], image_data_int[:, 2])))
    return result_image


if __name__ == '__main__':
    rendered_image = render(256, 256)
    rendered_image.show()
