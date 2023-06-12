from PIL import Image

import firefly


def render(width: int, height: int):
    camera: firefly.Camera = firefly.orthographic_camera(width, height, 1, position=[0, 0, 2], direction=[0, 0, -1])
    scene: firefly.Scene = firefly.merge_scene_content(firefly.sphere([0, 0, 0], 1), firefly.sphere([0, -50, 0], 49))
    image_data_int = camera(scene)

    result_image = Image.new("RGB", (width, height))
    result_image.putdata(list(zip(image_data_int[:, 0], image_data_int[:, 1], image_data_int[:, 2])))
    return result_image


if __name__ == '__main__':
    rendered_image = render(256, 256)
    rendered_image.show()
