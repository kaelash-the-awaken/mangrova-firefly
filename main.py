from PIL import Image

import firefly


def render(width: int, height: int):
    renderer = firefly.Renderer(background_color=[0.5, 0.5, 1.0])
    camera: firefly.Camera = firefly.orthographic_camera(width, height, 1, position=[0, 0, 2], direction=[0, 0, -1])
    scene = [
        firefly.sphere([2, 0, 0], 1, material=firefly.material.lambertian_material([0.8, 0.8, 0.8])),
        firefly.sphere([0, 0, 0], 1, material=firefly.material.transparent_material([1, 1, 1], 1.5)),
        firefly.sphere([0, -50, 0], 49, material=firefly.material.lambertian_material([1, 0, 0]))
    ]
    lights = [firefly.PointLight([10, 10, 10])]
    image_data_int = renderer(camera, scene, lights)

    result_image = Image.new("RGB", (width, height))
    result_image.putdata(list(zip(image_data_int[:, 0], image_data_int[:, 1], image_data_int[:, 2])))
    return result_image


if __name__ == '__main__':
    rendered_image = render(1024, 1024)
    rendered_image.show()
