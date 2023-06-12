# Firefly Ray Tracer

Firefly is a Python-based ray tracer implemented using numpy. It allows you to create stunning 3D images by simulating
the behavior of light and how it interacts with objects in a scene.

## Installation

Firefly requires Python 3.10 or higher. You can install it using pipenv, which manages project dependencies and virtual
environments.

1. Clone the repository:
   ```shell
   $ git clone https://github.com/kaelash-the-awaken/mangrova-firefly.git
   $ cd mangrova-firefly
   ```

2. Install pipenv if you haven't already:
   ```shell
   $ pip install pipenv
   ```

3. Install project dependencies:
   ```shell
   $ pipenv install
   ```

4. Activate the virtual environment:
   ```shell
   $ pipenv shell
   ```

## Usage

To render a scene with Firefly, you need to create a Python script and define the scene, camera, objects, and lights.
Here's a simple example:

``` python
import firefly

# Create the scene
scene = firefly.Scene(background_color=(0.2, 0.2, 0.2))

# Create objects
sphere = firefly.Sphere(center=(0, 0, 0), radius=1, material=firefly.Material(diffuse_color=(0.8, 0.2, 0.2)))
plane = firefly.Plane(normal=(0, 1, 0), distance=0, material=firefly.Material(diffuse_color=(0.5, 0.5, 0.5)))

# Add objects to the scene
scene.add_object(sphere)
scene.add_object(plane)

# Create a camera
camera = firefly.Camera(position=(0, 0, -5), look_at=(0, 0, 0), field_of_view=60)

# Create a light source
light = firefly.PointLight(position=(10, 10, -10), color=(1, 1, 1))

# Add light source to the scene
scene.add_light(light)

# Render the scene
image = firefly.render(scene, camera)

# Save the rendered image
image.save("output.png")
``` 

Save the above code to a Python script (e.g., render.py), and run it using the following command:

```shell
$ python render.py
```
The rendered image will be saved as output.png in the current directory.

# Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull
request.

Please make sure to follow the code of conduct and guidelines specified in the CONTRIBUTING file.

# License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.