# 3D Engine on Terminal

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FFelipePassarela%2F3d-engine-on-terminal%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

A minimalistic 3D engine that renders models in the terminal using unicode characters.
No external graphics libraries are used. The entire 3D rendering pipeline is implemented
from scratch, using only basic Python and NumPy for matrix operations.

## Features

- **3D Model Loading**: Supports loading 3D models from OBJ files.
- **Transformations**: Implements basic transformations: translation, rotation, and scaling.
- **Shading**: Simple shading based on the angle of the surface to a light source.
- **Terminal Rendering**: Renders the final output in the terminal using unicode characters.
- **Performance**: Optimized for real-time rendering in the terminal, achieving smooth animations.

## Usage

> [!IMPORTANT]
Before running the engine, resize the terminal window to a resolution of **120x40**
characters for proper display. You can do this through terminal settings or,
alternatively, simply run the engine and drag the window until the image is displayed
correctly.

Install the package and run the main script:

```sh
pip install .
graphicspipe
# uv sync
# uv run graphicspipe
```

## Controls

- **W/A/S/D**: Move the camera.
- **↑/←/↓/→/**: Rotate the camera.
- **R**: Rotate the model.
- **+/-**: Adjust FOV.
- **Esc**: Exit the engine.

## Project Demo

Click the image below to watch a demo of the engine in action:

[![Watch Demo](https://img.youtube.com/vi/GpyeFU3Kl4c/0.jpg)](https://www.youtube.com/watch?v=GpyeFU3Kl4c)
