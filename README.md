# SIPAI

> "Simplicity is the ultimate sophistication" _- Leonardo da Vinci (attributed)_

A Simple Image Processing Artificial Intelligence, by [Dams](https://dams-labs.net/).

SIPAI is a neural network that builds upon my previous [SMLP](https://github.com/obewan/SMLP) project.
Its initial focus is on image enhancement, with plans for future releases to incorporate additional features such as image anomaly detection and more.

It uses a Multi-Layer Perceptron (MLP) architecture, with a modification that incorporates elements of a Recurrent Neural Network (RNN) to handle spatial information.

```
WARNING : project still under early development...
WARNING : Vulkan feature is very experimental, and is currently limited to one hidden layer (in addition to the input and the output layers)
```

---

Requirements for compiling the source code:

- A C++20 compiler (like the GNU C++ `g++-12` on Linux or Visual Studio and its compiler on Windows)
- The [OpenCV](https://opencv.org/get-started/) library (on Windows, check [how to update the system path for OpenCV](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path), you must also add an `OpenCV_BUILD` env variable that refer to the `opencv build` directory.
- The [Intel TBB](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb) library
- The [Vulkan](https://www.vulkan.org/) library (on Debian: `sudo apt-get -y install libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-tools`, on Windows: https://vulkan.lunarg.com/sdk/home#windows).
- The GLSL tools to use with Vulkan (on Debian: `sudo apt-get -y install glslang-tools spirv-tools`)
- [CMake](https://cmake.org/)
- on Windows:
  - **Image Encoding**: Ensure that your image names are encoded in a format compatible with your system (for example, ASCII). Alternatively, you can install Unicode UTF-8 on your system for broader compatibility .
  - **Vulkan SDK Installation**: After installing the Vulkan SDK, make sure to log out and log back in. This step is necessary to update the system paths.
- on a Windows WSL Linux, be sure to use WSL 2 or better and to have enabled the GPU acceleration: after a `sudo apt-get install mesa-utils` the command `glxinfo | grep -i opengl` should show a line like `OpenGL renderer string: D3D12 (the 3D card)`. You should have a `/dev/dxg` device link as well.

---

### Changelog

[Click here to see the changelog.](./CHANGELOG.md)

&nbsp;

---

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

&nbsp;
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![API](https://img.shields.io/badge/API-Documentation-blue)](https://obewan.github.io/SIPAI/api/)
[![Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fobewan.github.io%2FSIPAI%2Fcoverage%2Fcoverage.json&query=coverage&label=coverage&color=green)](https://obewan.github.io/SIPAI/coverage/html/)

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
