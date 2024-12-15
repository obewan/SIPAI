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
- The [OpenCV](https://opencv.org/get-started/) library, v4.10.0 recommended.   
If using OpenCV with Vulkan: enable OpenGL support, edit the OpenCV CMakelists.txt:    
*OCV_OPTION(WITH_OPENGL "Include OpenGL support" ON     
  VISIBLE_IF TRUE    
  VERIFY HAVE_OPENGL)*   
  and rebuild its libs.
- The [Intel TBB](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb) library (for OpenCV)
- The [Vulkan SDK](https://www.vulkan.org/) library (on Debian: `sudo apt-get -y install libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-tools`, on Windows: https://vulkan.lunarg.com/sdk/home#windows).
- The GLSL tools to use with Vulkan (on Debian: `sudo apt-get -y install glslang-tools spirv-tools`)
- [CMake](https://cmake.org/)
- The [Qt6](https://www.qt.io/download-qt-installer-oss) libraries for the GUI version (on Debian: `sudo apt-get -y install qt6-base-dev libqt6svg6 qt6-svg-dev`).
- on Windows:
  - **OpenCV**: check [how to update the system path for OpenCV](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path). Also you may build OpenCV from its sources, if it doesn't match your config, then after build the Release and the Debug libs, update your system path (*%OPENCV_DIR%\bin\Release* *%OPENCV_DIR%\bin\Debug*).
  - **Image Encoding**: ensure that your image names are encoded in a format compatible with your system (for example, ASCII). Alternatively, you can install Unicode UTF-8 on your system for broader compatibilit .
  - **Vulkan SDK Installation**: after installing the Vulkan SDK, make sure to log out and log back in. This step is necessary to update the system paths.
  - **Qt6 Installation**: ensure to have set the environment path to the `msvc2022_64\bin` or `mingw_64\bin` folder (be sure to select the right compilers in the installer) of your installation and re-login.
  - **VSCode - Intellisense**: in the VSCode `C/C++ Configurations`, in `Include path` settings, add include paths for your VulkanSDK and opencv Includes, for example: 
  ```
  ${workspaceFolder}/**
  C:/Libs/VulkanSDK/1.3.296.0/Include/**  
  C:/Libs/opencv-4.10.0/include/**
  C:/Libs/opencv-4.10.0/build/**
  C:/Libs/opencv-4.10.0/modules/**
  ```
  - **Environments**: also be sure to have the same environments for all the libs used (x64, Debug, msvc...)
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
