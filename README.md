# SIPAI

> "Simplicity is the ultimate sophistication" _- Leonardo da Vinci (attributed)_

A Simple Image Processing Artificial Intelligence, by [Dams](https://dams-labs.net/).

SIPAI is a neural network that builds upon my previous [SMLP](https://github.com/obewan/SMLP) project.
Its initial focus is on image enhancement, with plans for future releases to incorporate additional features such as image anomaly detection and more.

It uses a Multi-Layer Perceptron (MLP) architecture, with a modification that incorporates elements of a Recurrent Neural Network (RNN) to handle spatial information.

```
Project under construction...
```

---

Requirements for compiling the source code:

- A C++20 compiler (like the GNU C++ `g++-12` on Linux or Visual Studio and its compiler on Windows)
- The [OpenCV](https://opencv.org/get-started/) library (on Windows, check [how to update the system path for OpenCV](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path), you must also add an `OpenCV_BUILD` env variable that refer to the `opencv build` directory.)
- The [Intel TBB](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb) library
- [CMake](https://cmake.org/)

Also on Windows, be sure that the name of your images is encoded for your system (ASCII for example), or try to use Unicode UTF-8.

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
