# SAMOnnxRunner
## Introduction

## Development Enviroments
>  - Windows 10 Professional 
>  - CUDA v11.3
>  - cmake version 3.26.2

## Quick Start

### Requirements
``` 
# onnxruntime 3rdparty
This repository use onnxruntime-win-x64-1.14.1
# opencv 3rdparty
This repository use opencv4.7.0
# CXX_STANDARD 17
```
### How to build and run
```
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# Enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```

### License