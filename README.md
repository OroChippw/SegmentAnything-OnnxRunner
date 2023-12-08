# SegmentAnything-OnnxRunner
| [English](README.md) | [简体中文](README_CN.md) |
<div align="center">
  <div style="display: flex; justify-content: center;">
    <img src="assets/truck_click_774_366_box_lt636_292_rb_874_454.jpg" width="400"/>
    <img src="assets/truck_click_774_366_box_lt636_292_rb_874_454_mask.png" width="400" style="margin-right: 3px;"/>
    <img src="assets/truck_click_774_366_box_lt636_292_rb_874_454_visual.jpg" width="400" style="margin-right: 3px;"/>
  </div>
  <p align="center">
    <em>
      <strong>Model</strong> : sam_vit_l_0b3195_encoder.onnx + sam_vit_l_0b3195_decoder ; 
      <strong>Image</strong> : truck.jpg ; 
      <strong>Clickinfo</strong> : [(774,366) , positive] ; 
      <strong>Boxinfo</strong> : [(636,292),(874,454)]</em>
  </p> 
</div>

## Introduction 📰
​    SegmentAnything-OnnxRunner is an example using Meta AI Research's SAM onnx model in C++.The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.This repository is used to record the experiment data that run SAM onnx model on CPU.At the same time, the encoder and decoder of SAM are decoupled in this repository.

## Attention⚠️
* Currently, the interface only supports CPU execution.The specific experimental data and equipment used are shown below. And the code is only supported on Windows and may encounter issues when running on Linux.

## Development Enviroments🖥️
​    The description only represents the development environment of this repository and does not represent any software version restriction information.
* Device 1 : Windows 10 Professional / CUDA v11.3 / cmake version 3.26.2 / CPU i5-13600KF
* Device 2 : Windows 11 Home / CUDA v11.7 / cmake version 3.27.1 / CPU i5-13500H

## Quick Start💡
### Requirements
``` 
# onnxruntime 3rdparty
This repository use onnxruntime-win-x64-1.14.1
# opencv 3rdparty
This repository use opencv4.8.0
# CXX_STANDARD 17
```
### Build
```
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
$ mkdir build
# Enter the build folder and run CMake to configure the project
$ cd build
$ cmake ..
# Use the build system to compile/link this project
$ cmake --build .
# If the specified compilation mode is debug or release, it is as follows
$ cmake --build . --config Debug
$ cmkae --build . --config Release
```
### Get Model Checkpoints
​    All models are available in [Baidu Pan](https://pan.baidu.com/s/1j0z1mHDOshOCcQWwetmFnQ?pwd=ljgr) (code: ljgr).The SAM encoder and decoder are decoupled and quantized. After decoupling, if you perform multiple interactive clicks on a picture, you don't need to re-encode it. The model with -quantize is the quantized version

### Startup Parameters
| Parameters | Required | Description  |
|:------|:----:|:------:|
| --encoder_model_path | ✅ | The path to store the encoder model |
| --decoder_model_path | ✅ | The path to store the decoder model |
| --image_path | ✅ | The path of the image to be segmented |
| --save_dir | / | Path to output segmentation results. Default is '../output' . If the folder does not exist, it will be created.  |
| --use_demo | / | Whether to use the graphical interface for SAM segmentation. Default 'true' |
| --use_boxinfo | / | Whether to use frame selection information to assist SAM segmentation. Default 'false' |
| --use_singlemask | / | Whether to use the Singmask model for SAM segmentation, not recommended. Default 'false' |
| --keep_boxinfo | / | Whether to retain box selection information in multi-step operations. Default 'true' |
| --threshold | / | IOU segmentation threshold, results below the threshold will not be saved. Default 0.9 | 

​    An example is shown below:
```
# Run in the build directory
$ Debug/main.exe --encoder_model_path {your_encoder_path} --decoder_model_path {your_decoder_path} --image_path {your_image_path} --use_demo true --use_boxinfo true

$ Release/main.exe --encoder_model_path {your_encoder_path} --decoder_model_path {your_decoder_path} --image_path {your_image_path} --use_demo true --use_boxinfo true
```
### Operating Instructions
​    It is divided into demo mode and cmd mode according to your --use_demo option.The following are some operation instructions in demo mode. In cmd mode, just enter the coordinates and frame information directly in the console.
| Operation | Mode | Description |
|:------:|:----:|:------:|
| Mouse Left Button Down | use_demo | Click the left mouse button to capture the coordinates (x, y) of the point, and set positive to ‘true’. The visualization effect is a green point. |
| Mouse Right Button Down | use_demo | Click the right mouse button to capture the coordinates (x, y) of the point, and set positive to 'false'. The visualization effect is a red point. |
| Keyboard Shift Key + Mouse Left Button Down | use_demo && use_boxinfo | Press shift and left-click at the same time to drag and drop to get box information.box_info.The box info includes the upper left corner point and the lower right corner point. |
| Keyboard 'q' or Keyboard 'esc' | use_demo | Quit, Press 'q' or 'esc' to quit Segment Anything Onnx Runner Demo |
| Keyboard 'c' or  | use_demo | Continue, Press 'C' to use the mask output from the previous run as the decoder's mask_input to continue segmentation. <br />**Note**: When clicking directly without pressing the C key, mask_input is not enabled, which is equivalent to restarting the single-step operation. |
<div align="center">
  <div style="display: flex; justify-content: center;">
    <img src="assets/dog_click_600_218_box_lt466_118_rb_668_264.jpg" width="400"/>
    <img src="assets/dog_click_600_218_box_lt466_118_rb_668_264_mask.png" width="400" style="margin-right: 3px;"/>
    <img src="assets/dog_click_600_218_box_lt466_118_rb_668_264_visual.jpg" width="400" style="margin-right: 3px;"/>
  </div>
  <p align="center">
    <em>
      <strong>Model</strong> : sam_vit_l_0b3195_encoder.onnx + sam_vit_l_0b3195_decoder ; 
      <strong>Image</strong> : dog.jpg ; 
      <strong>Clickinfo</strong> : [(600,218) , positive] ; 
      <strong>Boxinfo</strong> : [(466,118),(668,264)]</em>
  </p> 
</div>

## Experiment Record🗒️
Environment Device 1 : i5-13600KF + NVIDIA GeForce RTX 3060（12GB）
Input image resolution : 1920 * 1080 * 3  
All models are available in [Baidu Pan](https://pan.baidu.com/s/1j0z1mHDOshOCcQWwetmFnQ?pwd=ljgr) (code: ljgr).    
#### Encoder
| Encoder version | Model Size(MB/GB) | CPU encoding speed(s) | 
| :------------------:| :---------------: | :---------------: | 
| sam_vit_b_01ec64_encoder.onnx          | 342MB | 2.5485 | 
| sam_vit_b_01ec64_encoder-quantize.onnx | 103MB | 2.0446 | 
| sam_vit_l_0b3195_encoder.onnx          | 1.14GB | 6.0346 | 
| sam_vit_l_0b3195_encoder-quantize.onnx | 316MB | 4.1599 | 
#### Decoder
| Decoder version | Model Size(MB) | CPU decoding speed(s) | 
| :------------------:| :---------------: | :---------------: | 
| sam_vit_b_01ec64_decoder.onnx            | 15.7MB | 0.075 | 
| sam_vit_b_01ec64_decoder_singlemask.onnx | 15.7MB | 0.075 | 
| sam_vit_b_01ec64_decoder.onnx            | 15.7MB | 0.086 | 
| sam_vit_b_01ec64_decoder_singlemask.onnx | 15.7MB | 0.082 | 


## License
This project is licensed under the MIT License.