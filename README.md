# Kinect Recorder

## Overview

This is a work-in-progress repo.

This repo is an easy-to-use tool to record color + depth + body index stream from Microsoft Kinect 2.

## Usage

### Record

Run `python recorder.py` to run demo.

Class `KinectRecorder` is used for recording raw stream. Due to Python's low performance (and my bad coding skill), the recorder can:

* Display color, depth and body index stream at 4 fps.
* Write color stream to video file at 10 ~ 30 fps while recording.
* Write all color frames to video together after recording at 30 fps. However, to store the color frames. It requires around 14GB RAM per minute.

I recommend first setting everything up with `visualize=True`, then set `visualize=False` to begin real recording. If you have plenty of RAM, set `save_on_record=False` to obtain stable 30 fps video, otherwise set it to `False`.

### Depth-color Alignment

Run `python alignment.py` to run demo.

Method `align` is used to align color stream to depth stream, which means assigning color to every pixel in the depth image. Camera intrinsics are required. We provide a default intrinsics setting, but intrinsics of depth camera may vary greatly across sensors, while color cameras' intrinsics are always the same. The intrinsics of depth camera can be read from C++ SDK of Kinect, or calibrated manually.

### Visualization

To visualize the results, please refer to `visualizer.py`. `RawVisualizer` is used to visualize raw captured data, and `AlignedVisualizer` is used to visualize result after alignment.

## Dependencies

* numpy
* opencv-python
* tqdm
* pykinect2
* pygame

All dependencies are available via `pip install`.
