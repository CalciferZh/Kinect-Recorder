# Kinect Recorder

## Overview

This repo is an easy-to-use off-the-shell tool to record color + depth + body index stream from Microsoft Kinect 2.

## Usage

### Record

Run `python recorder.py`, and you will be required to enter several options. After that, you will see a window displaying color, depth and body index stream from Kinect. Press any key to start recording. During recording, the display window will be frozen (not updated) to ensure the speed of recording. To stop recording, just close the window. It will take some time to save the data.

The recorder takes around 14GB RAM per minute. You need to monitor your RAM usage and stop recording before your RAM being used up.

Class `KinectRecorder` is used for recording. Due to Python's low performance (and my bad coding skill), the recorder can:

* Write color stream to video file at 10 ~ 30 fps while recording.
* Cache all color frames and write them to video together after recording at 30 fps. It requires around 14GB RAM per minute.

If you want a stable frame rate, don't save-on-record; on the other hand, if you want to record for a long time (longer than your RAM can bear), please turn on save-on-record.

### Depth-color Alignment

See `alignment.py`.

Method `align` is used to align color stream to depth stream, which means assigning color to every pixel in the depth image. Camera intrinsic parameters are required. We provide a default intrinsic parameters setting, but for depth camera they may vary greatly across sensors, while for color cameras they are always the same. The intrinsic parameters of depth camera can be read from C++ SDK of Kinect, or calibrated manually.

### Visualization

To visualize the results, please refer to `visualizer.py`. `RawVisualizer` is used to visualize raw captured data, and `AlignedVisualizer` is used to visualize the result after alignment.

## Dependencies

* numpy
* opencv-python
* tqdm
* pykinect2
* pygame

All dependencies are available via `pip install`.
