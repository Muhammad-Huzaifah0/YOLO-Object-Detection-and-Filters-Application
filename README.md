# YOLO Object Detection and Filters Application

## Overview

This application provides a graphical user interface (GUI) for YOLO-based object detection and image processing filters. The GUI is built using Tkinter, and the core functionalities include:
- Loading YOLO model weights, configuration, and class names.
- Selecting and displaying images and videos.
- Live video streaming with YOLO object detection.
- Applying various image processing filters.

## Features

- **YOLO Object Detection**: Load YOLO model and perform object detection on images, videos, and live video streams.
- **Image Filters**: Apply various filters such as edge detection, sharpening, Gaussian blur, brightness adjustment, erosion, dilation, sepia tone, contrast adjustment, negative, and emboss to images and videos.
- **Live Video**: Capture live video from a webcam and apply YOLO detection and filters in real-time.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pillow (PIL Fork)
- Tkinter (comes with Python standard library)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yolo-object-detection-filters.git
    cd yolo-object-detection-filters
    ```

2. Install the required Python packages:
    ```bash
    pip install opencv-python-headless numpy pillow
    ```

## Usage

1. **Load YOLO Model**:
    - Click on the "Select Weights" button to load the YOLO weights file (`.weights`).
    - Click on the "Select CFG" button to load the YOLO configuration file (`.cfg`).
    - Click on the "Select Names" button to load the class names file (`.names`).

2. **Load Image/Video**:
    - Click on the "Select Image" button to load an image file.
    - Click on the "Select Video" button to load a video file.

3. **Live Video**:
    - Click on the "Live Video" button to start capturing live video from the webcam.

4. **Apply Filters**:
    - Use the filter buttons on the right to apply different filters to the loaded image or video frames.

5. **Toggle Object Detection**:
    - Click on the "Detect Objects" button to enable or disable YOLO object detection on the loaded image or video frames.

## Code Structure

- `YOLOFaceDetectionApp`: Main application class.
- `create_control_buttons`: Creates and configures the control buttons.
- `select_weights`, `select_cfg`, `select_names`: Functions to select and load YOLO model files.
- `load_yolo`: Loads the YOLO model.
- `select_image`, `select_video`, `toggle_live_video`: Functions to handle image and video selection and live video streaming.
- `load_image`: Loads and displays the selected image.
- `toggle_detect_objects`: Toggles YOLO object detection.
- `apply_yolo`: Applies YOLO object detection to a frame.
- `apply_filter_to_frame`: Applies the selected filter to a frame.
- `resize_video_frame`: Resizes video frames to fit the main frame.
- `show_frame_on_canvas`: Displays a frame on the canvas.
- `stop_running`: Stops live video streaming and resets flags.

## Example

Here is an example of how to use the application:

```python
import tkinter as tk
from YOLOFaceDetectionApp import YOLOFaceDetectionApp

root = tk.Tk()
app = YOLOFaceDetectionApp(root)
root.mainloop()
