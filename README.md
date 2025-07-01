# DeepVisionStream

**DeepVisionStream** is a modular DeepStream-based computer vision platform for real-time video analytics. It supports custom models like YOLO, SAM, and D-Fine, with C++ parsers and Python bindings. Inference results (bounding boxes, masks, metadata) are streamed to external apps using WebSocket.

---

## 🚀 Features

- 🎥 Real-time inference with DeepStream and TensorRT
- 🧩 Plugin support for YOLO, SAM, D-Fine, and more
- 🐍 Python bindings for accessing frames and metadata
- 🌐 WebSocket server to broadcast metadata to clients
- 🐳 Docker Compose setup for simplified deployment

---

## ⚡ Quick Start

### Model Support

Currently, **DeepVisionStream** supports **YOLO ** models ** 8, 9, 10, 11, 12 **. You can use both detection and segmentation models.

#### Converting YOLO Models

1. **For YOLO Detection:**
   - Use the script at `tools/export_yolo11.py` to convert your YOLO 11 detection model
   - This will generate a `.onnx` model file

2. **For YOLO Segmentation:**
   - Use Ultralytics (located in `tools/ultralytics`) to export your YOLO 11 segmentation model
   - Follow the Ultralytics README for proper export commands
   - This will also generate a `.onnx` model file

#### File Placement

After conversion, place your files as follows:

- **`.onnx` model files** → `deepstream/models/`
- **`labels.txt` files** → `deepstream/config/`

#### Configuration Updates

Update the configuration files to point to your models:

- **For YOLO Detection:** Update `deepstream/config/config_infer_primary_yolo11.txt`
- **For YOLO Segmentation:** Update `deepstream/config/config_pgie_yolo_seg.txt`

Make sure to adapt the paths in these config files to match your model and label file locations.

### Prerequisites

- NVIDIA GPU with CUDA support
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- CUDA runtime (included in Docker image)

### Run with Docker Compose

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DeepVisionStream.git
   cd DeepVisionStream
   ```
2. **Build and start the services:**
   ```bash
   docker-compose up --build
   ```
3. **Access the WebSocket server** at `ws://localhost:<port>` (see your configuration).

### API Usage

The project includes a built-in API to interact with the DeepStream pipeline:

- **Add sources** (files or RTSP links) dynamically
- **Delete sources** from the pipeline
- **Real-time inference results** via WebSocket

You can add video sources as files (`file:///deepstream_app/static/video.mp4`) or RTSP streams (`rtsp://camera-ip:port/stream`) and remove them as needed during runtime.

**Note:** When adding file sources, make sure to place your video files in the `static` folder first. By default, video files are played in a loop.

---

##  Project Structure

```bash
DeepVisionStream/
                ├── backend
                │   ├── app
                │   ├── requirements.txt
                ├── deepstream
                │   ├── app
                │   ├── config
                │   └── models
                ├── docker-compose.yml
                ├── docker_image
                │   ├── compile_nvdsinfer_yolo.sh
                │   ├── deepstream_python_apps
                │   ├── DeepStream-Yolo
                │   ├── Dockerfile
                │   ├── nvdsinfer_yolo
                │   ├── patch_libnvinfer.sh
                │   └── run.sh
                ├── docs
                ├── LICENSE
                ├── README.md
                └── tools
                    ├── export_yolo11.py
                    └── ultralytics

```

