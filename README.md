# DeepVisionStream

**DeepVisionStream** is a modular DeepStream-based computer vision platform for real-time video analytics. It supports custom models like YOLO, SAM, and D-Fine, with C++ parsers and Python bindings. Inference results (bounding boxes, masks, metadata) are streamed to external apps using RabbitMQ.


https://github.com/user-attachments/assets/4a86d335-045c-4770-bf3d-142e7b91c691



---

## 🚀 Features

- 🎥 Real-time inference with DeepStream and TensorRT
- 🧩 Plugin support for YOLO, SAM, D-Fine, and more
- 🐍 Python bindings for accessing frames and metadata
- 🌐 RabbitMQ server to broadcast metadata to clients
- 🐳 Docker Compose setup for simplified deployment

---

## ⚡ Quick Start

### Model Support

Currently, **DeepVisionStream** supports **YOLO ** models ** 8, 9, 10, 11, 12 **. You can use both detection and segmentation models.

#### Converting YOLO Models

1. **For YOLO Detection:**
   - Use the script at `tools/export_yolo11.py` to convert your YOLO  detection model
   - This will generate a `.onnx` model file

2. **For YOLO Segmentation:**
   - Use Ultralytics (located in `tools/ultralytics`) to export your YOLO segmentation model
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
3. **Connect to the RabbitMQ server** to receive inference metadata. The server details (host, port, credentials) are specified in your configuration files.
When you add a new source using the API, the response includes a unique UUID. This UUID is used as the queue name for receiving inference metadata from RabbitMQ.

### API Usage

The project includes a built-in API to interact with the DeepStream pipeline:

- **Add sources** (files or RTSP links) dynamically
- **Delete sources** from the pipeline
- **Real-time inference results** are now delivered via RabbitMQ

You can add video sources as files (`file:///deepstream_app/static/video.mp4`) or RTSP streams (`rtsp://camera-ip:port/stream`) and remove them as needed during runtime.

**Note:** When adding file sources, make sure to place your video files in the `static` folder first. By default, video files are played in a loop.

---

## Benchmarks (Latency + 20-Stream Test)

Test setup
GPU: NVIDIA RTX 4000 Ada (laptop)

SDK: NVIDIA DeepStream + TensorRT (FP32)

Models: YOLO-Seg family (segmentation head enabled)

Pipeline: Dynamic RTSP/file sources → real-time inference → per-source RTSP out with overlays

Backend: FastAPI + Python/C++ parsers, RabbitMQ for metadata


Results
Throughput: 20 streams @ 30 FPS per stream

GPU load: ~70% utilization

VRAM: ~1.3 GB used for the pipeline during the 20-stream run

Latency method: visual side-by-side of original vs. overlayed output to benchmark end-to-end delay (camera → decode → infer → encode → RTSP).

Note: Exact milliseconds vary with camera/codec/latency settings; the video makes the delta visible in real time.



https://github.com/user-attachments/assets/4146d65b-9104-4c21-8f3b-ca41a4baf3bd



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

