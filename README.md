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

## 🧰 Project Structure

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

