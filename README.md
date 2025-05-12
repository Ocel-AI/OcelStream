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
├── docker/                 # Dockerfile and compose files
│   ├── Dockerfile
│   └── docker-compose.yml
├── parsers/                # Custom C++ parsers for models
├── python_binding/         # Python API to access DeepStream output
├── socket_server/          # WebSocket server (Python)
├── configs/                # DeepStream pipeline configs
└── README.md

