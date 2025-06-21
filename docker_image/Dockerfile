FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
# FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Optional: install additional packages if needed
# RUN apt-get update && apt-get install -y vim git
# RUN ls bin/bash /opt/nvidia/deepstream/deepstream-7.1/sources/
WORKDIR /
COPY docker_image/deepstream_python_apps/ /deepstream_python_apps/
COPY docker_image/DeepStream-Yolo/ /DeepStream-Yolo/
COPY docker_image/run.sh  /
RUN  /bin/bash /run.sh
COPY docker_image/patch_libnvinfer.sh /
RUN chmod +x /patch_libnvinfer.sh
RUN /patch_libnvinfer.sh
COPY docker_image/compile_nvdsinfer_yolo.sh /
RUN chmod +x /compile_nvdsinfer_yolo.sh
COPY docker_image/nvdsinfer_yolo/ /nvdsinfer_yolo/
RUN /compile_nvdsinfer_yolo.sh
COPY backend/requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt
WORKDIR /deepstream_app/backend
CMD ["uvicorn", "backend.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["ls"]