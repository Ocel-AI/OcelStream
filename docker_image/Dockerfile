FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
# FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Optional: install additional packages if needed
# RUN apt-get update && apt-get install -y vim git
# RUN ls bin/bash /opt/nvidia/deepstream/deepstream-7.1/sources/
WORKDIR /
COPY deepstream_python_apps/ /deepstream_python_apps/
COPY DeepStream-Yolo/ /DeepStream-Yolo/
COPY ./run.sh  /
RUN  /bin/bash /run.sh
COPY ./patch_libnvinfer.sh /
RUN chmod +x /patch_libnvinfer.sh
RUN /patch_libnvinfer.sh
COPY ./compile_nvdsinfer_yolo.sh /
RUN chmod +x /compile_nvdsinfer_yolo.sh
COPY nvdsinfer_yolo/ /nvdsinfer_yolo/
RUN /compile_nvdsinfer_yolo.sh
WORKDIR /opt/nvidia/deepstream/deepstream-7.1/sources/
CMD [ "/bin/bash" ]