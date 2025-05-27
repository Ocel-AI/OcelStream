#/bin/sh

cd /opt/nvidia/deepstream/deepstream-7.1/sources/

git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

git clone https://github.com/marcoslucianops/DeepStream-Yolo.git


apt update -y
apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

cd deepstream_python_apps

pip3 install build

cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/
git submodule update --init
python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

cd bindings/3rdparty/gstreamer/subprojects/gst-python/
meson setup build
cd build
ninja
ninja install


cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/bindings
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
python3 -m build


cd dist/
pip3 install ./pyds-1.2.0-*.whl


pip3 install cuda-python
pip3 install opencv-python


pip install ultralytics
pip3 install torch

pip3 install onnx onnxslim onnxruntime


cd /opt/nvidia/deepstream/deepstream-7.1/sources/DeepStream-Yolo/
export CUDA_VER=12.6
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo