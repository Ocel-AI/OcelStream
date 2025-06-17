# NvDsInferYolo  parse Function -  DeepStream 

This repository provides a custom implementation of parsing function to DeepStream for YOLO models exported using custom Plugins EfficientNMSX_TRT and ROIAling_TRT for Instance Segmentation Models and EfficientNMS_TRT for Detection Models.

`NvDsInferYoloMask` - For Segmentation Models <br>
`NvDsInferYoloNMS`  - For Detection Models

# Deployment Guide for NvDsInferYolo

## Cloning Repository and Installation

To clone the repository and build\install the `libnvds_infer_yolo_nms.so` library, follow these steps:


```bash
git clone https://github.com/levipereira/nvdsinfer_yolo_nms.git

cp -R nvdsinfer_yolo_nms/ /opt/nvidia/deepstream/deepstream/sources/libs/

cd /opt/nvidia/deepstream/deepstream/sources/libs/

CUDA_VER=12.2 make -C nvdsinfer_yolo_nms install
```
Install Location:
`/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolo.so`

Usage on Deepstream

Snippet [Gst-nvinferserver](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html)  Configuration File for YOLO Models
```
  postprocess {
    labelfile_path: "labels.txt"
    detection {
      num_detected_classes: 80
      custom_parse_bbox_func: "NvDsInferYoloNMS"
    }
  }
  custom_lib {
    path : "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolo.so"
  }
```

