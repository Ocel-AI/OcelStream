/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author: Levi Pereira 
 * https://forums.developer.nvidia.com/u/levi_pereira/activity
 * https://github.com/levipereira
 */



#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferYoloNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferYoloMask(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList);
                                   

extern "C"
bool NvDsInferYoloNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if (outputLayersInfo.size() != 4 && outputLayersInfo.size() != 5)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 or 5 output buffers, detected in the network: "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.layerName && name == layer.layerName) {
                return &layer;
            }
        }
        return nullptr;
    };
    
    const NvDsInferLayerInfo *num_detsLayer = layerFinder("num_dets");
    const NvDsInferLayerInfo *boxesLayer = layerFinder("det_boxes");
    const NvDsInferLayerInfo *scoresLayer = layerFinder("det_scores");
    const NvDsInferLayerInfo *classesLayer = layerFinder("det_classes");
    const NvDsInferLayerInfo *indicesLayer = nullptr;  

    if (outputLayersInfo.size() == 5) 
    {
        indicesLayer = layerFinder("det_indices");
        
    }

    bool hasIndicesLayer = (indicesLayer != nullptr);

    if (!num_detsLayer || !boxesLayer || !scoresLayer || !classesLayer || (hasIndicesLayer  && !indicesLayer) ) {
        if (!num_detsLayer) {
            std::cerr << "  - num_detsLayer: Missing or unsupported data type." << std::endl;
        }

        if (!boxesLayer) {
            std::cerr << "  - boxesLayer: Missing or unsupported data type." << std::endl;
        }

        if (!scoresLayer) {
            std::cerr << "  - scoresLayer: Missing or unsupported data type." << std::endl;
        }

        if (!classesLayer) {
            std::cerr << "  - classesLayer: Missing or unsupported data type." << std::endl;
        }

        if (!indicesLayer) {
            std::cerr << "  - indicesLayer: Missing or unsupported data type." << std::endl;
        }
        return false;
    }

    if(num_detsLayer->inferDims.numDims != 1U) {
        std::cerr << "Network num_dets dims is : " <<
            num_detsLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(boxesLayer->inferDims.numDims != 2U) {
        std::cerr << "Network det_boxes dims is : " <<
            boxesLayer->inferDims.numDims << " expect is 2"<< std::endl;
        return false;
    }
    if(scoresLayer->inferDims.numDims != 1U) {
        std::cerr << "Network det_scores dims is : " <<
            scoresLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(classesLayer->inferDims.numDims != 1U) {
        std::cerr << "Network det_classes dims is : " <<
            classesLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if (hasIndicesLayer && indicesLayer->inferDims.numDims != 1U) {
        std::cerr << "Network det_indices dims is : " <<
            indicesLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }

    const char* log_enable = std::getenv("ENABLE_DEBUG");

    int* p_keep_count = (int *) num_detsLayer->buffer;
    float* p_bboxes = (float *) boxesLayer->buffer;
    int* p_indices = nullptr; 

    if (hasIndicesLayer){
        p_indices = (int *)indicesLayer->buffer;
    }
    NvDsInferDims inferDims_p_bboxes = boxesLayer->inferDims;
    int numElements_p_bboxes=inferDims_p_bboxes.numElements;

    float* p_scores = (float *) scoresLayer->buffer;
    unsigned int* p_classes = (unsigned int *) classesLayer->buffer;
    const float threshold = detectionParams.perClassThreshold[0];

    float max_bbox=0;
    for (int i=0; i < numElements_p_bboxes; i++)
    {
        if ( max_bbox < p_bboxes[i] )
            max_bbox=p_bboxes[i];
    }

    if (p_keep_count[0] > 0)
    {
        assert (!(max_bbox < 2.0));
        for (int i = 0; i < p_keep_count[0]; i++) {

            if ( p_scores[i] < threshold) continue;
            if ((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) {
                printf("Error: The number of classes configured in the GIE config-file (postprocess > num_detected_classes) is incorrect.\n");
                printf("Detected class index: %u\n", (unsigned int) p_classes[i]);
            }
            assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

            NvDsInferObjectDetectionInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];

            object.left=p_bboxes[4*i];
            object.top=p_bboxes[4*i+1];
            object.width=(p_bboxes[4*i+2] - object.left);
            object.height= (p_bboxes[4*i+3] - object.top);

            if(log_enable != NULL && std::stoi(log_enable)) {
                std::cout << "idx/label/conf/ x/y w/h -- ";
                if (hasIndicesLayer) { 
                    std::cout << p_indices[i] << " ";
                }
                else 
                {
                    std::cout << "None" << " ";
                }
                std::cout << p_classes[i] << " "
                    << p_scores[i] << " "
                    << object.left << " " << object.top << " " << object.width << " " << object.height
                    << std::endl;
            }

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
        }
    }
    return true;
}


extern "C" bool NvDsInferYoloMask(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    if (outputLayersInfo.size() != 5) {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 5 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
    
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.layerName && name == layer.layerName) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *num_detsLayer = layerFinder("num_dets");
    const NvDsInferLayerInfo *boxesLayer = layerFinder("det_boxes");
    const NvDsInferLayerInfo *scoresLayer = layerFinder("det_scores");
    const NvDsInferLayerInfo *classesLayer = layerFinder("det_classes");
    const NvDsInferLayerInfo *masksLayer = layerFinder("det_masks");

    if (!num_detsLayer || !boxesLayer || !scoresLayer || !classesLayer || !masksLayer) {
        if (!num_detsLayer) {
            std::cerr << "  - num_detsLayer: Missing or unsupported data type." << std::endl;
        }

        if (!boxesLayer) {
            std::cerr << "  - boxesLayer: Missing or unsupported data type." << std::endl;
        }

        if (!scoresLayer) {
            std::cerr << "  - scoresLayer: Missing or unsupported data type." << std::endl;
        }

        if (!classesLayer) {
            std::cerr << "  - classesLayer: Missing or unsupported data type." << std::endl;
        }

        if (!masksLayer) {
            std::cerr << "  - masksLayer: Missing or unsupported data type." << std::endl;
        }
        return false;
    }

    if(num_detsLayer->inferDims.numDims != 1U) {
        std::cerr << "Network num_dets dims is : " <<
            num_detsLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(boxesLayer->inferDims.numDims != 2U) {
        std::cerr << "Network det_boxes dims is : " <<
            boxesLayer->inferDims.numDims << " expect is 2"<< std::endl;
        return false;
    }
    if(scoresLayer->inferDims.numDims != 1U) {
        std::cerr << "Network det_scores dims is : " <<
            scoresLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(classesLayer->inferDims.numDims != 1U) {
        std::cerr << "Network det_classes dims is : " <<
            classesLayer->inferDims.numDims << " expect is 1"<< std::endl;
        return false;
    }
    if(masksLayer->inferDims.numDims != 2U) {
        std::cerr << "Network det_masks dims is : " <<
            masksLayer->inferDims.numDims << " expect is 2"<< std::endl;
        return false;
    }

    const char* log_enable = std::getenv("ENABLE_DEBUG");


    int* p_keep_count = (int *) num_detsLayer->buffer;
    float* p_bboxes = (float *) boxesLayer->buffer;
    float* p_scores = (float *) scoresLayer->buffer;
    unsigned int* p_classes = (unsigned int *) classesLayer->buffer;
    float *p_mask = (float *)  masksLayer->buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    NvDsInferDims inferDims_p_bboxes = boxesLayer->inferDims;
    int numElements_p_bboxes=inferDims_p_bboxes.numElements;
    
    const int mask_resolution = sqrt(masksLayer->inferDims.d[1]);

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout << "keep cout: " << p_keep_count[0] << std::endl;
    }

    float max_bbox=0;
    for (int i=0; i < numElements_p_bboxes; i++){
        if ( max_bbox < p_bboxes[i] )
            max_bbox=p_bboxes[i];
    }

    if (p_keep_count[0] > 0){
        assert (!(max_bbox < 2.0));

       for (int i = 0; i < p_keep_count[0]; i++) {
           
            if ( p_scores[i] < threshold) continue;

            if ((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) {
                printf("Error: The number of classes configured in the GIE config-file (postprocess > num_detected_classes) is incorrect.\n");
                printf("Detected class index: %u\n", (unsigned int) p_classes[i]);
                continue;
            }
            //assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);
            
            NvDsInferInstanceMaskInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];

            object.left=p_bboxes[4*i];
            object.top=p_bboxes[4*i+1];
            object.width=(p_bboxes[4*i+2] - object.left);
            object.height= (p_bboxes[4*i+3] - object.top);

            if (log_enable != NULL && std::stoi(log_enable)) {
                std::cout << "label/conf/ x/y w/h -- "
                << p_classes[i] << " "
                << p_scores[i] << " "
                << object.left << " " << object.top << " " << object.width << " "<< object.height << " "
                << std::endl;
            }

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);


            object.mask_size = sizeof(float) * mask_resolution * mask_resolution;
            object.mask = new float[mask_resolution * mask_resolution];
            object.mask_width = mask_resolution;
            object.mask_height = mask_resolution;

            const float* rawMask = reinterpret_cast<const float*>(p_mask + i * mask_resolution * mask_resolution);

            //float *rawMask = reinterpret_cast<float *>(p_mask + mask_resolution * mask_resolution * i);
            
            memcpy(object.mask, rawMask, sizeof(float) * mask_resolution * mask_resolution);

            objectList.push_back(object);
       }
    }
    return true;
}


CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferYoloNMS);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferYoloMask);