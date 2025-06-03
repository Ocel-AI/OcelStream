#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/apps/')

import math
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
import numpy as np
from utils import transform_image_to_base64, resize_mask
import cv2
import pyds

MUXER_BATCH_TIMEOUT_USEC = 33000
MEM_TYPE = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720


def osd_sink_pad_buffer_probe(pad, info, u_data):
    output_object = {}
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        frame_image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        if frame_image is not None:
            frame_image = np.frombuffer(frame_image, dtype=np.uint8)
            #? reshape the frame image to the correct dimensions
            _frame_image = frame_image.reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH, 4))  # RGBA
            #? convert to correct color format for saving
            _frame_image = cv2.cvtColor(_frame_image, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR for OpenCV

        l_obj = frame_meta.obj_meta_list
        obj_count = 0
        detection_data = []
        segmentation_data = []
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                #* Detection metadata
                rect = obj_meta.rect_params
                detection_data.append({
                    "object_id": obj_meta.object_id,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": {
                        "left": rect.left,
                        "top": rect.top,
                        "width": rect.width,
                        "height": rect.height
                    }
                })
                #* Segment output
                rectparams = obj_meta.rect_params # Retrieve rectparams for re-sizing mask to correct dims
                maskparams = obj_meta.mask_params # Retrieve maskparams
                class_id = obj_meta.class_id
                if maskparams is not None:
                    mask_image = resize_mask(maskparams, math.floor(rectparams.width), math.floor(rectparams.height))
                segmentation_data.append({
                    "object_id": obj_meta.object_id,
                    "class_id": class_id,
                    "mask": mask_image,
                })

                #? SEND DATA TO WEBSOCKET SERVER
                output_object["source_id"] = frame_meta.source_id
                print(f"Source ID: {output_object['source_id']}")
                output_object["segmentation_data"] = segmentation_data
                output_object["detection_data"] = detection_data
                output_object["frame_number"] = frame_number
                output_object["frame_image"] = transform_image_to_base64(_frame_image) if frame_image is not None else None

                l_obj = l_obj.next
                obj_count += 1
            except StopIteration:
                break


        try:
            #* Save detection and segmentation data to output_object
            #* send output_object to websocket server
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def main(args):
    # Check input arguments
    if(len(args)<2):
        sys.stderr.write("usage: %s <h264_elementary_stream>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization

    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")



    sgie = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to make sgie \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    nvvidconv.set_property("nvbuf-memory-type", MEM_TYPE)
    

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvosd.set_property("display-mask", 1)

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    print("Playing file %s " %args[1])
    source.set_property('location', args[1])
    streammux.set_property('width', OUTPUT_WIDTH)
    streammux.set_property('height', OUTPUT_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("nvbuf-memory-type", MEM_TYPE)

    #Set properties of pgie and sgie
    pgie.set_property('config-file-path', "../config/config_infer_primary_yolo11.txt")
    sgie.set_property('config-file-path', "../config/config_pgie_yolo_seg.txt")



    #! update for dsaving images----------------------------------------
    # Create a capsfilter to enforce RGBA format
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter.set_property("caps", caps)
    
    #!----------------------------------------------------------------
    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)  #! Add the capsfilter to the pipeline
    pipeline.add(nvosd)
    pipeline.add(sink)




    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(sgie)
    sgie.link(nvvidconv)
    nvvidconv.link(capsfilter) # nvvidconv.link(nvosd)   #!
    capsfilter.link(nvosd)  #! Link capsfilter to nvosd
    nvosd.link(sink)


    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)


    print("Starting pipeline \n")
    
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
      loop.run()
    except:
      pass

    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

