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
import platform
import configparser
import math
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
import numpy as np

import pyds

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000



def clip(val, low, high):
    if val < low:
        return low 
    elif val > high:
        return high 
    else:
        return val


# Resize and binarize mask array for interpretable segmentation mask
def resize_mask(maskparams, target_width, target_height):
    src = maskparams.get_mask_array() # Retrieve mask array
    if src.size == 0:
        # print("Mask array is None, returning empty array")
        return np.empty((target_height, target_width), dtype=np.uint8)

    dst = np.empty((target_height, target_width), src.dtype) # Initialize array to store re-sized mask
    original_width = maskparams.width
    original_height = maskparams.height
    ratio_h = float(original_height) / float(target_height)
    ratio_w = float(original_width) / float(target_width)
    threshold = maskparams.threshold
    channel = 1

    # Resize from original width/height to target width/height 
    for y in range(target_height):
        for x in range(target_width):
            x0 = float(x) * ratio_w
            y0 = float(y) * ratio_h
            left = int(clip(math.floor(x0), 0.0, float(original_width - 1.0)))
            top = int(clip(math.floor(y0), 0.0, float(original_height - 1.0)))
            right = int(clip(math.ceil(x0), 0.0, float(original_width - 1.0)))
            bottom = int(clip(math.ceil(y0), 0.0, float(original_height - 1.0)))

            for c in range(channel):
                # H, W, C ordering
                # Note: lerp is shorthand for linear interpolation
                left_top_val = float(src[top * (original_width * channel) + left * (channel) + c])
                right_top_val = float(src[top * (original_width * channel) + right * (channel) + c])
                left_bottom_val = float(src[bottom * (original_width * channel) + left * (channel) + c])
                right_bottom_val = float(src[bottom * (original_width * channel) + right * (channel) + c])
                top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left)
                bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left)
                lerp = top_lerp + (bottom_lerp - top_lerp) * (y0 - top)
                if (lerp < threshold): # Binarize according to threshold
                    dst[y,x] = 0
                else:
                    dst[y,x] = 255
    return dst

def osd_sink_pad_buffer_probe(pad, info, u_data):
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
        print(f"\n--- Frame {frame_number} ---")

        # 🔹 Print detection output from object metadata
        l_obj = frame_meta.obj_meta_list
        obj_count = 0
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                rect = obj_meta.rect_params
                # print(f"[Detection] Object {obj_count}: Class ID = {class_id}, Confidence = {confidence:.2f}, BBox = ({rect.left}, {rect.top}, {rect.width}, {rect.height})")
                
                #! Segment output
                rectparams = obj_meta.rect_params # Retrieve rectparams for re-sizing mask to correct dims
                maskparams = obj_meta.mask_params # Retrieve maskparams
                class_id = obj_meta.class_id
                if maskparams is not None:
                    mask_image = resize_mask(maskparams, math.floor(rectparams.width), math.floor(rectparams.height))
                    # print(f"[Segmentation] Object {obj_count}: Class ID = {class_id}, Mask Shape = {mask_image.shape}, Mask Type = {mask_image.dtype}")

                l_obj = l_obj.next
                obj_count += 1
            except StopIteration:
                break

        # 🔹 Print segmentation metadata (primary)
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_SEGMENTATION_META:
                    segmeta = pyds.NvDsInferSegmentationMeta.cast(user_meta.user_meta_data)
                    print(f"[Segmentation] Found - Width: {segmeta.width}, Height: {segmeta.height}, Classes: {segmeta.num_classes}")

                    # Extract mask array
                    try:
                        mask_array = pyds.get_segmentation_masks(segmeta)
                        print(f"[Segmentation] Mask shape: {mask_array.shape}")

                        # Save first class channel mask
                        mask_img = (mask_array[0] * 255).astype(np.uint8)
                        cv2.imwrite(f"segment_frame_{frame_number}.png", mask_img)
                    except Exception as e:
                        print("Failed to extract segmentation mask:", e)
                l_user = l_user.next
            except StopIteration:
                break

        # 🔹 Update performance metrics if needed
        stream_index = f"stream{frame_meta.pad_index}"
        # global perf_data
        # perf_data.update_fps(stream_index)

        try:
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

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # if not tracker:
    #     sys.stderr.write(" Unable to create tracker \n")



    sgie = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to make sgie \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

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
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    #Set properties of pgie and sgie
    pgie.set_property('config-file-path', "../config/config_infer_primary_yolo11.txt")
    sgie.set_property('config-file-path', "../config/config_pgie_yolo_seg.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(nvvidconv)
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
    nvvidconv.link(nvosd)
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

