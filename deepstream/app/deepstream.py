import threading
import gi
gi.require_version('Gst', '1.0') 
gi.require_version('GstRtspServer', '1.0')
import time
import sys
sys.path.append('/deepstream_python_apps/apps/common')
sys.path.append('/deepstream_app/deepstream/messaging')

from gi.repository import Gst, GstRtspServer, GLib
from bus_call import bus_call
from FPS import PERF_DATA
import pyds
from spotmanager import SpotManager
from source_bin_factory import SourceBinFactory
from rabbitmq import RabbitMQManager
from utils import transform_image_to_base64
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from utils import resize_mask, transform_image_to_base64, encode_mask_to_base64
import math
import ctypes
import queue
import re
import base64
import os
import re


class DynamicRTSPPipeline:
    """DeepStream pipeline that supports runtime add & remove of sources.

    Each source gets its own RTSP mount (rtsp://<host>:8554/ds-test<id>). The class
    now exposes remove_source() in addition to add_source().
    """

    def __init__(self, max_sources: int = 5, notification_callback=None):
        Gst.init(None)
        # --- Pipeline‑wide parameters ---
        self.max_sources = int(os.getenv("MAX_RESOURCES", 15))
        self.codec = "H264"
        self.bitrate = 4_000_000  # 4 Mbps for H264, adjust as needed
        # --- Environment variables for configuration ---
        self.hide_class_ids = os.getenv("DEEPSTREAM_IGNORE_CLASS_IDS_SEGMENTATION", "") # List of class IDs to hide in rtsp stream segmentation
        self.hide_class_ids = [int(x.strip()) for x in self.hide_class_ids.split(",") if x.strip().isdigit()]
        self.disable_box_segmentation = os.getenv("DEEPSTREAM_DISABLE_BOX_SEGMENTATION", "0").lower() == "1" # Disable bounding box segmentation if set to 1
        # --- GStreamer elements ---
        self.pipeline = Gst.Pipeline()
        self.streammux = Gst.ElementFactory.make("nvstreammux", "stream-mux")
        self.streammux.set_property("batch-size", self.max_sources)
        self.streammux.set_property("width", int(os.getenv("STREMUX_WIDTH", 1920)))
        self.streammux.set_property("height", int(os.getenv("STREMUX_HEIGHT", 1080)))
        self.streammux.set_property("batched-push-timeout", int(os.getenv("batched_push_timeout", 66666)))
        self.streammux.set_property("live-source", 1)
        self.pipeline.add(self.streammux)
        #------ hide class names from GIEs ------
        self.hide_class_names = []
        hide_pattern = re.compile(r"^GIE_(\d+)_HIDE_CLASS_NAMES$")
        for key, value in os.environ.items():
            match = hide_pattern.match(key)
            if match:
                names = {v.strip() for v in value.split(",") if v.strip()}
                for name in names:
                    self.hide_class_names.append(name)
        # ========= Primary and secondary inference elements ==============
        gie_pattern = re.compile(r"^GIE_(\d+)_CONFIG$")
        gie_configs = {}
        for key, value in os.environ.items():
            match = gie_pattern.match(key)
            if match:
                index = int(match.group(1))
                gie_configs[index] = value.strip()
        
        self.gie_configs = [gie_configs[i] for i in sorted(gie_configs.keys())]
        self.gies = []
        previous_elm = self.streammux
        for i, config in enumerate(self.gie_configs):
            gie = Gst.ElementFactory.make("nvinfer", f"gie_{i}")
            if not gie:
                raise RuntimeError(f"Failed to create GIE element for config {config}")
            gie.set_property("config-file-path", config)
            self.pipeline.add(gie)
            previous_elm.link(gie)
            self.gies.append(gie)
            previous_elm = gie
        
        self.demux = Gst.ElementFactory.make("nvstreamdemux", "stream-demux")
        self.pipeline.add(self.demux)
        previous_elm.link(self.demux)
        # Pre‑create request pads on demux for potential sources
        self.demux_src_pads = [self.demux.get_request_pad(f"src_{i}") for i in range(self.max_sources)]
        # --- Runtime bookkeeping ---
        self.sources = {}
        self.branches = {}
        self.urls_sources = []
        self._rtsp_mount_paths = set()
        self.notification_callback = notification_callback
        self.loop_event = asyncio.get_event_loop()
        # --- GLib/RTSP setup ---
        self.loop = GLib.MainLoop()
        self.rtsp_server = GstRtspServer.RTSPServer()
        self.rtsp_server.props.service = "8554"
        self.rtsp_server.attach(None)

        # --- Performance data ---
        self.pad_to_index = {}
        self.perf_data = PERF_DATA()
        # --- Processing queue for metadata ---
        self.process_queue = asyncio.Queue()
        self.source_bin_factory = SourceBinFactory()
        self.spot_manager = SpotManager(max_sources)
        # get the host from environment variable or default to localhost
        try:
            self.rabbitmq_manager = RabbitMQManager(host=os.getenv("RABBITMQ_HOST"),
                                                    username=os.getenv("RABBITMQ_DEFAULT_USER"),
                                                    password=os.getenv("RABBITMQ_DEFAULT_PASS"))
        except Exception as e:
            # raise RuntimeError(f"Failed to initialize RabbitMQ manager: {e}")
            print(f"Failed to initialize RabbitMQ manager: {e}")
            self.rabbitmq_manager = None
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_labels_status(self) -> dict:
        """
        Returns a visibility map of all labels across all GIEs.

        Output format:
        {
            "belt": False,
            "user": False,
            "helmet": True,
            ...
        }

        Where False = hidden, True = visible
        """
        all_labels = set()
        all_hidden_labels = set()

        i = 0
        while True:
            label_file = os.getenv(f"LABEL_{i}_FILE")
            if not label_file:
                break

            # Load label names
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    labels = {line.strip() for line in f if line.strip()}
                    all_labels.update(labels)

            i += 1

        # Build visibility map
        return {
            label: label not in self.hide_class_names
            for label in all_labels
        }
    
    def hide_class_name(self, class_name: str) -> bool:
        """
        Check if a class name is hidden in any GIE.
        
        Args:
            class_name (str): The class name to check.
        
        Returns:
            bool: True if the class name is hidden, False otherwise.
        """
        if not class_name:
            return False
        if class_name in self.hide_class_names:
            return True
        else:
            self.hide_class_names.append(class_name)
        return True
    
    def enable_class_name(self, class_name: str) -> bool:
        """
        Enable a class name that was previously hidden in any GIE.
        
        Args:
            class_name (str): The class name to enable.
        
        Returns:
            bool: True if the class name was enabled, False if it was not hidden.
        """
        if not class_name:
            return False
        if class_name in self.hide_class_names:
            self.hide_class_names.remove(class_name)
            return True
        return False

    def add_source(self, uri: str, rtsp_output_width: int = 640, rtsp_output_height: int = 640) -> int:
        '''
        Add a new source to the pipeline and create an RTSP mount for it.
            args:
                uri (str): The URI of the source (file or RTSP).
                rtsp_output_width (int): Width of the RTSP output stream.
                rtsp_output_height (int): Height of the RTSP output stream.
            returns:
                int: The index of the added source.
            raises:
                RuntimeError: If the pipeline is not running, or if the URI is invalid.
        '''

        if self.pipeline.get_state(1).state != Gst.State.PLAYING:
            raise RuntimeError("Pipeline is not running. Start the pipeline before adding sources.")
    
        if uri.startswith("file:///"):
            if not re.match(r"^file:///.+", uri):
                raise RuntimeError(f"Invalid file URI: {uri}")
            if not uri[7:] or not uri[7:].strip():
                raise RuntimeError(f"File URI is empty: {uri}")
    
        elif self.check_rtsp_link(uri) is False or not uri.startswith("rtsp://") or uri is None:
            raise RuntimeError(f"Invalid RTSP link: {uri}")

        spot, uuid, is_fresh = self.spot_manager.acquire()
        if spot is None:
            raise RuntimeError("No available spots for new source")

        # 1. Create and link source bin
        src_bin = self.source_bin_factory.create_source_bin(uuid, uri, "nvurisrcbin")
        self.pipeline.add(src_bin)
        src_pad = src_bin.get_static_pad("src")
        if is_fresh:
            mux_pad = self.streammux.get_request_pad(f"sink_{spot}")
        else:
            mux_pad = self.streammux.get_static_pad(f"sink_{spot}")
        if not mux_pad:
            raise RuntimeError(f"Failed to get request pad sink_{uuid} — maybe not released?")
        self.pad_to_index[src_pad] = spot

        src_pad.link(mux_pad)

        self.sources[spot] = src_bin

        # 2. Build per‑stream output branch and RTSP mount
        self._setup_output_branch(spot, uuid, rtsp_output_width, rtsp_output_height)
        
        src_bin.sync_state_with_parent()
        src_bin.set_state(Gst.State.PLAYING)
        self.urls_sources.append(uri)
        if self.rabbitmq_manager is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.rabbitmq_manager.create_queue(queue=str(uuid)),
                    self.loop_event
                )
            except Exception as e:
                print(f"[add_source] Failed to create RabbitMQ queue for {uuid}: {e}")

        return uuid

    # ============================================================================================================
    # check later this function not remove all the resources
    # ============================================================================================================
    def remove_source(self, uuid: str):
        """Remove an existing stream and clean up all associated resources."""

        index = self.spot_manager.get_spot_by_uuid(uuid)
        if index not in self.sources:
            print(f"No source with index {index}")
            return

        # --- Stop & remove output branch ---
        branch_elems = self.branches.get(index, [])
        for elem in branch_elems:
            elem.set_state(Gst.State.NULL)
            self.pipeline.remove(elem)
        self.branches.pop(index, None)

        # Remove RTSP mount
        mount_points = self.rtsp_server.get_mount_points()
        mount_points.remove_factory(f"/ds-test{index}")

        # --- Unlink & remove source bin ---
        src_bin = self.sources.pop(index)
        src_bin.set_state(Gst.State.NULL)
        self.pipeline.remove(src_bin)
        self.demux_src_pads[index].unlink(src_bin.get_static_pad("src"))
        # self.spot_manager.release(index)
        print(f"[✓] Removed source-bin-{index} and released spot {index}")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_output_branch(self, index: int, uuid: str, width: int, height: int):
        '''
        Setup the output branch for a specific stream index.
            args:
                index (int): The stream index.
                width (int): The width of the output video.
                height (int): The height of the output video.
                uuid: (str): unique value for the stream, used for RTSP mount path.
            returns:
                None
        '''

        conv1 = Gst.ElementFactory.make("nvvideoconvert", f"conv1_{uuid}")
        capsfilter1 = Gst.ElementFactory.make("capsfilter", f"capsfilter1_{uuid}")
        capsfilter1.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self.streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))
        conv1.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))

        osd = Gst.ElementFactory.make("nvdsosd", f"osd{uuid}")
        osd.set_property("display-bbox", 1)
        osd.set_property("display-mask", 1)

        conv2 = Gst.ElementFactory.make("nvvideoconvert", f"conv2_{uuid}")
        conv2.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))

        capsfilter2 = Gst.ElementFactory.make("capsfilter", f"capsfilter2_{uuid}")
        capsfilter2.set_property("caps", Gst.Caps.from_string(f"video/x-raw(memory:NVMM),width={width},height={height}, format=NV12"))

        enc = Gst.ElementFactory.make("nvv4l2h264enc", f"enc{uuid}")
        enc.set_property("bitrate", self.bitrate)

        pay = Gst.ElementFactory.make("rtph264pay", f"pay{uuid}")
        sink = Gst.ElementFactory.make("udpsink", f"sink{uuid}")
        sink.set_property("sync", 0)
        port = 5400 + index
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", port)

        for elem in (conv1, capsfilter1, osd, conv2, capsfilter2, enc, pay, sink):
            self.pipeline.add(elem)
            elem.sync_state_with_parent()

        self.demux_src_pads[index].link(conv1.get_static_pad("sink"))

        conv1.link(capsfilter1)

        capsfilter1.link(osd)

        osd.link(conv2)

        conv2.link(capsfilter2)

        capsfilter2.link(enc)

        enc.link(pay)

        pay.link(sink)

        osd.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self.conv_pad_buffer_probe, 0
        )

        self.branches[index] = [conv1, capsfilter1, osd, conv2, capsfilter2, enc, pay, sink]

        # RTSP setup
        factory = GstRtspServer.RTSPMediaFactory()
        launch = (
            f"( udpsrc name=pay0 port={port} buffer-size=524288 "
            f"caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" )"
        )
        factory.set_launch(launch)
        factory.set_shared(True)
        self.rtsp_server.get_mount_points().add_factory(f"/ds-test{uuid}", factory)
        print(f"Stream {uuid} at rtsp://localhost:8554/ds-test{uuid}")
        self._rtsp_mount_paths.add(f"/ds-test{uuid}")



    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            print(f"[bus] Element message: {struct.get_name()}")
            if struct and struct.get_name() == "attempt-exceeded":
                stream_id = struct.get_uint("stream-id")[1]
                print(f"[bus] Attempt exceeded for stream {stream_id}")
                # Notify about attempt exceeded
                if self.notification_callback:
                    asyncio.run_coroutine_threadsafe(
                        self.notification_callback({
                            "type": "attempt_exceeded",
                            "stream_id": stream_id,
                            "message": f"Attempt exceeded for stream {stream_id}"
                        }),
                        self.loop_event
                    )
                self.remove_source(stream_id)
            if struct and struct.get_name() == "GstNvStreamEos":
                stream_id = struct.get_uint("stream-id")[1]
                print(f"[bus] Stream {stream_id} EOS detected")
                self.remove_source(stream_id)
        elif t == Gst.MessageType.EOS:
            print("[bus] Global pipeline EOS (should not happen unless all sources ended)")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            struct = message.get_structure()
            stream_id = struct.get_uint("stream-id")[1]
            # Notify about error
            if self.notification_callback:
                asyncio.run_coroutine_threadsafe(
                    self.notification_callback({
                        "type": "error",
                        "stream_id": stream_id,
                        "message": f"Error on stream {stream_id}: {err.message}",
                        "debug": debug
                    }),
                    self.loop_event
                )
            print(f"[bus] Error on stream {stream_id}: {err.message} ({debug})")
            self.remove_source(stream_id)
        return True


    def perf_print_callback(self):
        fps_report = {
            stream_id: fps_obj.get_fps()
            for stream_id, fps_obj in self.perf_data.all_stream_fps.items()
        }
        print(f"FPS data: {fps_report}")
        # Notify about performance data
        if self.notification_callback:
            asyncio.run_coroutine_threadsafe(
                self.notification_callback({
                    "type": "performance",
                    "FPS": fps_report
                }),
                self.loop_event
            )
        return True

    
    def conv_pad_buffer_probe(self, pad, info, u_data):
        '''
        This probe function processes each buffer that reaches the OSD pad.
        It extracts metadata, enqueues work for processing, and handles mask data.
        It also calculates FPS for each stream and updates the performance data.

        '''        
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            flat_frame = np.array(n_frame, copy=True)

            #? hide the mask data for objects with classes that are not in the range of interest
            l_obj = frame_meta.obj_meta_list
            index = 0
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                next_obj = l_obj.next
                gie_unique_id = obj_meta.unique_component_id
                
                self.process_queue.put_nowait({
                    "flat_frame": flat_frame,
                    "class_id": obj_meta.class_id if obj_meta.class_id is not None else None,
                    "confidence": obj_meta.confidence if obj_meta.confidence is not None else None,
                    "label": obj_meta.obj_label if obj_meta.obj_label is not None else None,
                    "gie_unique_id": gie_unique_id,
                    "object_id": obj_meta.object_id if obj_meta.object_id is not None else None,
                    "source_id": frame_meta.source_id if frame_meta.source_id is not None else None,
                    "frame_number": frame_meta.frame_num if frame_meta.frame_num is not None else None,
                    "left": obj_meta.rect_params.left if obj_meta.rect_params is not None else None,
                    "top": obj_meta.rect_params.top if obj_meta.rect_params is not None else None,
                    "width": obj_meta.rect_params.width if obj_meta.rect_params is not None else None,
                    "height": obj_meta.rect_params.height if obj_meta.rect_params is not None else None,
                    "mask": obj_meta.mask_params.get_mask_array().reshape(
                        (obj_meta.mask_params.height, obj_meta.mask_params.width)
                    ) if obj_meta.mask_params is not None and obj_meta.mask_params.data else None,
                    "therchold": obj_meta.mask_params.threshold if obj_meta.mask_params is not None else None,
                    "mask_width": obj_meta.mask_params.width if obj_meta.mask_params is not None else None,
                    "mask_height": obj_meta.mask_params.height if obj_meta.mask_params is not None else None,
                    "index": index
                })

                label = obj_meta.obj_label if obj_meta.obj_label is not None else None
                if label and label in self.hide_class_names:
                        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                        
                if obj_meta.mask_params is not None and obj_meta.mask_params.data:
                    #? hide the bounding box of only segmentation objects
                    if self.disable_box_segmentation: 
                        rect = obj_meta.rect_params
                        rect.border_width = 0
                        rect.border_color.alpha = 0.0
                l_obj = next_obj
                index += 1
            
            # ----------------------------------------------------------------------
            # CALCULATE FPS
            # ----------------------------------------------------------------------
            stream_id = f"stream{frame_meta.pad_index}"

            if stream_id not in self.perf_data.all_stream_fps:
                from FPS import GETFPS
                self.perf_data.all_stream_fps[stream_id] = GETFPS(stream_id)

            self.perf_data.update_fps(stream_id)
            # ----------------------------------------------------------------------

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    async def _processing_worker_loop(self):
        objects = []
        while True:
            task = await self.process_queue.get()
            flat_frame = task["flat_frame"]
            class_id = task.get("class_id")
            confidence = task.get("confidence")
            gie_unique_id = task.get("gie_unique_id")
            object_id = task.get("object_id")
            source_id = task.get("source_id")
            frame_number = task.get("frame_number")
            label = task.get("label")
            left = task.get("left")
            top = task.get("top")
            width = task.get("width")
            height = task.get("height")
            mask = task.get("mask")
            index = task.get("index")

            try:

                frame_image = cv2.cvtColor(flat_frame, cv2.COLOR_RGBA2BGR)
                mask_b64 = None
                mask_img = None
                if mask is not None:
                    mask_img = resize_mask(mask, math.floor(width), math.floor(height), task.get("therchold"))
                    mask_b64 = encode_mask_to_base64(mask_img)
                if left is not None or mask_b64 is not None:
                    objects.append({
                        "object_id": object_id,
                        "model_id": gie_unique_id,
                        "class_id": class_id,
                        "label": label,
                        "confidence": confidence,
                        "bbox": {
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height
                        },
                        "mask": mask_b64,
                    })
                if index == 0 and objects.__len__() > 0:
                    if objects.__len__() > 0:
                        uuid = self.spot_manager.get_uuid(source_id)
                        metadata = {
                            "source_id": uuid,
                            "frame_number": frame_number,
                            "objects": objects,
                            "frame_base64": transform_image_to_base64(frame_image),
                        }
                    else:
                        metadata = {}

                    if self.rabbitmq_manager:
                        try:
                            await self.rabbitmq_manager.publish_message(
                                queue=str(uuid),
                                message=metadata
                            )
                        except Exception as e:
                            print(f"[worker] Failed to send message to RabbitMQ: {e}")
                        
                        objects = []
                


            except Exception as e:
                print(f"[worker] Error processing frame: {e}")


    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    def start(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        GLib.timeout_add(5000, self.perf_print_callback)
        time.sleep(1)
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def check_rtsp_link(self, uri: str) -> bool:
        """Check if the RTSP link is valid by trying to open it with OpenCV."""

        cap = cv2.VideoCapture(uri)
        try:
            if not cap.isOpened():
                return False
            else:
                return True
        except Exception as e:
            print(f"Error checking RTSP link {uri}: {e}")
            return False
        return False




