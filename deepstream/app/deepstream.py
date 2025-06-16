import threading
import gi
import time
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/apps/common')

from gi.repository import Gst, GstRtspServer, GLib
from bus_call import bus_call
from FPS import PERF_DATA
import pyds
from spotmanager import SpotManager
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


gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

class DynamicRTSPPipeline:
    """DeepStream pipeline that supports runtime add & remove of sources.

    Each source gets its own RTSP mount (rtsp://<host>:8554/ds-test<id>). The class
    now exposes remove_source() in addition to add_source().
    """

    def __init__(self, max_sources: int = 5, broadcast_callback=None):

        Gst.init(None)
        # --- Pipeline‑wide parameters ---
        self.max_sources = max_sources
        self.codec = "H264"
        self.bitrate = 4_000_000  # 4 Mbps per stream
        # --- GStreamer elements ---
        self.pipeline = Gst.Pipeline()
        self.streammux = Gst.ElementFactory.make("nvstreammux", "stream-mux")
        self.streammux.set_property("batch-size", max_sources)
        self.streammux.set_property("width", 1920)
        self.streammux.set_property("height", 1080)
        self.streammux.set_property("batched-push-timeout", 40_000)
        self.pipeline.add(self.streammux)

        self.pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        self.sgie = Gst.ElementFactory.make("nvinfer", "spgie")
        self.pgie.set_property("config-file-path", "/opt/nvidia/deepstream/deepstream-7.1/sources/my_data/deepstream/config/config_infer_primary_yolo11.txt")
        self.sgie.set_property("config-file-path", "/opt/nvidia/deepstream/deepstream-7.1/sources/my_data/deepstream/config/config_pgie_yolo_seg.txt")
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.sgie)

        self.demux = Gst.ElementFactory.make("nvstreamdemux", "stream-demux")
        self.pipeline.add(self.demux)

        # Pre‑create request pads on demux for potential sources
        self.demux_src_pads = [self.demux.get_request_pad(f"src_{i}") for i in range(self.max_sources)]

        # Link static portion of pipeline
        self.streammux.link(self.pgie)
        self.pgie.link(self.sgie)
        self.sgie.link(self.demux)
        self.pgie.link(self.demux)

        # --- Runtime bookkeeping ---
        self.sources = {}          # index -> source bin
        self.branches = {}         # index -> list[Gst.Element] (conv/osd/enc/pay/sink)
        self.urls_sources = []     # index -> uri
        self._rtsp_mount_paths = set()

        # --- GLib/RTSP setup ---
        self.loop = GLib.MainLoop()
        self.rtsp_server = GstRtspServer.RTSPServer()
        self.rtsp_server.props.service = "8554"
        self.rtsp_server.attach(None)

        # --- Performance data ---
        self.pad_to_index = {}
        self.perf_data = PERF_DATA()

        # --- Spot management for dynamic sources ---
        self.spot_manager = SpotManager(max_sources)

        # --- Pad probe for conv sink ---
        self.MIN_CONFIDENCE = 0.3
        self.MAX_CONFIDENCE = 0.4

        # --- Callbacks for send data via WebSocket ---
        self.broadcast_callback = broadcast_callback
        self.loop_event = asyncio.get_event_loop()
        # self.streammux.set_property('live-source', 1)  # Enable live source mode

        self.process_queue = queue.Queue()
        self.processor_thread = threading.Thread(target=self._processing_worker_loop, daemon=True)
        self.processor_thread.start()
    # ------------------------------------------------------------------
    # Source bin helpers
    # ------------------------------------------------------------------
    def _create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        """Builds a uridecodebin wrapped in a Bin with a ghost src pad."""
        bin_name = f"source-bin-{index}"
        bin_ = Gst.Bin.new(bin_name)

        uridecodebin = Gst.ElementFactory.make("uridecodebin", f"uridecodebin-{index}")
        uridecodebin.set_property("uri", uri)
        uridecodebin.connect("pad-added", self._cb_decode_pad_added, bin_)
        bin_.add(uridecodebin)

        # Create ghost pad with no target
        ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin_.add_pad(ghost)

        return bin_

    @staticmethod
    def _cb_decode_pad_added(decodebin, pad, bin_):
        if pad.get_current_caps().to_string().startswith("video"):
            ghost = bin_.get_static_pad("src")
            if ghost.get_target():
                ghost.set_target(None)
            ghost.set_target(pad)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_source(self, uri: str) -> int:
        """Add a new stream. Returns its stream index."""
        # index = len(self.sources)
        spot, is_fresh = self.spot_manager.acquire()
        if spot is None:
            raise RuntimeError("No available spots for new source")

        # 1. Create and link source bin
        src_bin = self._create_source_bin(spot, uri)
        self.pipeline.add(src_bin)
        src_pad = src_bin.get_static_pad("src")
        if is_fresh:
            mux_pad = self.streammux.get_request_pad(f"sink_{spot}")
        else:
            mux_pad = self.streammux.get_static_pad(f"sink_{spot}")
        if not mux_pad:
            raise RuntimeError(f"Failed to get request pad sink_{spot} — maybe not released?")
        self.pad_to_index[src_pad] = spot

        src_pad.link(mux_pad)

        self.sources[spot] = src_bin

        # 2. Build per‑stream output branch and RTSP mount
        self._setup_output_branch(spot)
        src_bin.sync_state_with_parent()
        src_bin.set_state(Gst.State.PLAYING)
        self.urls_sources.append(uri)

        return spot

    # ============================================================================================================
    # check later this function not remove all the resources
    # ============================================================================================================
    def remove_source(self, index: int):
        """Remove an existing stream and clean up all associated resources."""
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
        # self.spot_manager.release(index)
        print(f"[✓] Removed source-bin-{index} and released spot {index}")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_output_branch(self, index: int):
        conv1 = Gst.ElementFactory.make("nvvideoconvert", f"conv1_{index}")
        capsfilter1 = Gst.ElementFactory.make("capsfilter", f"capsfilter1_{index}")
        capsfilter1.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self.streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))
        conv1.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))

        osd = Gst.ElementFactory.make("nvdsosd", f"osd{index}")
        osd.set_property("display-bbox", 1)
        osd.set_property("display-mask", 1)

        conv2 = Gst.ElementFactory.make("nvvideoconvert", f"conv2_{index}")
        conv2.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))
        capsfilter2 = Gst.ElementFactory.make("capsfilter", f"capsfilter2_{index}")
        capsfilter2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        enc = Gst.ElementFactory.make("nvv4l2h264enc", f"enc{index}")
        enc.set_property("bitrate", self.bitrate)

        pay = Gst.ElementFactory.make("rtph264pay", f"pay{index}")
        sink = Gst.ElementFactory.make("udpsink", f"sink{index}")
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

        # Pad probes (optional but keep for eos/debug)
        conv1.get_static_pad("sink").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            lambda pad, info: self.eos_probe_callback(pad, info, index)
        )
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
        self.rtsp_server.get_mount_points().add_factory(f"/ds-test{index}", factory)
        print(f"Stream {index} at rtsp://localhost:8554/ds-test{index}")
        self._rtsp_mount_paths.add(f"/ds-test{index}")


    # ------------------------------------------------------------------

    def eos_probe_callback(self, pad, info, index):
        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
            event = info.get_event()
            if event.type == Gst.EventType.EOS:
                print(f"[pad-probe] EOS detected on stream {index}")
                # self.remove_source(index)
        return Gst.PadProbeReturn.OK

    # def bus_call(self, bus, message, loop):
    #     t = message.type
    #     if t == Gst.MessageType.EOS:
    #         print("End-of-stream")
    #         # self.loop.quit()
    #         # self.start()
    #     elif t == Gst.MessageType.ERROR:
    #         err, debug = message.parse_error()
    #         print("Error:", err, debug)
    #         self.loop.quit()
    #     return True

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct and struct.get_name() == "GstNvStreamEos":
                stream_id = struct.get_uint("stream-id")[1]  # .get_uint returns (bool, val)
                print(f"[bus] Stream EOS detected for stream {stream_id}")
                self.remove_source(stream_id)
        elif t == Gst.MessageType.EOS:
            print("[bus] Global pipeline EOS (should not happen unless all sources ended)")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("[bus] Error:", err, debug)
            self.loop.quit()
        return True


    def perf_print_callback(self):
        self.perf_data.perf_print_callback()
        return True

    
    def conv_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            # Enqueue work
            self.process_queue.put({
                "gst_buffer": gst_buffer,
                "batch_id": frame_meta.batch_id,
                "frame_meta": frame_meta
            })
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

    def _processing_worker_loop(self):
        while True:
            task = self.process_queue.get()
            gst_buffer = task["gst_buffer"]
            batch_id = task["batch_id"]
            frame_meta = task["frame_meta"]

            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
                flat_frame = np.array(n_frame, copy=True)
                frame_image = cv2.cvtColor(flat_frame, cv2.COLOR_RGBA2BGR)

                objects = []
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect = obj_meta.rect_params
                    maskparams = obj_meta.mask_params
                    mask_b64 = None
                    if maskparams is not None and maskparams.data:
                        mask_img = resize_mask(maskparams, math.floor(rect.width), math.floor(rect.height))
                        mask_b64 = encode_mask_to_base64(mask_img)
                    objects.append({
                        "object_id": obj_meta.object_id,
                        "class_id": obj_meta.class_id,
                        "confidence": obj_meta.confidence,
                        "bbox": {
                            "left": rect.left,
                            "top": rect.top,
                            "width": rect.width,
                            "height": rect.height
                        },
                        "mask": mask_b64
                    })
                    l_obj = l_obj.next

                metadata = {
                    "source_id": frame_meta.source_id,
                    "frame_number": frame_meta.frame_num,
                    "objects": objects,
                    "frame_base64": transform_image_to_base64(frame_image)
                }

                if self.broadcast_callback:
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_callback(metadata),
                        self.loop_event
                    )

            except Exception as e:
                print(f"[worker] Error processing frame: {e}")


    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    def start(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        GLib.timeout_add(1000, self.perf_data.perf_print_callback)
        # time.sleep(1)
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)






# ----------------------------------------------------------------------
# Stand‑alone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    uri_file = "file:///opt/nvidia/deepstream/deepstream-7.1/samples/streams/sample_1080p_h264.mp4"
    file1_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/1.mp4"
    file2_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/2.mp4"
    file3_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/3.mp4"
    file4_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/4.mp4"
    file5_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/5.mp4"
    file6_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/6.mp4"
    file7_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/4.mp4"
    file8_uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/static/8.mp4"
    live_uri = "rtsp://localhost:4000/looped"
    app = DynamicRTSPPipeline(max_sources=7)
    threading.Thread(target=app.start, daemon=True).start()
    time.sleep(1)  # Ensure pipeline is up
    id0 = app.add_source("rtsp://localhost:4000/looped")
    # time.sleep(1)
    # # # app.remove_source(id0)
    # id00 = app.add_source("rtsp://localhost:4001/looped")
    # time.sleep(1)

    # id000 = app.add_source("rtsp://localhost:4002/looped")
    # time.sleep(1)
    # id000 = app.add_source(file7_uri)
    # time.sleep(1)
    # id000 = app.add_source(uri_file)
    # time.sleep(1)
    # id000 = app.add_source(file3_uri)
    # time.sleep(1)
    # id000 = app.add_source(file1_uri)
    # # time.sleep(10)
    # app.remove_source(id000)
    # for i in range(30):
    # app.add_source(uri_file)
    while True:
        time.sleep(1)

