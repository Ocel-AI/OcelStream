import threading
import gi
import time
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/apps/common')

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
from bus_call import bus_call

Gst.init(None)

class DynamicRTSPPipeline:
    """DeepStream pipeline that supports runtime add & remove of sources.

    Each source gets its own RTSP mount (rtsp://<host>:8554/ds-test<id>). The class
    now exposes remove_source() in addition to add_source().
    """

    def __init__(self, max_sources: int = 4):
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
        self.pgie.set_property("config-file-path", "./config_infer_primary_yolo11.txt")
        self.pipeline.add(self.pgie)

        self.demux = Gst.ElementFactory.make("nvstreamdemux", "stream-demux")
        self.pipeline.add(self.demux)

        # Pre‑create request pads on demux for potential sources
        self.demux_src_pads = [self.demux.get_request_pad(f"src_{i}") for i in range(self.max_sources)]

        # Link static portion of pipeline
        self.streammux.link(self.pgie)
        self.pgie.link(self.demux)

        # --- Runtime bookkeeping ---
        self.sources = {}          # index -> source bin
        self.branches = {}         # index -> list[Gst.Element] (conv/osd/enc/pay/sink)

        # --- GLib/RTSP setup ---
        self.loop = GLib.MainLoop()
        self.rtsp_server = GstRtspServer.RTSPServer()
        self.rtsp_server.props.service = "8554"
        self.rtsp_server.attach(None)


        self.pad_to_index = {}

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

        ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin_.add_pad(ghost)
        return bin_

    @staticmethod
    def _cb_decode_pad_added(decodebin, pad, bin_):
        if pad.get_current_caps().to_string().startswith("video"):
            ghost = bin_.get_static_pad("src")
            if not ghost.has_current_caps():
                ghost.set_target(pad)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_source(self, uri: str) -> int:
        """Add a new stream. Returns its stream index."""
        index = len(self.sources)
        if index >= self.max_sources:
            raise RuntimeError("Maximum sources reached")

        # 1. Create and link source bin
        src_bin = self._create_source_bin(index, uri)
        self.pipeline.add(src_bin)
        src_pad = src_bin.get_static_pad("src")
        mux_pad = self.streammux.get_request_pad(f"sink_{index}")
        self.pad_to_index[src_pad] = index
        src_pad.link(mux_pad)
        src_bin.sync_state_with_parent()

        self.sources[index] = src_bin

        # 2. Build per‑stream output branch and RTSP mount
        self._setup_output_branch(index)
        return index

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

        sink_pad = self.streammux.get_static_pad(f"sink_{index}")
        if sink_pad:
            self.streammux.release_request_pad(sink_pad)

        demux_pad = self.demux_src_pads[index]
        if demux_pad.is_linked():
            demux_pad.unlink(demux_pad.get_peer())

        print(f"Source {index} removed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_output_branch(self, index: int):
        """Create conv → osd → enc → pay → udpsink chain for stream index."""
        conv = Gst.ElementFactory.make("nvvideoconvert", f"conv{index}")
        osd = Gst.ElementFactory.make("nvdsosd", f"osd{index}")
        enc = Gst.ElementFactory.make("nvv4l2h264enc", f"enc{index}")
        enc.set_property("bitrate", self.bitrate)
        pay = Gst.ElementFactory.make("rtph264pay", f"pay{index}")
        sink = Gst.ElementFactory.make("udpsink", f"sink{index}")
        port = 5400 + index
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", port)

        for elem in (conv, osd, enc, pay, sink):
            self.pipeline.add(elem)
            elem.sync_state_with_parent()

        # Link demux -> conv ... sink
        self.demux_src_pads[index].link(conv.get_static_pad("sink"))
        conv.link(osd)
        conv.get_static_pad("sink").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            lambda pad, info: self.eos_probe_callback(pad, info, index)
        )
        osd.link(enc)
        enc.link(pay)
        pay.link(sink)

        # Store branch for cleanup
        self.branches[index] = [conv, osd, enc, pay, sink]

        # Expose via RTSP
        factory = GstRtspServer.RTSPMediaFactory()
        launch = (
            f"( udpsrc name=pay0 port={port} buffer-size=524288 "
            f"caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" )"
        )
        factory.set_launch(launch)
        factory.set_shared(True)
        self.rtsp_server.get_mount_points().add_factory(f"/ds-test{index}", factory)
        print(f"Stream {index} at rtsp://localhost:8554/ds-test{index}")

    def eos_probe_callback(self, pad, info, index):
        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
            event = info.get_event()
            if event.type == Gst.EventType.EOS:
                print(f"[pad-probe] EOS detected on stream {index}")
                self.remove_source(index)
        return Gst.PadProbeReturn.OK


    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error:", err, debug)
            self.loop.quit()
        return True

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    def start(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline started")
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
    uri = "file:///opt/nvidia/deepstream/deepstream-7.1/sources/my_data/best.mp4"
    app = DynamicRTSPPipeline(max_sources=4)
    threading.Thread(target=app.start, daemon=True).start()

    time.sleep(3)
    id0 = app.add_source(uri)
    id1 = app.add_source(uri)
    time.sleep(10)
    # Remove first stream after 10 s
    app.remove_source(id0)
    # Keep running …
    while True:
        time.sleep(1)
