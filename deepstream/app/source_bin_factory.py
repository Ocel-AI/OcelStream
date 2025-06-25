import gi
gi.require_version('Gst', '1.0') 
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst

class SourceBinFactory:
    def __init__(self):
        """
        Initialize the SourceBinFactory.
        This factory creates source bins for different types of sources.
        """
        Gst.init(None)

    def create_source_bin(self, index: int, uri: str, source_type: str = "nvurisrcbin") -> Gst.Bin:
        """
        Create a source bin based on the specified type.

        :param index: Source index (used for naming)
        :param uri: The URI or location of the stream
        :param source_type: Type of source ('uridecodebin', 'nvurisrcbin', 'rtspsrc')
        :return: Gst.Bin containing the configured source
        """
        if source_type == "uridecodebin":
            return self._create_uridecodebin(index, uri)
        elif source_type == "nvurisrcbin":
            return self._create_nvurisrcbin(index, uri)
        elif source_type == "rtspsrc":
            return self._create_rtspsrc(index, uri)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    def _create_uridecodebin(self, index: int, uri: str) -> Gst.Bin:
        bin_name = f"source-bin-{index}"
        bin_ = Gst.Bin.new(bin_name)

        uridecodebin = Gst.ElementFactory.make("uridecodebin", f"uridecodebin-{index}")
        uridecodebin.set_property("uri", uri)
        uridecodebin.connect("pad-added", self._cb_decode_pad_added, bin_)
        bin_.add(uridecodebin)

        ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin_.add_pad(ghost)

        return bin_

    def _create_nvurisrcbin(self, index: int, uri: str) -> Gst.Bin:
        """ 
            This GstBin is a GStreamer source bin. This bin is a wrapper over uridecodebin with additional 
            functionality of the file looping, rtsp reconnection and smart record.
        """
        bin_name = f"source-bin-{index}"
        bin_ = Gst.Bin.new(bin_name)

        nvurisrc = Gst.ElementFactory.make("nvurisrcbin", f"src-{index}")
        nvurisrc.set_property("uri", uri)
        nvurisrc.set_property("rtsp-reconnect-interval", 5)
        nvurisrc.set_property("rtsp-reconnect-attempts", 10)
        nvurisrc.set_property("select-rtp-protocol", 4)
        nvurisrc.set_property("disable-audio", True)


        bin_.add(nvurisrc)

        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin_.add_pad(ghost_pad)

        nvurisrc.connect("pad-added", self._on_nvurisrc_pad_added, ghost_pad)

        return bin_

    def _create_rtspsrc(self, index: int, uri: str) -> Gst.Bin:
        bin_name = f"source-bin-{index}"
        bin_ = Gst.Bin.new(bin_name)

        rtspsrc = Gst.ElementFactory.make("rtspsrc", f"rtspsrc-{index}")
        rtspsrc.set_property("location", uri)
        rtspsrc.set_property("latency", 300)
        rtspsrc.set_property("protocols", 3)
        rtspsrc.set_property("timeout", 120000000)
        rtspsrc.set_property("tcp-timeout", 12000000)
        rtspsrc.set_property("drop-on-latency", True)
        rtspsrc.set_property("udp-reconnect", True)

        depay = Gst.ElementFactory.make("rtph264depay", f"depay-{index}")
        parse = Gst.ElementFactory.make("h264parse", f"parse-{index}")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{index}")

        if not all([rtspsrc, depay, parse, decoder]):
            raise RuntimeError("Could not create one of the RTSP source elements")

        for elem in [depay, parse, decoder]:
            bin_.add(elem)

        bin_.add(rtspsrc)

        depay.link(parse)
        parse.link(decoder)

        rtspsrc.connect("pad-added", lambda src, pad: self._on_rtspsrc_pad_added(pad, depay))

        src_pad = decoder.get_static_pad("src")
        ghost = Gst.GhostPad.new("src", src_pad)
        bin_.add_pad(ghost)

        return bin_

    @staticmethod
    def _cb_decode_pad_added(decodebin, pad, bin_):
        if pad.get_current_caps().to_string().startswith("video"):
            ghost = bin_.get_static_pad("src")
            if ghost.get_target():
                ghost.set_target(None)
            ghost.set_target(pad)

    @staticmethod
    def _on_nvurisrc_pad_added(src, pad, ghost):
        print(f"[+] nvurisrcbin pad added: {pad.get_name()}")
        if not ghost.get_target():
            ghost.set_target(pad)

    @staticmethod
    def _on_rtspsrc_pad_added(pad, depay):
        print(f"[+] Pad added to rtspsrc: {pad.get_name()}")
        sink_pad = depay.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)
