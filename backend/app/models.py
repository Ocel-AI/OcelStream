from pydantic import BaseModel

class StreamRequest(BaseModel):
    uri: str
    rtsp_output_width: int
    rtsp_output_height: int