from pydantic import BaseModel

class StreamRequest(BaseModel):
    uri: str
