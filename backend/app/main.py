from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from typing import Set
from pydantic import BaseModel
import threading
import time

from fastapi.responses import JSONResponse
import shutil
import os
import queue
import json
from typing import Dict
import asyncio

import sys
sys.path.append("../../deepstream/app")  # relative from backend/
from deepstream import DynamicRTSPPipeline


# ================== for testing purposes ==================
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
upload_streams = {}
# ==========================================================

app = FastAPI()
connected_clients: Set[WebSocket] = set()


# Keep track of stream IDs and URIs
active_streams = {}

class StreamRequest(BaseModel):
    uri: str


#-----------------------------------------------------------------------
# for real-time detection updates
#-----------------------------------------------------------------------
detection_queue = queue.Queue()

async def broadcast_data(data: Dict):
    """Broadcast data to all connected WebSocket clients"""
    message = json.dumps(data)
    for client in connected_clients.copy():
        try:
            await client.send_text(message)
        except:
            connected_clients.remove(client)

async def process_detection_queue():
    """Process detection queue and broadcast to WebSocket clients"""
    while True:
        try:
            data = detection_queue.get_nowait()
            await broadcast_data(data)
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error processing detection queue: {e}")
            await asyncio.sleep(0.1)

def queue_detection(data: Dict):
    """Queue detection data for processing in the main event loop"""
    detection_queue.put(data)


@app.on_event("startup")
async def startup_event():
    """Start the detection queue processor on startup"""
    asyncio.create_task(process_detection_queue())

pipeline = DynamicRTSPPipeline(max_sources=4, broadcast_callback=queue_detection)
# Start DeepStream pipeline in background
threading.Thread(target=pipeline.start, daemon=True).start()
time.sleep(3)  # Ensure pipeline is up

@app.post("/add")
def add_stream(req: StreamRequest):
    try:
        idx = pipeline.add_source(req.uri)
        active_streams[idx] = req.uri
        return {"message": "Stream added", "index": idx, "rtsp": f"rtsp://localhost:8554/ds-test{idx}"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/remove/{index}")
def remove_stream(index: int):
    if index not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    pipeline.remove_source(index)
    uri = active_streams.pop(index)
    return {"message": "Stream removed", "index": index, "uri": uri}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except:
        connected_clients.remove(websocket)


# ================== for testing purposes ==================
@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Format for DeepStream
        uri = f"file://{os.path.abspath(file_path)}"

        # Add to pipeline
        index = pipeline.add_source(uri)
        upload_streams[index] = file.filename

        return {
            "message": "Video submitted and added to DeepStream pipeline",
            "index": index,
            "uri": uri,
            "rtsp": f"rtsp://localhost:8554/ds-test{index}"
        }

    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/cleanup/{index}")
def cleanup_video(index: int):
    if index not in upload_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    pipeline.remove_source(index)
    filename = upload_streams.pop(index)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": "Stream removed and file deleted", "index": index}

# ======================================================================

@app.get("/streams")
def list_streams():
    return active_streams
