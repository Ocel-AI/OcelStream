from fastapi import APIRouter, HTTPException, WebSocket, File, UploadFile, WebSocketDisconnect
from fastapi.responses import JSONResponse
from backend.app.models import StreamRequest

import os
import shutil
import threading
import time
import json
import asyncio
import queue
from typing import Set, Dict


import sys
sys.path.append("deepstream/app") 
from deepstream import DynamicRTSPPipeline


router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
upload_streams = {}
active_streams = {}
connected_clients: Set[WebSocket] = set()
clients_notifications: Set[WebSocket] = set()
detection_queue = queue.Queue()
notification_queue = queue.Queue()



async def queue_detection(data: Dict):
    """Queue detection data for processing in the main event loop"""
    detection_queue.put(data)


async def notification_handler(data: Dict):
    """Handle notifications and queue them for WebSocket clients"""
    notification_queue.put(data)


# ----------------- Pipeline -----------------
pipeline = DynamicRTSPPipeline(max_sources=15, metadata_callback=queue_detection, notification_callback=notification_handler)
threading.Thread(target=pipeline.start, daemon=True).start()
time.sleep(3)


# ----------------- Endpoints -----------------
@router.post("/add")
def add_stream(req: StreamRequest):
    try:
        rtsp_output_width = req.rtsp_output_width
        rtsp_output_height = req.rtsp_output_height
        source_uri = req.uri 
        idx = pipeline.add_source(source_uri, rtsp_output_width, rtsp_output_height)

        active_streams[idx] = req.uri
        return {"message": "Stream added", "index": idx, "rtsp": f"rtsp://localhost:8554/ds-test{idx}"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/remove/{index}")
def remove_stream(index: int):
    if index not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    pipeline.remove_source(index)
    uri = active_streams.pop(index)
    return {"message": "Stream removed", "index": index, "uri": uri}


@router.get("/streams")
def list_streams():
    return active_streams

@router.websocket("/ws")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            if not detection_queue.empty():
                data = detection_queue.get_nowait()
                await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except:
        connected_clients.remove(websocket)


@router.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()
    clients_notifications.add(websocket)
    try:
        while True:
            if not notification_queue.empty():
                data = notification_queue.get_nowait()
                await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except:
        clients_notifications.remove(websocket)
    

# -------------------------------------------------------
# for testing purposes, we can upload a video file
#--------------------------------------------------------

@router.post("/test/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uri = f"file://{os.path.abspath(file_path)}"
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

@router.delete("/test/cleanup/{index}")
def cleanup_video(index: int):
    if index not in upload_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    pipeline.remove_source(index)
    filename = upload_streams.pop(index)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": "Stream removed and file deleted", "index": index}
