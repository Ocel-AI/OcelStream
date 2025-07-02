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
from deepstream import SpotManager


router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
upload_streams = {}
active_streams = {}
connected_clients: Set[WebSocket] = set()
clients_notifications: Set[WebSocket] = set()
detection_queue = queue.Queue()
notification_queue = queue.Queue()
stream_clients: Dict[int, Set[WebSocket]] = {}





async def queue_detection(data: Dict):
    """Queue detection data for processing in the main event loop"""
    stream_id = data.get("source_id")
    if stream_id is None:
        return

    if stream_id in stream_clients:
        for ws in list(stream_clients[stream_id]):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                stream_clients[stream_id].remove(ws)


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
        rtsp_output_width = 640
        rtsp_output_height = 640
        source_uri = req.uri 
        uuid = pipeline.add_source(source_uri, rtsp_output_width, rtsp_output_height)
        active_streams[uuid] = f"rtsp://localhost:8554/ds-test{uuid}"
        return {"message": "Stream added", "uuid": uuid, "rtsp": f"rtsp://localhost:8554/ds-test{uuid}"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/remove/{uuid}")
def remove_stream(uuid: int):
    # if index not in active_streams:
    #     raise HTTPException(status_code=404, detail="Stream not found")
    pipeline.remove_source(uuid)
    # uri = active_streams.pop(index)
    uri = "for now we do not return the uri"
    return {"message": "Stream removed", "uuid": uuid, "uri": uri}


@router.get("/streams")
def list_streams():
    return active_streams


@router.websocket("/ws/{uuid}")
async def stream_specific_ws(websocket: WebSocket, uuid: str):

    await websocket.accept()
    if uuid not in stream_clients:
        stream_clients[uuid] = set()
    stream_clients[uuid].add(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        stream_clients[uuid].remove(websocket)
        if not stream_clients[uuid]:
            del stream_clients[uuid]


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
    
