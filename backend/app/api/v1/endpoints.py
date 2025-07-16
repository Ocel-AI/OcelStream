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

active_streams = {}
clients_notifications: Set[WebSocket] = set()
notification_queue = queue.Queue()




async def notification_handler(data: Dict):
    """Handle notifications and queue them for WebSocket clients"""
    notification_queue.put(data)

# ----------------- Pipeline -----------------
pipeline = DynamicRTSPPipeline(max_sources=15, notification_callback=notification_handler)
threading.Thread(target=pipeline.start, daemon=True).start()
time.sleep(3)





# ----------------- Endpoints -----------------
@router.post("/add")
def add_stream(req: StreamRequest):
    try:
        rtsp_output_width = req.rtsp_output_width
        rtsp_output_height = req.rtsp_output_height
        source_uri = req.uri 
        print (f"Adding stream: {source_uri} with width: {rtsp_output_width} and height: {rtsp_output_height}")
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

@router.get("/labels/status")
def get_labels_status():
    try:
        labels_status = pipeline.get_labels_status()
        return JSONResponse(content=labels_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hide_class_name")
def hide_class_name(class_name: str):
    try:
        pipeline.hide_class_name(class_name)
        return {"message": f"Class name '{class_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enable_class_name")
def enable_class_name(class_name: str):
    try:
        pipeline.enable_class_name(class_name)
        return {"message": f"Class name '{class_name}' enabled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams")
def list_streams():
    return active_streams


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

    
