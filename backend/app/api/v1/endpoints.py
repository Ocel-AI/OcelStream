from fastapi import APIRouter, HTTPException, WebSocket, File, UploadFile, WebSocketDisconnect
from fastapi.responses import JSONResponse
from backend.app.models import StreamRequest
from backend.app.core.context import pipeline

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
def remove_stream(uuid: str):
    if uuid not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    pipeline.remove_source(uuid)
    del active_streams[uuid]
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