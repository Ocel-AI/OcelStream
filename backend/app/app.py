from fastapi import FastAPI
from backend.app.api.v1.endpoints import router
import asyncio
from backend.app.core.context import pipeline
import threading
import time

app = FastAPI(title="DeepStream API")

app.include_router(router, prefix="/api/v1", tags=["deepstream"])
# ----------------- Pipeline -----------------
threading.Thread(target=pipeline.start, daemon=True).start()
time.sleep(3)

@app.on_event("startup")
async def startup_event():
    await pipeline.rabbitmq_manager.connect()
    asyncio.create_task(pipeline._processing_worker_loop())

