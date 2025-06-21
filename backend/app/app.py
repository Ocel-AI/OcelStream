from fastapi import FastAPI
from backend.app.api.v1.endpoints import router
import asyncio

app = FastAPI(title="DeepStream API")

app.include_router(router, prefix="/api/v1", tags=["deepstream"])