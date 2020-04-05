import pickle
import os

import aioredis
from fastapi import FastAPI

from .data import ScreenCorners, Segment, Device, DeviceRecord

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    app.state.redis = await aioredis.create_redis_pool(os.environ.get("BACKEND_REDIS_URL", "redis://localhost"))


@app.post("/monitor_data")
async def monitor_data_post(monitorData: DeviceRecord):
    await app.state.redis.set("monitor_data:" + monitorData.deviceId, pickle.dumps(monitorData))


@app.get("/monitor_data/{monitorId}")
async def monitor_data_get(monitorId: str):
    res = await app.state.redis.get("monitor_data:" + monitorId)
    if res:
        return pickle.loads(res)
    return None
