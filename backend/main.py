from datetime import datetime
import pickle
import os

import aioredis
from fastapi import FastAPI, Header

from .backend_data import MonitorDataPost, MonitorDataGetResponse, Monitor, MonitorImagePostResponse

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    app.state.redis = await aioredis.create_redis_pool(os.environ.get("CVMONITOR_BACKEND_REDIS_URL", "redis://localhost"))


@app.post("/monitor_data/{monitorId}")
async def monitor_data_post(monitorId: str, monitorData: MonitorDataPost):
    mainKey = f"monitor_data:monitorId_imageId_timestamp:{monitorData.monitorId}_{monitorData.imageId}_{monitorData.timestamp}"
    # TODO pipeline
    await app.state.redis.set(mainKey, pickle.dumps(dict(monitorData)))
    await app.state.redis.set(f"monitor_data_index:monitorId:{monitorData.monitorId}", mainKey)
    await app.state.redis.set(f"monitor_data_index:monitorId_imageId:{monitorData.monitorId}_{monitorData.imageId}", mainKey)
    # TODO index timestamp


@app.get("/monitor_data/{monitorId}", response_model=MonitorDataGetResponse)
async def monitor_data_get(monitorId: str, x_image_id: str = Header(None)):
    if x_image_id:
        key = await app.state.redis.get(f"monitor_data_index:monitorId_imageId:{monitorId}_{x_image_id}")
    else:
        key = await app.state.redis.get(f"monitor_data_index:monitorId:{monitorId}")
    if key:
        data = await app.state.redis.get(key)
        data = pickle.loads(data)
        return data
    # TODO how to return error ?


@app.post("/monitor/{monitorId}")
async def monitor_post(monitorId: str, monitor: Monitor):
    pass


@app.get("/monitor/{monitorId}", response_model=Monitor)
async def monitor_get(monitorId: str):
    pass


@app.delete("/monitor/{monitorId}")
async def monitor_delete(monitorId: str):
    pass


@app.get("/monitor/list")
async def monitor_list_get():
    pass


@app.post("/monitor_image/{monitorId}", response_model=MonitorImagePostResponse)
async def monitor_image_post(monitorId: str, x_image_id: str = Header(None), x_monitor_id: str = Header(None), x_timestamp: datetime = Header(None)):
    pass


@app.get("/monitor_image/{monitorId}")
async def monitor_image_get(monitorId: str, x_image_id: str = Header(None)):
    pass
