from datetime import datetime
import pickle
import os

import aioredis
from fastapi import FastAPI, Request, Header

from .backend_data import MonitorDataPost, MonitorDataGetResponse, Monitor, MonitorImagePostResponse

app = FastAPI()


# TODO do pipeline ?
# TODO how to handle errors of non existent data ?


@app.on_event("startup")
async def on_startup():
    app.state.redis = await aioredis.create_redis_pool(os.environ.get("CVMONITOR_BACKEND_REDIS_URL", "redis://localhost"))


@app.post("/monitor_data/{monitorId}")
async def monitor_data_post(monitorId: str, monitorData: MonitorDataPost):
    mainKey = f"monitor_data:monitorId_imageId:{monitorData.monitorId}_{monitorData.imageId}"
    await app.state.redis.set(mainKey, pickle.dumps(dict(monitorData)))
    #await app.state.redis.set(f"monitor_data_index:monitorId:{monitorData.monitorId}", mainKey)
    #await app.state.redis.set(f"monitor_data_index:monitorId_imageId:{monitorData.monitorId}_{monitorData.imageId}", mainKey)
    # TODO index timestamp ?


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


@app.post("/monitor/{monitorId}")
async def monitor_post(monitorId: str, monitor: Monitor):
    mainKey = f"monitor:monitorId"
    pass


@app.get("/monitor/{monitorId}", response_model=Monitor)
async def monitor_get(monitorId: str):
    data = await app.state.redis.get(f"monitor:monitorId:{monitorId}")
    if data:
        data = pickle.loads(data)
        return data


@app.delete("/monitor/{monitorId}")
async def monitor_delete(monitorId: str):
    pass


@app.get("/monitor/list")
async def monitor_list_get():
    data = await app.state.redis.smembers('monitor')
    return data


@app.post("/monitor_image/{monitorId}", response_model=MonitorImagePostResponse)
async def monitor_image_post(monitorId: str, request: Request, x_image_id: str = Header(None), x_monitor_id: str = Header(None), x_timestamp: datetime = Header(None)):
    image = await request.body()
    # TODO what to do with this crappy timestamp...
    key = f"monitor_data:monitorId_imageId:{monitorId}_{x_image_id}"
    await app.state.redis.set(key, image)


@app.get("/monitor_image/{monitorId}")
async def monitor_image_get(monitorId: str, x_image_id: str = Header(None)):
    pass
