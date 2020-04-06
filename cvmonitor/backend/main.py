from datetime import datetime
import io
import pickle
import os
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from typing import List, Union
from datetime import datetime
import redis
from fastapi import FastAPI, Request, Header, File

from .data import (
    MonitorDataPost,
    MonitorDataGetResponse,
    Monitor,
    MonitorImagePostResponse,
)
from . import data as Data


def add_device(redis: redis.StrictRedis, device: BaseModel):
    device_id = device.monitorId
    device_key = f"device:{device_id}"
    for k, v in device.items():
        redis.hset(device_key, k, pickle.dumps(v, device_id))
    redis.sadd("devices", device_key)


def get_device(redis: redis.StrictRedis, device_id: str) -> Data.Device:
    device_key = f"device:{device_id}"
    data = redis.hgetall(device_key)
    device_dict = {}
    for k  in data.keys():
        device_dict[k.decode()] = pickle.loads(data[k])
    if device_dict:
        return Data.Device.parse_obj(device_dict)


def add_device_data(
    redis: redis.StrictRedis, record_part: Data.DeviceRecord, expiration
):
    device_id = record_part.monitorId
    key = f"device_record:{device_id}_{record_part.imageId}"
    for k, v in record_part.dict().items():
        redis.hset(key, k, pickle.dumps(v))

    records_key = f"device_records:{device_id}"
    redis.expire(key, expiration)
    redis.zadd(records_key, {key: record_part.timestamp.timestamp()})


def get_device_data(redis: redis.StrictRedis, device_id: str, image_id=None):
    
    device_key = f"device:{device_id}"
    records_key = f"device_records:{device_id}"
    if image_id is None:
        timestamp = datetime.now().timestamp()
        image_record = redis.zrevrangebyscore(records_key, timestamp, 0, start=0,num=1)[0].decode()
    else:
        image_record = f"device_record:{device_id}_{image_id}"
    if image_record is None:
        return None
    record_data = redis.hgetall(image_record)
    record_dict = {}
    for k  in record_data.keys():
        record_dict[k.decode()] = pickle.loads(record_data[k])
    
    return Data.DeviceRecord.parse_obj(record_dict)


def remove_device_data(redis: redis.StrictRedis, device_id: str, image_id: str):
    key = f"device_record:{device_id}_{image_id}"
    redis.hdel(key)


def remove_device(redis: redis.StrictRedis, device_id: str):
    device_key = f"device:{device_id}"
    records_key = f"device_records:{device_id}"
    redis.delete(device_key)
    while True:
        item = redis.lpop(records_key)
        if not item:
            break
        redis.hdel(item)
    redis.zrem("devices", device_key)


class Server:
    def __init__(self):
        self.app = FastAPI()
        self.record_expiration = int(
                os.environ.get("CVMONITOR_DATA_EXPIRATION", 60 * 60 * 10)
            )
            
        @self.app.on_event("startup")
        def on_startup():
            self.app.state.redis = redis.Redis(
                os.environ.get("CVMONITOR_BACKEND_REDIS_URL", "localhost"),
                int(os.environ.get("CVMONITOR_BACKEND_REDIS_PORT", "6379")),
            )

        @self.app.post("/monitor_data/{monitorId}")
        def monitor_data_post(monitorId: str, monitorData: MonitorDataPost):
            add_device_data(self.app.state.redis, monitorData, self.record_expiration)

        @self.app.get(
            "/monitor_data/{monitorId}", response_model=MonitorDataGetResponse
        )
        def monitor_data_get(monitorId: str, x_image_id: str = Header(None)):
            device =  get_device(self.app.state.redis, monitorId)
            device_record = get_device_data(self.app.state.redis, monitorId, x_image_id)
            result = {}
            result.update(device)
            # This should be second to override image id, timestamp etc.
            result.update(device_record)
            return result

        @self.app.post("/monitor/{monitorId}", status_code=200)
        def monitor_post(monitorId: str, device: Data.Device):
            add_device(self.app.state.redis, device)

        @self.app.get("/monitor/{monitorId}", response_model=Data.Device)
        def monitor_get(monitorId: str):
            return get_device(self.app.state.redis, monitorId)

        @self.app.delete("/monitor/{monitorId}")
        def monitor_delete(monitorId: str):
            remove_device(self.app.state.redis, monitorId)

        @self.app.get("/monitor/list")
        def monitor_list_get(response_model=List[str]):
            return self.app.state.redis.hgetall("devices")

        @self.app.post(
            "/monitor_image", response_model=MonitorImagePostResponse
        )
        def monitor_image_post(
            image: bytes = File(None),
            x_image_id: str = Header(None),
            x_monitor_id: str = Header(None),
            x_timestamp: datetime = Header(None),
        ):
            data = Data.DeviceRecord(
                imageId=x_image_id,
                timestamp=x_timestamp,
                monitorId=x_monitor_id,
                image=image,
            )
            add_device_data(self.app.state.redis, data, self.record_expiration)
            return MonitorImagePostResponse(
                frameRate=0.2,
                monitorId=x_monitor_id,
                nextImageId=int(x_image_id) + 1,
                minResolutionWidth=0,
                minResolutionHeight=0,
            )

        @self.app.get("/monitor_image/{monitorId}")
        def monitor_image_get(monitorId: str, x_image_id: str = Header(None)):
            device_data = get_device_data(self.app.state.redis, monitorId, x_image_id)
            return StreamingResponse(
                io.BytesIO(device_data.image), media_type="image/jpeg"
            )
