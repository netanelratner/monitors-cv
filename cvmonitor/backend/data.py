from pydantic import BaseModel
from typing import List, Union, Tuple, Optional
from datetime import datetime


class Codes(BaseModel):
    data: str
    top: int
    left: int
    bottom: int
    right: int
    code_type: str


class Segment(BaseModel):
    top: int
    left: int
    bottom: int
    right: int
    name: Optional[str] = None
    value: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


class ScreenCorners(BaseModel):
    left_top: Tuple[int, int]
    right_top: Tuple[int, int]
    left_bottom: Tuple[int, int]
    right_bottom: Tuple[int, int]


class Monitor(BaseModel):
    monitorId: str
    imageId: Optional[str] = None
    timestamp: Optional[datetime] = None
    patientId: Optional[str] = None
    roomId: Optional[str] = None

    deviceCategory: str
    screenCorners: ScreenCorners
    segments: Optional[List[Segment]] = None

class Device(Monitor):
    pass

class DeviceRecord(BaseModel):
    imageId: str
    timestamp: datetime
    monitorId: str
    deviceCategory: str
    segments: Optional[List[Segment]] = None
    image: bytes


class MonitorDataGetResponse(BaseModel):
    imageId: str
    monitorId: str
    timestamp: datetime
    patientId: Union[str, None] = None
    roomId: Union[str, None] = None
    deviceCategory: str
    segments: List[Segment]

class MonitorDataPost(BaseModel):
    imageId: str
    monitorId: str
    timestamp: datetime
    segments: List[Segment]

class MonitorImagePostResponse(BaseModel):
    frameRate: int
    monitorId: str
    nextImage: str
    minResolutionWidth: int
    minResolutionHeight: int
    