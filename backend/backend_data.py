from pydantic import BaseModel
from typing import List, Union, Tuple
from datetime import datetime


class Segment(BaseModel):
    top: int
    left: int
    bottom: int
    right: int
    name: str
    value: Union[str, None] = None
    source: str
    score: Union[float, None] = None


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


class ScreenCorners(BaseModel):
    left_top: Tuple[int, int]
    right_top: Tuple[int, int]
    left_bottom: Tuple[int, int]
    right_bottom: Tuple[int, int]


class Monitor(BaseModel):
    monitorId: Union[str, None] = None
    imageId: Union[str, None] = None
    patientId: Union[str, None] = None
    roomId: Union[str, None] = None
    deviceCategory: str
    screenCorners: ScreenCorners
    segments: List[Segment]


class MonitorImagePostResponse(BaseModel):
    frameRate: int
    monitorId: str
    nextImage: str
    minResolutionWidth: int
    minResolutionHeight: int
    