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
    score: float = 0.0
    source: Union[str, None] = None


class Codes(BaseModel):
    data: str
    top: int
    left: int
    bottom: int
    right: int
    code_type: str


class ScreenCorners(BaseModel):
    left_top: Tuple[int, int]
    right_top: Tuple[int, int]
    left_bottom: Tuple[int, int]
    right_bottom: Tuple[int, int]


class Device(BaseModel):
    deviceId: str
    refImageId: str
    refTimestamp: datetime
    patientId: str
    roomId: str
    deviceCategory: str
    screenCorners: ScreenCorners
    segments: List[Segment]


class DeviceRecord(BaseModel):
    imageId: str
    timestamp: str
    deviceId: str
    deviceCategory: str
    segments: List[Segment]
    image: bytes
