from pydantic import BaseModel
from typing import List, Union, Tuple, Optional
from datetime import datetime


class Segment(BaseModel):
    top: int
    left: int
    bottom: int
    right: int
    name: Optional[str] = None
    value: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


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
    monitorId: str
    imageId: Optional[str] = None
    timestamp: Optional[datetime] = None
    patientId: Optional[str] = None
    roomId: Optional[str] = None
    deviceCategory: str
    screenCorners: ScreenCorners
    segments: Optional[List[Segment]] = None


class DeviceRecord(BaseModel):
    imageId: str
    timestamp: datetime
    monitorId: str
    deviceCategory: str
    segments: Optional[List[Segment]] = None
    image: bytes
