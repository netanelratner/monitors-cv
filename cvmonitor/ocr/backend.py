import redis
from dataclasses import dataclass
from typing import Union, List
import redis

@dataclass
class Segment:
    top: int
    left: int
    bottom: int
    right: int
    name: str
    value: str
    score: float
    source: Union[str,None]

@dataclass
class ScreenCorners:
    left_top: List[int,int]
    right_top: List[int,int]
    left_bottom: List[int,int]
    right_bottom: List[int,int]


@dataclass
class Device:
    deviceId: str
    refImageId: str
    refTimestamp: str # iso 8601 timestamp
    patientId: str
    roomId: str
    deviceCategory: str
    screenCorners: ScreenCorners
    segments: List[Segment]

@dataclass
class DeviceRecord:
    imageId: str
    timestamp: str
    deviceId: str
    segments: List[Segment]
    image: bytes




def update_device_record(record: DeviceRecord, ocr_results: List[Segment]) -> DeviceRecord:
    indexes = {segment['name']: index for index, segment in  enumerate(ocr_results)}
    for segment in record.segments:
        if segment['name'] in indexes:
            res = ocr_results[segment['name']]
            segment['value'] = res['value']
            segment['source'] = res['source']
            segment['score'] = res['score']


def 