import cv2
import numpy as np
import imageio
import os
import base64
import ujson as json
import pytest
from pylab import imshow, show
import pylab
from datetime import datetime
from cvmonitor.ocr.text_spotting import text_spotting
from cvmonitor.ocr.utils import get_fields_info, is_text_valid
from cvmonitor.cv import image_align
from cvmonitor.cv.cv import ComputerVision
from cvmonitor.backend import data as Data


@pytest.fixture(scope="module")
def cv():
    return ComputerVision()


def test_codes(cv):
    image = open(os.path.dirname(__file__) + "/data/barcode.png", "rb").read()
    res = cv.detect_codes(image)
    assert res[0].data == "Foramenifera"


def test_align(cv):
    image = open(os.path.dirname(__file__) + "/data/qrcode.png", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = cv.align_image(image)[0]
    res_image = np.asarray(imageio.imread(res))
    assert res_image.shape[0] > 0


def test_align_whole_image(cv):
    image = open(os.path.dirname(__file__) + "/data/barcode_monitor.jpg", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = cv.align_image(image)[0]
    res_image = np.asarray(imageio.imread(res))
    assert res_image.shape[0] > 0


def test_exif_align(cv):
    src_image = open(os.path.dirname(__file__) + "/data/sample.jpeg", "rb").read()
    assert len(src_image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = ""
    res = cv.align_image(src_image)[0]
    res_image = np.asarray(imageio.imread(res))
    up_image = imageio.imread(os.path.dirname(__file__) + "/data/sample_up.jpg")
    assert np.median(np.abs(res_image - up_image)) < 2.0


def test_ocr_with_segments(cv):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    segments = []
    devices_names = ["HR", "RR", "SpO2", "IBP-Systole", "IBP-Diastole"]
    for i, b in enumerate(bbox_list):
        segments.append(
            {
                "left": int(b[0]),
                "top": int(b[1]),
                "right": int(b[2]),
                "bottom": int(b[3]),
                "name": str(devices_names[i]),
            }
        )
    data = {
        "image": image,
        "segments": segments,
        "monitorId": "1231234",
        "imageId": "1111",
        "deviceCategory": "monitor",
        "timestamp": datetime.now(),
    }
    
    results = [x.dict() for x in cv.run_ocr(Data.DeviceRecord.parse_obj(data))]
    for r, e in zip(
        results,
        [
            {"name": devices_names[0], "value": "52"},
            {"name": devices_names[1], "value": "15"},
            {"name": devices_names[2], "value": "93"},
            {"name": devices_names[3], "value": "115"},
            {"name": devices_names[4], "value": "45"},
        ],
    ):
        assert e.items() <= r.items()


def test_ocr_no_segments(cv):
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    data = {
        "image": image,
        "segments": None,
        "monitorId": "1231234",
        "imageId": "1111",
        "deviceCategory": "monitor",
        "timestamp": datetime.now(),
    }
    segments = [x.dict() for x in cv.run_ocr(Data.DeviceRecord.parse_obj(data))]

    devices_names = ["HR", "RR", "SpO2", "IBP-Systole", "IBP-Diastole"]
    expected = [
        {"name": devices_names[0], "value": "52"},
        {"name": devices_names[1], "value": "15"},
        {"name": devices_names[2], "value": "93"},
        {"name": devices_names[3], "value": "115"},
        {"name": devices_names[4], "value": "45"},
    ]

    box_res = [[s["left"], s["top"], s["right"], s["bottom"]] for s in segments]
    box_expected = bbox_list
    best_matches, _ = text_spotting.match_boxes(box_res, box_expected)
    for i in range(len(best_matches)):
        assert text_spotting.iou(box_res[i], box_expected[best_matches[i]]) > 0.75
        assert expected[best_matches[i]]["value"] in segments[i]["value"]


def test_bad_bb(cv):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    data = {
        "image": image,
        "segments": [{"left": 0, "top": 0, "right": 1, "bottom": 1, "name": "RR"}],
        "monitorId": "1231234",
        "imageId": "1111",
        "deviceCategory": "monitor",
        "timestamp": datetime.now(),
    }
    result = [x.dict() for x in cv.run_ocr(Data.DeviceRecord.parse_obj(data))]
    assert {"name": "RR", "value": None}.items() <= result[0].items()


def test_ocr_with_partial_segments(cv):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    devices_names = ["HR", "RR", "SpO2", "IBP-Systole", "IBP-Diastole"]
    data = {
        "image": image,
        "monitorId": "1231234",
        "imageId": "1111",
        "deviceCategory": "monitor",
        "timestamp": datetime.now(),
        "segments": [
            {
                "left": int(s[0]),
                "top": int(s[1]),
                "right": int(s[2]),
                "bottom": int(s[3]),
                "name": devices_names[i],
            }
            for i, s in enumerate(bbox_list[:-2])
        ],
    }
    res = [x.dict() for x in cv.run_ocr(Data.DeviceRecord.parse_obj(data))]
    for r, e in zip(
        res,
        [
            {"name": devices_names[0], "value": "52"},
            {"name": devices_names[1], "value": "15"},
            {"name": devices_names[2], "value": "93"},
        ],
    ):
        assert e.items() <= r.items()


def test_show_ocr_with_segments(cv):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))

    segments = []
    devices_names = ["HR", "RR", "SpO2", "IBP-Systole", "IBP-Diastole"]
    for i, b in enumerate(bbox_list):
        segments.append(
            {
                "left": int(b[0]),
                "top": int(b[1]),
                "right": int(b[2]),
                "bottom": int(b[3]),
                "name": str(devices_names[i]),
                "value": 10,
            }
        )
    data = {
        "image": image,
        "segments": segments,
        "monitorId": "1231234",
        "imageId": "1111",
        "deviceCategory": "monitor",
        "timestamp": datetime.now(),
}
    res =cv.show_ocr(Data.DeviceRecord.parse_obj(data))
    image_res = imageio.imread(res)
    assert imageio.imread(image).shape == image_res.shape
