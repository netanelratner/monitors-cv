from flask import url_for
import cv2
import numpy as np
import imageio
import os
import base64
import ujson as json
import pytest
from pylab import imshow, show
from .. import cv, image_align
from cvmonitor.ocr.text_spotting import text_spotting
from cvmonitor.ocr.utils import get_fields_info, is_text_valid
import pylab


def test_ping(client):

    res = client.get(url_for("ping"))
    assert res.data == b"pong"


def test_cv_ping(client):
    res = client.get(url_for("cv.ping"))
    assert res.data == b"pong cv"


def test_codes(client):
    image = open(os.path.dirname(__file__) + "/data/barcode.png", "rb").read()
    assert len(image) > 0
    res = client.post(
        url_for("cv.detect_codes"),
        data=image,
        headers={"content-type": "application/png"},
    )
    assert res.json[0]["data"] == "Foramenifera"


def test_align(client):
    image = open(os.path.dirname(__file__) + "/data/qrcode.png", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = client.post(
        url_for("cv.align_image"),
        data=image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_align_whole_image(client):
    image = open(os.path.dirname(__file__) + "/data/barcode_monitor.jpg", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = client.post(
        url_for("cv.align_image"),
        data=image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_exif_align(client):
    src_image = open(os.path.dirname(__file__) + "/data/sample.jpeg", "rb").read()
    assert len(src_image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = ""
    res = client.post(
        url_for("cv.align_image"),
        data=src_image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    up_image = imageio.imread(os.path.dirname(__file__) + "/data/sample_up.jpg")
    assert np.median(np.abs(res_image - up_image)) < 2.0


def test_ocr_with_segments(client):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image_buffer = base64.encodebytes(image).decode()
    segments = []
    devices_names = [
        'HR','RR','SpO2','IBP-Systole','IBP-Diastole'
    ]
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
    data = {"image": image_buffer, "segments": segments}
    res = client.post(url_for("cv.run_ocr"), json=data)
    results = res.json
    assert res.json
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


def test_ocr_no_segments_suggest(client):
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    image_buffer = base64.encodebytes(image).decode()
    data = {
        "image": image_buffer,
    }
    os.environ['CVMONITOR_SUGGEST_SEGMENT']='TRUE'
    res = client.post(url_for("cv.run_ocr"), json=data)
    segments = res.json

    devices_names = [
        'HR','RR','SpO2','IBP-Systole','IBP-Diastole'
    ]
    expected = [
        {"name": devices_names[0], "value": "52"},
        {"name": devices_names[1], "value": "15"},
        {"name": devices_names[2], "value": "93"},
        {"name": devices_names[3], "value": "115"},
        {"name": devices_names[4], "value": "45"},
    ]

    data = {"image": image_buffer, "segments": segments}


    box_res = [[s["left"], s["top"], s["right"], s["bottom"]] for s in segments]
    box_expected = bbox_list
    best_matches, _, _, _ = text_spotting.match_boxes(box_res, box_expected)
    for i in range(len(best_matches)):
        assert text_spotting.iou(box_res[i], box_expected[best_matches[i]]) > 0.75
        assert expected[best_matches[i]]["value"] in segments[i]["value"]
    os.environ['CVMONITOR_SUGGEST_SEGMENT']='FALSE'


def test_ocr_no_segments_no_suggest(client):
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    image_buffer = base64.encodebytes(image).decode()
    data = {
        "image": image_buffer,
    }
    os.environ['CVMONITOR_SUGGEST_SEGMENT']='FALSE'
    res = client.post(url_for("cv.run_ocr"), json=data)
    segments = res.json
    assert not segments

def test_bad_bb(client):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    image_buffer = base64.encodebytes(image).decode()
    data = {
        "image": image_buffer,
        "segments": [{"left": 0, "top": 0, "right": 1, "bottom": 1, "name": "RR"}],
    }
    res = client.post(url_for("cv.run_ocr"), json=data)
    result = res.json
    assert {"name": "RR", "value": None}.items() <= result[0].items()


def test_ocr_with_partial_segments(client):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image_buffer = base64.encodebytes(image).decode()
    devices_names = [
        'HR','RR','SpO2','IBP-Systole','IBP-Diastole'
    ]
    data = {
        "image": image_buffer,
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
    res = client.post(url_for("cv.run_ocr"), json=data)
    for r, e in zip(
        res.json,
        [
            {"name": devices_names[0], "value": "52"},
            {"name": devices_names[1], "value": "15"},
            {"name": devices_names[2], "value": "93"},
        ],
    ):
        assert e.items() <= r.items()


def test_show_ocr_with_segments(client):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))
    image_buffer = base64.encodebytes(image).decode()
    segments = []
    devices_names = [
        'HR','RR','SpO2','IBP-Systole','IBP-Diastole'
    ]
    for i, b in enumerate(bbox_list):
        segments.append(
            {
                "left": float(b[0]),
                "top": float(b[1]) +0.1,
                "right": int(b[2]),
                "bottom": int(b[3]),
                "name": str(devices_names[i]),
                'value': 10
            }
        )
    data = {"image": image_buffer, "segments": segments}
    res = client.post(url_for("cv.show_ocr"), json=data)
    image_res = imageio.imread(res.data)
    assert imageio.imread(image).shape == image_res.shape


def test_get_measurements(client):
    assert 'HR' in client.get(url_for("cv.get_measurements",device='monitor')).json
    assert not client.get(url_for("cv.get_measurements", device='unknown')).json
    assert 'Rate' in client.get(url_for("cv.get_measurements", device='respirator')).json