from flask import url_for
import cv2
import numpy as np
import imageio
import os
import ujson as json
import pytest
from pylab import imshow, show
from cvmonitor.ocr.utils import get_fields_info, is_text_valid, add_decimal_notation
from ..ocr.utils import get_ocr_expected_boxes
from .. import ocr
from ..qr import find_qrcode
from ..image_align import align_by_qrcode
from .. import qr, image_align
import pylab


def test_pdf_generate():
    qr.generate_pdf("test.pdf", "something", 4, 6)


def test_find_qrcode():
    image = imageio.imread(os.path.dirname(__file__) + "/data/barcode_monitor.jpg")
    qrcode = qr.find_qrcode(image, "")
    assert qrcode.data.decode().startswith("http")


def test_align_image1():
    image = imageio.imread(os.path.dirname(__file__) + "/data/barcode_monitor.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_image2():
    image = imageio.imread(os.path.dirname(__file__) + "/data/test.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_rotate():
    image = imageio.imread(os.path.dirname(__file__) + "/data/rotated.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_flipped():
    image = imageio.imread(os.path.dirname(__file__) + "/data/flipped.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)

    assert warpped.shape[0] > 0


def test_align_90deg_large():
    image = imageio.imread(os.path.dirname(__file__) + "/data/90_deg_rotate.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_90deg_small():
    image = imageio.imread(os.path.dirname(__file__) + "/data/90_deg_rotate_small.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_another():
    image = imageio.imread(os.path.dirname(__file__) + "/data/another.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_printed_qr():
    image = imageio.imread(os.path.dirname(__file__) + "/data/printed.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)

    assert warpped.shape[0] > 0


def test_generated_qr():
    image = imageio.imread(os.path.dirname(__file__) + "/data/generated_from_pdf.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert "dba7b418e0ef450c" in qrcode.data.decode()
    assert warpped.shape[0] > 0


def test_orient_by_qr():
    res = []
    rwarpped = []
    print()
    for i in [1, 2, 3, 4]:
        im_file = open(os.path.dirname(__file__) + f"/data/bad{i}.jpg", "rb")
        image, _, _ = image_align.get_oriented_image(im_file, use_qr=True)
        res.append(image)
        # Detect QR again, so it will be eaactly the same
        qrcode = find_qrcode(image, "")
        print(qrcode)
        warpped, M = align_by_qrcode(image, qrcode)
        rwarpped.append(warpped)
        # The original rotaion was lossy...
        assert np.median(res[0] - res[-1]) < 2.0


def test_ocr_by_segments():
    image = imageio.imread(
        os.path.dirname(__file__) + "/data/cvmonitors-cvmonitor-9a6d4db29fa04bfa.jpg"
    )
    segments = json.load(
        open(
            os.path.dirname(__file__)
            + "/data/cvmonitors-cvmonitor-9a6d4db29fa04bfa.json"
        )
    )
    devices = get_fields_info()
    text_spotting = ocr.text_spotting.text_spotting.Model()
    expected_boxes = get_ocr_expected_boxes(segments["segments"], devices, 0.5, 0.6)
    texts, boxes, scores, _ = text_spotting.forward(
        image, expected_boxes=expected_boxes
    )


def test_is_text_valid_str():
    devices = get_fields_info()
    assert is_text_valid("simv+", devices["Medication Name"])


def test_is_text_valid_int():
    devices = get_fields_info()
    assert is_text_valid("100", devices["HR"])


def test_is_text_valid_float():
    devices = get_fields_info()
    assert is_text_valid(add_decimal_notation("37.8", devices["Temp"]), devices["Temp"])
    assert is_text_valid(add_decimal_notation("378", devices["Temp"]), devices["Temp"])
    assert is_text_valid(add_decimal_notation("370", devices["Temp"]), devices["Temp"])

