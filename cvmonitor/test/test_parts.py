from flask import url_for
import cv2
import numpy as np
import imageio
import os
import ujson as json
import pytest
from pylab import imshow, show
from .. import cv, image_align
import pylab

def test_pdf_generate():
    cv.generate_pdf('test.pdf', 'something',4,6)


def test_find_qrcode():
    image = imageio.imread(os.path.dirname(__file__)+'/data/barcode_monitor.jpg')
    qrcode = cv.find_qrcode(image, '')
    assert qrcode.data.decode().startswith('http')


def test_align_image1():
    image = imageio.imread(os.path.dirname(__file__)+'/data/barcode_monitor.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_image2():
    image = imageio.imread(os.path.dirname(__file__)+'/data/test.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_rotate():
    image = imageio.imread(os.path.dirname(__file__)+'/data/rotated.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_flipped():
    image = imageio.imread(os.path.dirname(__file__)+'/data/flipped.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)

    assert warpped.shape[0] > 0


def test_align_90deg_large():
    image = imageio.imread(os.path.dirname(__file__)+'/data/90_deg_rotate.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_90deg_small():
    image = imageio.imread(os.path.dirname(__file__)+'/data/90_deg_rotate_small.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0

def test_align_another():
    image = imageio.imread(os.path.dirname(__file__)+'/data/another.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0

def test_printed_qr():
    image = imageio.imread(os.path.dirname(__file__)+'/data/printed.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    
    assert warpped.shape[0] > 0

def test_generated_qr():
    image = imageio.imread(os.path.dirname(__file__)+'/data/generated_from_pdf.jpg')
    qrcode = cv.find_qrcode(image, '')
    warpped, M = cv.align_by_qrcode(image, qrcode)
    assert 'dba7b418e0ef450c' in qrcode.data.decode()
    assert warpped.shape[0] > 0


def test_orient_by_qr():
    res = []
    rwarpped = []
    print()
    for i in [1,2,3,4]:
        im_file = open(os.path.dirname(__file__)+f'/data/bad{i}.jpg','rb')
        image, _, _ = image_align.get_oriented_image(im_file, use_qr=True)
        res.append(image)
        # Detect QR again, so it will be eaactly the same
        qrcode = cv.find_qrcode(image, '')
        print(qrcode)
        warpped, M = cv.align_by_qrcode(image, qrcode)
        rwarpped.append(warpped)
        # The original rotaion was lossy...
        assert np.median(res[0]-res[-1])<2.0
    