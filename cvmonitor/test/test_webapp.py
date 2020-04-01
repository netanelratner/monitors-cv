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
import pylab

def test_ping(client):

    res = client.get(url_for('ping'))
    assert res.data == b'pong'


def test_cv_ping(client):
    res = client.get(url_for('cv.ping'))
    assert res.data == b'pong cv'


def test_codes(client):
    image = open(os.path.dirname(__file__)+'/data/barcode.png', 'rb').read()
    assert len(image) > 0
    res = client.post(url_for('cv.detect_codes'), data=image, headers={'content-type': 'application/png'})
    assert res.json[0]['data'] == 'Foramenifera'


def test_align(client):
    image = open(os.path.dirname(__file__)+'/data/qrcode.png', 'rb').read()
    assert len(image) > 0
    os.environ['CVMONITOR_QR_PREFIX'] = 'http'
    res = client.post(url_for('cv.align_image'), data=image, headers={'content-type': 'application/png'})
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_align_whole_image(client):
    image = open(os.path.dirname(__file__)+'/data/barcode_monitor.jpg', 'rb').read()
    assert len(image) > 0
    os.environ['CVMONITOR_QR_PREFIX'] = 'http'
    res = client.post(url_for('cv.align_image'), data=image, headers={'content-type': 'application/png'})
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_exif_align(client):
    src_image = open(os.path.dirname(__file__)+'/data/sample.jpeg', 'rb').read()
    assert len(src_image) > 0
    os.environ['CVMONITOR_QR_PREFIX'] = ''
    res = client.post(url_for('cv.align_image'), data=src_image, headers={'content-type': 'application/png'})
    res_image = np.asarray(imageio.imread(res.data))
    up_image = imageio.imread(os.path.dirname(__file__)+'/data/sample_up.jpg')
    assert np.median(np.abs(res_image-up_image)) < 2.0


def test_ocr(client):
    image = open(os.path.dirname(__file__)+'/data/11.jpg', 'rb').read()
    bbox_list = np.load(open(os.path.dirname(__file__)+'/data/11_recs.npy', 'rb'))
    image_buffer = base64.encodebytes(image).decode()
    data = {
        'image': image_buffer,
        'segments': [{'left': int(s[0]), 'top':int(s[1]), 'right':int(s[2]), 'bottom':int(s[3]), 'name':str(i)} for i, s in enumerate(bbox_list)]
    }
    res = client.post(url_for('cv.run_ocr'), json=data)
    assert res.json == [{'segment_name': '0', 'value': '52'}, {'segment_name': '1', 'value': '15'}, {
        'segment_name': '2', 'value': '93'}, {'segment_name': '3', 'value': '115'}, {'segment_name': '4', 'value': '45'}]


