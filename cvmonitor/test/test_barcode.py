from flask import url_for
import cv2
import numpy as np
import imageio
import os

def test_ping(client):
    
    res = client.get(url_for('ping'))
    assert res.data == b'pong'

def test_cv_ping(client):
    res = client.get(url_for('cv.ping'))
    assert res.data == b'pong cv'

def test_codes(client):
    image = open(os.path.dirname(__file__)+'/data/barcode.png','rb').read()
    assert len(image)>0
    res = client.post(url_for('cv.detect_codes'),data=image,headers={'content-type':'application/png'})
    assert res.json[0]['data'] == 'Foramenifera'

def test_align(client):
    image = open(os.path.dirname(__file__)+'/data/qrcode.png','rb').read()
    assert len(image)>0
    res = client.post(url_for('cv.align_image'),data=image,headers={'content-type':'application/png'})
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0]>0


def test_align_whole_image(client):
    image = open(os.path.dirname(__file__)+'/data/barcode_monitor.jpg','rb').read()
    assert len(image)>0
    res = client.post(url_for('cv.align_image'),data=image,headers={'content-type':'application/png'})
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0]>0



