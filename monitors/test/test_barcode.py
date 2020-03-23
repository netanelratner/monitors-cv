from flask import url_for
import cv2
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
    import pdb; pdb.set_trace()
