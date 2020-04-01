#! /bin/env python
import imageio
import os
import pytest
from cvmonitor.ocr.text_spotting import text_spotting
import sys
import logging


@pytest.fixture(scope='module')
def model():
    assert 'gevent' not in sys.modules.keys()
    init_logs()
    return text_spotting.Model(max_seq_len=6)


def init_logs():
    for logger in (
        
        logging.getLogger(),
    ):
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        sh  = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def test_simple_runt(model):
    
    image = imageio.imread(os.path.dirname(__file__)+'/data/11.jpg')
    res = model.forward(image)

def test_expected_boxes(model):
    
    image = imageio.imread(os.path.dirname(__file__)+'/data/11.jpg')
    expected_boxes = [
        {'max_length': len('93')+2, 'bbox': [209.857, 276.766, 247.619, 311.333]},
        {'max_length': len('15')+2, 'bbox':  [ 84.656, 277.909, 122.706, 310.801]},
        {'max_length': len('45')+2, 'bbox': [108.981, 333.588, 148.733, 367.113]},
        {'max_length': len('52')+2, 'bbox': [241.387, 226.571, 287.536, 267.902]},
        {'max_length': len('1151')+2, 'bbox':  [ 39.299, 334.754, 104.618, 367.895]},
        {'max_length': len('1444')+2, 'bbox': [261.369, 212.533, 289.603, 222.765]},
        {'max_length': len('64')+2, 'bbox': [144.722, 372.172, 158.052, 385.809]},
    ]
    # for e in expected_boxes:
    #     # left, top, right, bottom
    #     e['box'][0]+=5
    #     e['box'][1]+-5
    #     e['box'][2]+=5
    #     e['box'][3]+=7
    res = model.forward(image, expected_boxes)




if __name__ == "__main__":
    test_run_one_image(model())