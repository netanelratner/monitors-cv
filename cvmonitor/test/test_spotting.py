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
    #prev_ val is just for the test
    expected_boxes = [
        {'max_length': len('93')+2, 'bbox': [209.857, 276.766, 247.619, 311.333], 'prev_val': '93'},
        {'max_length': len('15')+2, 'bbox':  [ 84.656, 277.909, 122.706, 310.801], 'prev_val': '15'},
        {'max_length': len('45')+2, 'bbox': [108.981, 333.588, 148.733, 367.113], 'prev_val': '45'},
        {'max_length': len('52')+2, 'bbox': [241.387, 226.571, 287.536, 267.902], 'prev_val': '52'},
        {'max_length': len('1151')+2, 'bbox':  [ 39.299, 334.754, 104.618, 367.895], 'prev_val': '1151'},
        {'max_length': len('1444')+2, 'bbox': [261.369, 212.533, 289.603, 222.765], 'prev_val': '1444'},
        {'max_length': len('64')+2, 'bbox': [144.722, 372.172, 158.052, 385.809], 'prev_val': '64'},
    ]
    for e in expected_boxes:
        # left, top, right, bottom
        e['bbox'][0]+=2
        e['bbox'][1]+-2
        e['bbox'][2]+=3
        e['bbox'][3]+=2

    texts, _, _, _ = model.forward(image, expected_boxes)
    for text, expected_boxe in zip(texts, expected_boxes):
        assert expected_boxe['prev_val']==text




if __name__ == "__main__":

    model = text_spotting.Model(max_seq_len=6)

    test_expected_boxes(model)
    # test_run_one_image(model())