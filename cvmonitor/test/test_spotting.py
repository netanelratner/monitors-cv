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
    # true_val is just for the test
    expected_boxes = [
        {'name': 'NIBP-Diastole', 'max_len': len('93')+2, 'bbox': [209.857, 276.766, 247.619, 311.333], 'true_val': '93'},
        {'name': 'RR', 'max_len': len('15')+2, 'bbox':  [ 84.656, 277.909, 122.706, 310.801], 'true_val': '15'},
        {'name': 'Peep', 'max_len': len('45')+2, 'bbox': [108.981, 333.588, 148.733, 367.113], 'true_val': '45'},
        {'name': 'FIO2', 'max_len': len('52')+2, 'bbox': [241.387, 226.571, 287.536, 267.902], 'true_val': '52'},
        {'name': 'SpO2', 'max_len': len('115')+2, 'bbox':  [ 39.299, 334.754, 104.618, 367.895], 'true_val': '115'},
        {'name': 'Expiratory Tidal Volume', 'max_len': len('1444')+2, 'bbox': [261.369, 212.533, 289.603, 222.765], 'true_val': '1444'},
        {'name': 'I:E Ratio', 'max_len': len('64')+2, 'bbox': [144.722, 372.172, 158.052, 385.809], 'true_val': '64'},
    ]
    for e in expected_boxes:
        # left, top, right, bottom
        e['bbox'][0]+=2
        e['bbox'][1]+-2
        e['bbox'][2]+=3
        e['bbox'][3]+=2

    texts, _, _, _ = model.forward(image, expected_boxes)
    for text, expected_box in zip(texts[:-2], expected_boxes[:-2]):
        assert expected_box['true_val']==text




if __name__ == "__main__":

    model = text_spotting.Model(max_seq_len=6)

    test_expected_boxes(model)
    # test_run_one_image(model())