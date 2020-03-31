#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function

import logging as log
import os
import sys
import time
import pylab
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from openvino.inference_engine import IECore

from .tracker import StaticIOUTracker
from .visualizer import Visualizer

from .. import get_models
SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask

class Model():

    def __init__(self, device='CPU', track=False, visualize=False, prob_threshold=0.5):
        mask_rcnn_model_xml = get_models()['FP32/text-spotting-0001-detector.xml']
        mask_rcnn_model_bin = get_models()['FP32/text-spotting-0001-detector.bin']
        
        
        text_enc_model_xml = get_models()['FP32/text-spotting-0001-recognizer-encoder.xml']
        text_enc_model_bin = get_models()['FP32/text-spotting-0001-recognizer-encoder.bin']
        
        
        text_dec_model_xml = get_models()['FP32/text-spotting-0001-recognizer-decoder.xml']
        text_dec_model_bin = get_models()['FP32/text-spotting-0001-recognizer-decoder.bin']
        
        # Plugin initialization for specified device and load extensions library if specified.
        log.info('Creating Inference Engine...')
        ie = IECore()
        # Read IR
        log.info('Loading network files:\n\t{}\n\t{}'.format(mask_rcnn_model_xml, mask_rcnn_model_bin))
        mask_rcnn_net = IENetwork(model=mask_rcnn_model_xml, weights=mask_rcnn_model_bin)

        log.info('Loading network files:\n\t{}\n\t{}'.format(text_enc_model_xml, text_enc_model_bin))
        text_enc_net = IENetwork(model=text_enc_model_xml, weights=text_enc_model_bin)

        log.info('Loading network files:\n\t{}\n\t{}'.format(text_dec_model_xml, text_dec_model_bin))
        text_dec_net = IENetwork(model=text_dec_model_xml, weights=text_dec_model_bin)

        supported_layers = ie.query_network(mask_rcnn_net, 'CPU')
        not_supported_layers = [l for l in mask_rcnn_net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error('Following layers are not supported by the plugin for specified device {}:\n {}'.
                        format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
            sys.exit(1)

        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(mask_rcnn_net.inputs.keys()), \
            'Demo supports only topologies with the following input keys: {}'.format(', '.join(required_input_keys))
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks', 'text_features'}
        assert required_output_keys.issubset(mask_rcnn_net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        n, c, h, w = mask_rcnn_net.inputs['im_data'].shape
        assert n == 1, 'Only batch 1 is supported by the demo application'
        self.shape = [n, c, h, w]  
        log.info('Loading IR to the plugin...')
        self.mask_rcnn_exec_net = ie.load_network(network=mask_rcnn_net, device_name=device, num_requests=2)
        self.text_enc_exec_net = ie.load_network(network=text_enc_net, device_name=device)
        self.text_dec_exec_net = ie.load_network(network=text_dec_net, device_name=device)
        self.hidden_shape = text_dec_net.inputs['prev_hidden'].shape
        self.prob_threshold = prob_threshold
        self.tracker = None
        if track:
            self.tracker = StaticIOUTracker()
        if visualize:
            self.visualizer = Visualizer(['__background__', 'text'], show_boxes=True, show_scores=True)
    
    def forward(self, frame):
        [n, c, h, w] = self.shape
        # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
        scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
        input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                            (0, w - input_image_size[1]),
                                            (0, 0)),
                                mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)

        # Run the net.
        inf_start = time.time()
        outputs = self.mask_rcnn_exec_net.infer({'im_data': input_image, 'im_info': input_image_info})

        # Parse detection results of the current request
        boxes = outputs['boxes']
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        raw_masks = outputs['raw_masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > self.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y


        texts = []
        alphabet = '  0123456789abcdefghijklmnopqrstuvwxyz'
        for feature in text_features:
            feature = self.text_enc_exec_net.infer({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for i in range(MAX_SEQ_LEN):
                decoder_output = self.text_dec_exec_net.infer({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature})
                symbols_distr = decoder_output['output']
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += alphabet[prev_symbol_index]
                hidden = decoder_output['hidden']

            texts.append(text)

        inf_end = time.time()
        inf_time = inf_end - inf_start
        # performance stats.
        log.debug('Inference and post-processing time: {:.3f} ms'.format(inf_time * 1000))


        if self.tracker is not None or self.visualizer is not None:
            render_start = time.time()
            masks = []
            for box, cls, raw_mask in zip(boxes, classes, raw_masks):
                raw_cls_mask = raw_mask[cls, ...]
                mask = segm_postprocess(box, raw_cls_mask, frame.shape[0], frame.shape[1])
                masks.append(mask)

            if len(boxes):
                log.debug('Detected boxes:')
                log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
                for box, cls, score, mask in zip(boxes, classes, scores, masks):
                    log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

            # Get instance track IDs.
            masks_tracks_ids = None
            if self.tracker is not None:
                masks_tracks_ids = self.tracker(masks, classes)

            if self.visualizer is not None:
                # Visualize masks.
                frame = self.visualizer(frame, boxes, classes, scores, masks, texts, masks_tracks_ids)
            render_end = time.time()
            render_time = render_end - render_start
            log.debug('OpenCV rendering time: {:.3f} ms'.format(render_time * 1000))



        return texts, boxes, scores, frame