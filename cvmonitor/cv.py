from flask import Flask, Blueprint, request, abort, Response
import math
import cv2
import logging
import imageio
import ujson as json
import numpy as np
from pyzbar import pyzbar
import io
import os
import base64
import time
import qrcode
from uuid import uuid4
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .ocr import monitor_ocr
from prometheus_client import Summary
import pytesseract
from pylab import imshow, show
from .qr import generate_pdf, find_qrcode, read_codes
from image_align import get_oriented_image, align_by_qrcode
np.set_printoptions(precision=3)

class ComputerVision:

    def __init__(self):
        self.blueprint = Blueprint('cv', __name__)
        self.qrDecoder = cv2.QRCodeDetector()
        self.model_ocr = None

        @self.blueprint.route('/ping/')
        def ping():
            return 'pong cv'

        @self.blueprint.route('/detect_codes', methods=['POST'])
        def detect_codes():
            """
            Get QR or barcodes in an image
            ---
            description: Get QR codes and barcodes in an image
            requestBody:
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary
            responses:
             '200':
                dsecription: array of detections
                content:
                    application/json:
                        schema:
                            type: array
                            items: 
                                type: object
                                properties:
                                    data:
                                        type: string
                                    top:
                                        type: number
                                    left:
                                        type: number
                                    bottom:
                                        type: number
                                    right:
                                        type: number
            """
            image = np.asarray(imageio.imread(request.data))
            codes = []
            decodedObjects = pyzbar.decode(image)
            codes = read_codes(image)
            return json.dumps(codes), 200, {'content-type':'application/json'}

        @self.blueprint.route('/align_image', methods=['POST'])
        def align_image():
            """
            Given a jpeg image with that containes the  QR code, use that QR code to align the image
            ---
            description: Gets a jpeg and returns a jpeg
            requestBody:
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary
            responses:
              '200':
                descritption: jpeg image
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary

            """
            imdata= io.BytesIO(request.data())
                    
            imdata.seek(0)
            
            image = np.asarray(imageio.imread(request.data))

            if os.environ.get('CVMONITOR_SAVE_BEFORE_ALIGN')=='TRUE':
                imageio.imwrite('original_image.jpg',image)
            qrprefix = str(os.environ.get('CVMONITOR_QR_PREFIX','cvmonitor'))
            qrsize = int(os.environ.get('CVMONITOR_QR_TARGET_SIZE',100))
            boundery = float(os.environ.get('CVMONITOR_QR_BOUNDERY_SIZE',50))
            detected_qrcode = find_qrcode(image, qrprefix)
            if detected_qrcode is None:
                abort(400, "Could not find the qr code to aling the image")
            data = detected_qrcode.data.decode()
            image = rotate_by_qr_code(image, detected_qrcode)
            detected_qrcode = find_qrcode(image, qrprefix)
            if not (os.environ.get('CVMONITOR_SKIP_ALIGN')=='TRUE'):
                image, _ = align_by_qrcode(image, detected_qrcode, qrsize=qrsize, boundery = boundery)
            if os.environ.get('CVMONITOR_SAVE_AFTER_ALIGN')=='TRUE':
                imageio.imwrite('aligned_image.jpg',image)
            b = io.BytesIO()
            imageio.imwrite(b, image, format='jpeg')
            b.seek(0)
            return b.read(), 200, {'content-type':'image/jpeg','X-MONITOR-ID': data}

        @self.blueprint.route('/run_ocr', methods=['POST'])
        def run_ocr():
            """
            Run ocr on an image
            ---
            description: run ocr on image
            requestBody:
                content:
                    application/json:
                      schema:
                        type: object
                        properties:
                            image:
                                type: string
                                contentEncoding: base64
                                contentMediaType: image/jpeg
                            segments:
                                type: array
                                items:
                                    type: object
                                    properties:
                                        top:
                                            type: number
                                            format: integer
                                        left:
                                            type: number
                                            format: integer
                                        bottom:
                                            type: number
                                            format: integer
                                        right:
                                            type: number
                                            format: integer
            responses:
              '200':
                descritption: ocr results
                content:
                    application/json:
                      schema:
                        type: array
                        items:
                            type: object
                            properties:
                                segment_name: 
                                    type: string
                                value:
                                    type: string
            """
            threshold = float(os.environ.get("CVMONITOR_OCR_THRESHOLD","0.2"))
            if not self.model_ocr:
                self.model_ocr = monitor_ocr.build_model()
            if not self.model_ocr:
                abort(500,'NN Model not found, could not run ocr')
            data = request.json
            assert 'image' in data
            image = np.asarray(imageio.imread(base64.decodebytes(data['image'].encode())))
            if not data.get('segments'):
                return json.dumps([]), 200, {'content-type':'application/json'}
            segments = data['segments']
            if len(segments) == 0:
                return json.dumps([]), 200, {'content-type':'application/json'}
            texts = self.model_ocr.ocr(segments, image, threshold)
            results = []
            for s,t in zip(segments,texts):
                results.append({'segment_name': s['name'], 'value':t})
            return json.dumps(results), 200, {'content-type':'application/json'}

        @self.blueprint.route('/qr/<title>', methods=['GET'])
        def qr(title):
            """
            Generate pdf of qr codes, after the /qr/ put title for
            each qr.
            The data in the qr code will be cvmonitor-title-16_random_characters
            ---
            description: get pdf of qr codes 
            get:
            parameters:
            - in: path
              name: title
              schema:
                  type: string
                  required: true
                  default: cvmonitor
            - in: query
              name: width
              schema:
                  type: number
                  required: false
            - in: query
              name: height
              schema:
                  type: number
                  required: false
            responses:
              '200':
                descritption: pdf of results
                content:
                    application/pdf:
            """
            try:
                width = int(request.args.get('width'))
            except:
                width = None
            try:
                height = int(request.args.get('height'))
            except:
                height = None

            headers = {
                "Content-Type":'application/pdf',
                'Content-Disposition': 'attachment; filename="random-qr.pdf"'
            }
            pdf_buffer = io.BytesIO()
            generate_pdf(pdf_buffer,title,width,height)
            pdf_buffer.seek(0)
            return pdf_buffer.read(), 200, headers