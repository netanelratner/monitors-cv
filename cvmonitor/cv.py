from flask import Flask, Blueprint, request, abort, Response
import cv2
import imageio
import ujson as json
import numpy as np
from pyzbar import pyzbar
import io
import os


class ComputerVision:

    def __init__(self):
        self.blueprint = Blueprint('cv', __name__)
        self.qrDecoder = cv2.QRCodeDetector()

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
            for obj in decodedObjects:
                try:
                    codes.append({
                        'data': obj.data.decode(),
                        'top': obj.rect.top,
                        'left': obj.rect.left,
                        'bottom': obj.rect.top + obj.rect.height,
                        'right': obj.rect.left + obj.rect.width,
                        'type': obj.type,
                    })
                except:
                    abort(500,"Error decoding bardcode data.")
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
            if os.environ.get('CVMONITOR_SKIP_ALIGN')=='TRUE':
                return request.data
            image = np.asarray(imageio.imread(request.data))
            # Detect and decode the qrcode
            data,bbox,rectifiedImage = self.qrDecoder.detectAndDecode(image)
            b = io.BytesIO()
            imageio.imwrite(b,rectifiedImage, format='jpeg')
            b.seek(0)
            return b.read(), 200, {'content-type':'image/jpeg'}


