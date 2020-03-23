from flask import Flask, Blueprint, request, abort
import cv2
import imageio
import ujson as json
import numpy as np
from pyzbar import pyzbar


class ComputerVision:

    def __init__(self):
        self.blueprint = Blueprint('cv', __name__)
        self.qrDecoder = cv2.QRCodeDetector()

        @self.blueprint.route('/ping/')
        def ping():
            return 'pong cv'

        @self.blueprint.route('/detect_codes', methods=['POST'])
        def detect_codes():
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
            return json.dumps(codes)
