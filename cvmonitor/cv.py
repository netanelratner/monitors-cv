from flask import Flask, Blueprint, request, abort, Response
import cv2
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

def generate_pdf(pdf_file,title):
    nrows = int(os.environ.get('CVMOINTOR_QR_PDF_ROWS',6))
    ncols = int(os.environ.get('CVMOINTOR_QR_PDF_ROWS',4))
    with  PdfPages(pdf_file) as pdf:
        index = 0
        fig, axarr = plt.subplots(nrows,ncols, figsize= [8 , 11])
        fig.tight_layout(pad=4, h_pad=3, w_pad=2)
        for y in range(nrows):
            for x in range(ncols):
                index+=1
                qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
                text = f'cvmonitors-{title}-{uuid4().hex[:16]}'
                qr.add_data(text)
                qr.make(fit=True)
                img = qr.make_image(fill_color='black', back_color='white')
                axarr[y,x].set_title(f'{title}\n{uuid4().hex[:16]}', fontsize=8)
                axarr[y,x].set_xticks([])
                axarr[y,x].set_yticks([])
                axarr[y,x].set_yticklabels([])
                axarr[y,x].set_xticklabels([])
                axarr[y,x].imshow(img,cmap='gray')


        pdf.savefig(fig)


def read_codes(image):
    """
    read barcodes in an image
    """
    decodedObjects = pyzbar.decode(image)
    codes = []
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
            continue
    return codes


class ComputerVision:

    def __init__(self):
        self.blueprint = Blueprint('cv', __name__)
        self.qrDecoder = cv2.QRCodeDetector()
        self.model_ocr = monitor_ocr.build_model()

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
            if os.environ.get('CVMONITOR_SKIP_ALIGN')=='TRUE':
                return request.data, 200, {'content-type':'image/jpeg','X-MONITOR-ID': data}
            image = np.asarray(imageio.imread(request.data))
            # Detect and decode the qrcode
            data,bbox,rectifiedImage = self.qrDecoder.detectAndDecode(image)
            decodedObjects = pyzbar.decode(image)
            
            max_width = os.environ.get('CVMONITOR_MAX_IMAGE_WIDTH')
            
            b = io.BytesIO()
            imageio.imwrite(b,rectifiedImage, format='jpeg')
            cv2.resize()
            b.seek(0)
            return b.read(), 200, {'content-type':'image/jpeg','X-MONITOR-ID': data}

        @self.blueprint.route('/run_ocr', methods=['POST'])
        def run_ocr():
            """
            Run ocr on an image
            ---
            """
            if not self.model_ocr:
                abort(500,'NN Model not found, could not run ocr')
            data = request.json
            assert 'image' in data
            image = np.asarray(imageio.imread(base64.decodebytes(data['image'].encode())))
            if not data.get('segments'):
                return json.dumps([]), 200, {'content-type':'application/json'}
            segments = data['segments']
            bbox_list = [[s['left'],s['top'],s['right'],s['bottom']] for s in segments]
            texts = monitor_ocr.detect(self.model_ocr, bbox_list, image)
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
            """
            headers = {
                "Content-Type":'application/pdf',
                'Content-Disposition': 'attachment; filename="random-qr.pdf"'
            }
            pdf_buffer = io.BytesIO()
            generate_pdf(pdf_buffer,title)
            pdf_buffer.seek(0)
            return pdf_buffer.read(), 200, headers