from flask import Flask, Blueprint, request, abort, Response
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

def order_points(pts):
    """
    order such that [top-left,top-right,bottom-right,bottom-left]
    """
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def align_by_qrcode(image, detected_qrcode, qrsize=100, boundery = 20.0):
    """
    Aling image by qrcode, normalize by qrcode size
    """
    tgt_pts = np.array([[boundery,boundery],[qrsize,boundery],[qrsize,qrsize],[boundery,qrsize]],np.float32)
    shape_pts = np.array([[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0.0,image.shape[0]]],np.float32)
    src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
    src_pts=order_points(src_pts)
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)
    res = M @ np.concatenate([shape_pts,np.ones((4,1))],1).transpose()
    for r in range(4):
        res[:,r]/=res[-1,r]
    width = int(np.ceil(max(res[0,:]))) #+ int(np.floor(min(res[0,:])))
    height = int(np.ceil(max(res[1,:]))) #+ int(np.floor(min(res[1,:])))
    warped = cv2.warpPerspective(image, M,(width,height))
    return warped

def find_qrcode(image, prefix):
    if len(image.shape)==3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    image = clahe.apply(image)
    decodedObjects = pyzbar.decode(image)
    detected_qrcode = None
    for obj in decodedObjects:
        text = obj.data.decode()
        if text.startswith(prefix):
            detected_qrcode = obj
            break
    return detected_qrcode

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
            if os.environ.get('CVMONITOR_SKIP_ALIGN')=='TRUE':
                return request.data, 200, {'content-type':'image/jpeg','X-MONITOR-ID': data}
            image = np.asarray(imageio.imread(request.data))

            qrprefix = str(os.environ.get('CVMONITOR_QR_PREFIX','cvmonitor'))
            qrsize = int(os.environ.get('CVMONITOR_QR_TARGET_SIZE',100))
            boundery = float(os.environ.get('CVMONITOR_QR_BOUNDERY_SIZE',20))
            detected_qrcode = find_qrcode(image, qrprefix)
            if detected_qrcode is None:
                abort(400, "Could not find the qr code to aling the image")
            data = detected_qrcode.data.decode()

            aligned_image = align_by_qrcode(image, detected_qrcode, qrsize=qrsize, boundery = boundery)
            b = io.BytesIO()
            imageio.imwrite(b, aligned_image, format='jpeg')
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
            bbox_list = [[s['left'],s['top'],s['right'],s['bottom']] for s in segments]
            if len(bbox_list) == 0:
                return json.dumps([]), 200, {'content-type':'application/json'}
            texts, preds = monitor_ocr.detect(self.model_ocr, bbox_list, image)
            more_texts = []
            for t, p, b in zip(texts,preds,bbox_list):
                if p>10.2:
                    more_texts.append(texts)
                else:
                    tt = pytesseract.image_to_string(image[b[1]:b[3],b[0]:b[2],:])
                    more_texts.append(tt)
                    print(f'tessercat predicted {tt}')
                print(f'pytorch predicted {p} : {t}')

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
            responses:
              '200':
                descritption: pdf of results
                content:
                    application/pdf:
            """
            headers = {
                "Content-Type":'application/pdf',
                'Content-Disposition': 'attachment; filename="random-qr.pdf"'
            }
            pdf_buffer = io.BytesIO()
            generate_pdf(pdf_buffer,title)
            pdf_buffer.seek(0)
            return pdf_buffer.read(), 200, headers