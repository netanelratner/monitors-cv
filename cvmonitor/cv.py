from flask import Flask, Blueprint, request, abort, Response
import cv2
import random
import imageio
import ujson as json
import numpy as np
from pyzbar import pyzbar
import io
import os
import base64
import qrcode
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .ocr import monitor_ocr
from prometheus_client import Summary
import pytesseract
import copy
from pylab import imshow, show
from .qr import generate_pdf, find_qrcode, read_codes
from .image_align import get_oriented_image, align_by_qrcode
from .ocr.text_spotting import text_spotting
from cvmonitor.ocr.utils import get_fields_info, is_text_valid, draw_segments
from .ocr.utils import get_ocr_expected_boxes
import time
np.set_printoptions(precision=3)

class ResultLogger():

    def __init__(self):
        self.basedir = './log/'
        self.index = 0
    
    def log_ocr(self, image, segments, server_ocr):
        self.index +=1
        folder_name = self.basedir + time.strftime("%Y_%m_%d_%H")  
        os.makedirs(folder_name,exist_ok=True)
        json.dump({'segments':segments,'server_ocr':server_ocr}, open(f'{folder_name}/{self.index:09}.json','w'))
        with open(f'{folder_name}/{self.index:09}.jpg','wb') as f:
            f.write(image)

class ComputerVision:
    def __init__(self):
        self.blueprint = Blueprint("cv", __name__)
        self.qrDecoder = cv2.QRCodeDetector()
        self.model_ocr = None
        self.devices = get_fields_info()
        self.resultsLogger = ResultLogger()
        prob_threshold = float(
            os.environ.get("CVMONITOR_SPOTTING_PROB_THRESHOLD", "0.3")
        )
        max_seq_len = int(os.environ.get("CVMONITOR_SPOTTING_MAX_SEQ_LEN", "10"))
        iou_threshold = float(
            os.environ.get("CVMONITOR_SPOTTING_IOU_THRESHOLD", "0.01")
        )
        sportting_model = os.environ.get("CVMONITOR_SPOTTING_MODEL_TYPE", "FP32")

        self.text_spotting = text_spotting.Model(
            prob_threshold=prob_threshold,
            max_seq_len=max_seq_len,
            iou_threshold=iou_threshold,
            model_type=sportting_model,
        )

        @self.blueprint.route("/ping/")
        def ping():
            return "pong cv"

        @self.blueprint.route("/detect_codes", methods=["POST"])
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
            codes = read_codes(image)
            return json.dumps(codes), 200, {"content-type": "application/json"}

        @self.blueprint.route("/align_image", methods=["POST"])
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
            corners = []
            try:
                corners = json.loads(request.headers.get('X-CORNERS'))
            except:
                pass
            use_exif = os.environ.get("CVMONITOR_ORIENT_BY_EXIF", "TRUE") == "TRUE"
            use_qr = os.environ.get("CVMONITOR_ORIENT_BY_QR", "FALSE") == "TRUE"
            qrprefix = str(os.environ.get("CVMONITOR_QR_PREFIX", "cvmonitor"))
            qrsize = int(os.environ.get("CVMONITOR_QR_TARGET_SIZE", 100))
            boundery = float(os.environ.get("CVMONITOR_QR_BOUNDERY_SIZE", 50))
            align_image_by_qr = (
                os.environ.get("CVMONITOR_SKIP_ALIGN", "TRUE") == "FALSE"
            )
            save_before_align = os.environ.get("CVMONITOR_SAVE_BEFORE_ALIGN") == "TRUE"
            save_after_align = os.environ.get("CVMONITOR_SAVE_AFTER_ALIGN") == "TRUE"

            imdata = io.BytesIO(request.data)
            image, detected_qrcode, _ = get_oriented_image(
                imdata, use_exif=use_exif, use_qr=use_qr
            )

            headers = {"content-type": "image/jpeg"}
            if save_before_align:
                imdata.seek(0)
                with open("original_image.jpg", "wb") as f:
                    f.write(imdata)

            if detected_qrcode is None:
                if align_image_by_qr:
                    abort(
                        400,
                        "Could not align the image by qr code, no such code detected",
                    )
            else:
                headers["X-MONITOR-ID"] = detected_qrcode.data.decode()

            if align_image_by_qr:
                logging.debug("Trying to align image by qr code")
                image, _ = align_by_qrcode(
                    image, detected_qrcode, qrsize, boundery, qrprefix
                )

            if save_after_align:
                imageio.imwrite("aligned_image.jpg", image)

            b = io.BytesIO()
            imageio.imwrite(b, image, format="jpeg")
            b.seek(0)
            return b.read(), 200, headers

        @self.blueprint.route("/run_ocr", methods=["POST"])
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
                description: ocr results
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
            segment_threshold = float(
                os.environ.get("CVMONITOR_SEGMENT_THRESHOLD", "0.95")
            )
            device_ocr_default_score = float(
                os.environ.get("CVMONITOR_DEVICE_OCR_DEFAULT_SCORE", "0.5")
            )
            device_ocr_score_threshold = float(
                os.environ.get("CVMONITOR_DEVICE_OCR_SCORE_THRESHOLD", "0.8")
            )
            suggest_segments = float(
                os.environ.get("CVMONITOR_SUGGEST_SEGMENT", "FALSE") == "TRUE"
            )
            spotting_ocr = os.environ.get("CVMONITOR_OCR_SPOTTING", "TRUE") == "TRUE"
            threshold = float(os.environ.get("CVMONITOR_SERVER_OCR_THRESHOLD", "0.8"))
            if not self.model_ocr:
                self.model_ocr = monitor_ocr.build_model()
            if not self.model_ocr:
                abort(500, "NN Model not found, could not run ocr")

            data = request.json
            assert "image" in data
            image_data = base64.decodebytes(data["image"].encode())
            image = np.asarray(imageio.imread(image_data))
            # Suggest segments
            have_names = False
            for s in  data.get("segments",[]) or []:
                if 'name' in s:
                    have_names = True
            if suggest_segments and (not data.get("segments") or not have_names):
                # Let's run segment detection.
                texts, boxes, scores, _ = self.text_spotting.forward(image)
                segments = []
                for text, box, score in zip(texts, boxes, scores):
                    if score > segment_threshold:
                        segments.append(
                            {
                                "value": text,
                                "left": float(box[0]),
                                "top": float(box[1]),
                                "right": float(box[2]),
                                "bottom": float(box[3]),
                                "score": float(score),
                                "source": "server",
                            }
                        )
                logging.debug(f"Detections (new): {segments}")
                return json.dumps(segments), 200, {"content-type": "application/json"}
            segments = copy.deepcopy(data.get("segments",[]))
            if len(segments) == 0:
                logging.error("No segments")
                return json.dumps([]), 200, {"content-type": "application/json"}

            # We will at most give results on segments:
            expected_boxes = get_ocr_expected_boxes(
                segments,
                self.devices,
                device_ocr_default_score,
                device_ocr_score_threshold,
            )
            if spotting_ocr:
                texts, boxes, scores, _ = self.text_spotting.forward(
                    image, expected_boxes=expected_boxes
                )
            else:
                texts, scores = self.model_ocr.ocr(expected_boxes, image, threshold)
            should_log = False
            if texts:
                for eb, text, score in zip(expected_boxes, texts, scores):
                    if score > threshold:
                        if segments[eb["index"]].get("value") != text:
                            should_log = True
                        segments[eb["index"]]["value"] = text
                        segments[eb["index"]]["score"] = float(score)
                        segments[eb["index"]]["source"] = "server"
            if should_log or random.randint(0,100)==0 or not texts:
                self.resultsLogger.log_ocr(image_data,data["segments"], {'expected':expected_boxes,'texts': texts, 'scores': [float(s) for s in scores]})
            for s in segments:
                if 'value' not in s:
                    s['value']=None
            logging.debug(f"Detections: {segments}")
            return json.dumps(segments), 200, {"content-type": "application/json"}


        @self.blueprint.route("/show_ocr/", methods=["POST"])
        def show_ocr():
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
                description: ocr results
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
            data = request.json
            assert "image" in data
            image = np.asarray(
                imageio.imread(base64.decodebytes(data["image"].encode()))
            )
            # Suggest segments
            if data.get("segments"):
                image = draw_segments(image, data.get('segments'))

            headers = {"content-type": "image/jpeg"}
            b = io.BytesIO()
            imageio.imwrite(b, image, format="jpeg")
            b.seek(0)
            return b.read(), 200, headers
            




        @self.blueprint.route("/qr/<title>", methods=["GET"])
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
                width = int(request.args.get("width"))
            except:
                width = None
            try:
                height = int(request.args.get("height"))
            except:
                height = None

            headers = {
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="random-qr.pdf"',
            }
            pdf_buffer = io.BytesIO()
            generate_pdf(pdf_buffer, title, width, height)
            pdf_buffer.seek(0)
            return pdf_buffer.read(), 200, headers


        @self.blueprint.route("/measurements/<device>", methods=["GET"])
        def get_measurements(device):
            return json.dumps([x for x in get_fields_info([device]).keys()]), 200, {'content-type': 'application/json'}
