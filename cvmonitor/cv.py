from flask import Flask, Blueprint, request, abort, Response
from typing import List
import cv2
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
from pylab import imshow, show
from .qr import generate_pdf, find_qrcode, read_codes
from .image_align import get_oriented_image, align_by_qrcode
from .ocr.text_spotting import text_spotting
from cvmonitor.ocr.utils import get_fields_info, is_text_valid, draw_segments
from .ocr.utils import get_ocr_expected_boxes
from ..backend import data as Data
np.set_printoptions(precision=3)


class ComputerVision:
    def __init__(self):
        self.blueprint = Blueprint("cv", __name__)
        self.qrDecoder = cv2.QRCodeDetector()
        self.model_ocr = None
        self.devices = get_fields_info()
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

    def detect_codes(self, jpeg_data: bytes) -> Data.Codes:
        image = np.asarray(imageio.imread(jpeg_data))
        return  read_codes(image)

    def align_image(self, jpeg_data: bytes, corners: Data.ScreenCorners) -> (bytes, str):
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

        imdata = io.BytesIO(jpeg_data)
        image, detected_qrcode, _ = get_oriented_image(
            imdata, use_exif=use_exif, use_qr=use_qr
        )
        if save_before_align:
            imdata.seek(0)
            with open("original_image.jpg", "wb") as f:
                f.write(imdata)

        if detected_qrcode is None:
            raise RuntimeError("Could not align the image by qr code, no such code detected")
        monitor_id = detected_qrcode.data.decode()

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
        return b.read(),  monitor_id

    def run_ocr(self, record: Data.DeviceRecord) -> List[Data.Segment]:
        segment_threshold = float(
            os.environ.get("CVMONITOR_SEGMENT_THRESHOLD", "0.95")
        )
        device_ocr_default_score = float(
            os.environ.get("CVMONITOR_DEVICE_OCR_DEFAULT_SCORE", "0.5")
        )
        device_ocr_score_threshold = float(
            os.environ.get("CVMONITOR_DEVICE_OCR_SCORE_THRESHOLD", "0.8")
        )

        spotting_ocr = os.environ.get("CVMONITOR_OCR_SPOTTING", "TRUE") == "TRUE"
        threshold = float(os.environ.get("CVMONITOR_SERVER_OCR_THRESHOLD", "0.8"))

        image = np.asarray(imageio.imread(record.image))
        # Suggest segments
        if not record.segments:
            # Let's run segment detection.
            texts, boxes, scores, _ = self.text_spotting.forward(image)
            segments: List[Data.Segment] = []
            for text, box, score in zip(texts, boxes, scores):
                if score > segment_threshold:
                    segment = Data.Segment()
                    segment.value =  text
                    segment.left =  float(box[0])
                    segment.top =  float(box[1])
                    segment.right =  float(box[2])
                    segment.bottom =  float(box[3])
                    segment.score =  float(score)
                    segment.source =  "server"
                    segments.append(segment)
            logging.debug(f"Detections (new): {segments}")
            return segments

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
        for eb, text, score in zip(expected_boxes, texts, scores):
            if score > threshold:
                segments[eb["index"]].value = text
                segments[eb["index"]].score = float(score)
                segments[eb["index"]].source = "server"

        for s in segments:
            if 'value' not in s:
                s.value=None
        return segments


    def show_ocr(self, record: Data.DeviceRecord) -> bytes:
        image = np.asarray(imageio.imread(record.image))
        if record.segments:
            image = draw_segments(image, record.segments)

        b = io.BytesIO()
        imageio.imwrite(b, image, format="jpeg")
        b.seek(0)
        return b.read()
        
    def qr(self, title: str, width:int, height: int) -> bytes:
        """
        Returns a pdf with qr codes
        """
        pdf_buffer = io.BytesIO()
        generate_pdf(pdf_buffer, title, width, height)
        pdf_buffer.seek(0)
        return pdf_buffer.read()
