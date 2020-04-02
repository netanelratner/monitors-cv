import gevent
from gevent.monkey import patch_all;
patch_all()
import logging
import ujson as json
import cv2
from flask import Flask, Blueprint, request
from gevent.pywsgi import WSGIServer
import gevent
import os
from .ocr import get_models
from .cv import ComputerVision
from flasgger import Swagger
from prometheus_flask_exporter import PrometheusMetrics
from setuptools_scm import get_version
import exifread
from . import __version__


class Server:

    def __init__(self,log_level=logging.DEBUG):
        self.app = Flask(__name__)
        self.app.logger.setLevel(log_level)
        self.metrics = PrometheusMetrics(self.app)
        self.metrics.info('app_info', 'Version info', version=__version__)
        self.cv = ComputerVision()
        self.app.register_blueprint(self.cv.blueprint, url_prefix='/v1/')
        self.app.config['SWAGGER'] = {
            'title': 'Corona Medical Monitors Camera Monitoring API',
            'uiversion': 3,
            'openapi': '3.0.2',
            'version': __version__
        }
        self.swagger = Swagger(self.app)

        @self.app.route('/ping/')
        def ping() ->str:
            """
            ping
            ---
            description:  get a pong
            """
            return 'pong'

def init_logs():
    log_level_name =  os.environ.get('CVMONITOR_LOG_LEVEL','DEBUG')
    log_level = logging.DEBUG
    if (log_level_name=='INFO'):
        log_level = logging.INFO
    if (log_level_name=='WARNING'):
        log_level = logging.WARNING
    if (log_level_name=='ERROR'):
        log_level = logging.ERROR
    
    for logger in (
        
        logging.getLogger(),
    ):
        logger.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        sh  = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    exifread.exif_log.setup_logger(False,'white')    
    return log_level
    

def main():
    log_level = init_logs()
    server = Server(log_level)
    host=os.environ.get('CVMONITOR_HOST','0.0.0.0')
    port=int(os.environ.get('CVMONITOR_PORT','8088'))
    logging.info('checking if model exists:')
    get_models()
    logging.info(f'serving on http://{host}:{port}/apidocs')
    WSGIServer((host, port), server.app).serve_forever()
