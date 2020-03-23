import ujson as json
import cv2
from flask import Flask, Blueprint, request
from gevent.pywsgi import WSGIServer
import gevent
import os
from .cv import ComputerVision

class Server:

    def __init__(self):
        self.app = Flask(__name__)
        self.cv = ComputerVision()
        self.app.register_blueprint(self.cv.blueprint, url_prefix='/v1/')

        @self.app.route('/ping/')
        def ping():
            return 'pong'


def main():
    server = Server()
    host=os.environ.get('CVMONITOR_HOST','127.0.0.1')
    port=int(os.environ.get('CVMONITOR_PORT','8088'))
    print(f'serving on {host}:{port}')
    WSGIServer((host, port), server.app).serve_forever()
