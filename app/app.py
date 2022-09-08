from importlib import import_module
from unicodedata import name
from flask import Flask, request, Response
from flask import render_template

import os
import sys
sys.path.append("../code")
from yolo import load_capture, generate_frame


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

def gen_frame(camera):
    while True:
        frame = load_capture()
        yield 


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()