from __future__ import print_function
from flask import Flask, request
import time
from darkflow.net.build import TFNet
import cv2
import numpy
import json
from flask import Flask, Response

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.25, "gpu": 1.0}
tfnet = TFNet(options)

app = Flask(__name__)


@app.route('/heartbeat')
def heartbeat():
    return Response(status=200)


@app.route('/test_upload', method=['POST'])
def hello_world():
    r = request
    return Response(status=200)


@app.route('/test_download')
def hello_world():
    return ''


@app.route("/detect", methods=['POST'])
def calculate_complexity():
    r = request
    start = time.time()
    nparr = numpy.fromstring(r.data, numpy.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # start = time.time()
    result = tfnet.return_predict(img_np)
    print(start - time.time())

    for obj in result:
        obj['confidence'] = str(obj['confidence'])

    json_result = json.dumps(result)

    return Response(response=json_result, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host="192.168.0.25", port=5010)