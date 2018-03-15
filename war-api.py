import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import time
import json
from flask import Flask, request, Response

dn.set_gpu(0)
net = dn.load_net("cfg/yolo.cfg", "yolo.weights", 0)
meta = dn.load_meta("cfg/coco.data")

app = Flask(__name__)

@app.route('/heartbeat')
def heartbeat():
    return Response(status=200)


@app.route('/test_upload', methods=['POST'])
def test_upload():
    r = request
    return Response(status=200)


@app.route('/test_download')
def test_download():
    return ''

@app.route("/detect", methods = ['POST'])
def calculate_complexity():
    start = time.time()
    r = request
    start = time.time()
    fh = open("./testImage.jpg", "wb")
    fh.write(r.data)
    fh.close()

    # start = time.time()
    # result = tfnet.return_predict(img_np)
    boxes = dn.detect(net, meta, "./testImage.jpg")
    print(boxes)

    # TODO: change to x,y,w,h
    boxesInfo = list()
    for box in boxes:
        boxesInfo.append({
            "label": box[0],
            "confidence": box[1],
            "topleft": {
                "x": box[2][0],
                "y": box[2][1]},
            "bottomright": {
                "x": box[2][2],
                "y": box[2][3]}
        })
    print(start - time.time())

    json_result = json.dumps(boxesInfo)

    # print(result)

    # for obj in result:
    #     obj['confidence'] = str(obj['confidence'])
    #
    # json_result = json.dumps(result)
    #
    # print(json_result)

    # example_json = {"topleft": {"y": 124, "x": 76}, "confidence": "0.793635", "bottomright": {"y": 455, "x": 556}, "label": "bicycle"}
    # json_eg = json.dumps(example_json)
    # print(json_eg)
    print(start - time.time())
    return Response(response=json_result, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host="192.168.6.131", port=5010)


