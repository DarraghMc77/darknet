import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import time
import json
from flask import Flask, request, Response, send_file

dn.set_gpu(0)
net = dn.load_net("cfg/yolov2-voc.cfg", "yolov2-voc.weights", 0)
meta = dn.load_meta("cfg/voc.data")

app = Flask(__name__)

@app.route('/heartbeat')
def heartbeat():
    return Response(status=200)

@app.route('/test_download')
def test_download():
    filename = 'test_image.jpg'
    return send_file(filename, mimetype='image/jpg')

@app.route("/test_upload", methods = ['POST'])
def test_upload():
    start = time.time()
    r = request
    fh = open("./testupImage.jpg", "wb")
    fh.write(r.data)
    fh.close()
    print(time.time()- start)
    return Response(status=200)

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

    boxesInfo = list()
    for box in boxes:
        centre_x = box[2][0]
        centre_y = box[2][1]
        box_width = box[2][2]
        box_height = box[2][3]

        top_x = max(0, (centre_x - box_width / 2))
        top_y = max(0, (centre_y - box_height / 2))
        bottom_x = min(416 - 1, (centre_x + box_width / 2))
        bottom_y = min(416 - 1, (centre_y + box_height / 2))

        boxesInfo.append({
            "label": box[0],
            "confidence": box[1],
            "topleft": {
                "x": top_x,
                "y": top_y},
            "bottomright": {
                "x": bottom_x,
                "y": bottom_y}
        })

    # TODO: change to x,y,w,h
    # boxesInfo = list()
    # for box in boxes:
    #     boxesInfo.append({
    #         "label": box[0],
    #         "confidence": box[1],
    #         "topleft": {
    #             "x": box[2][0],
    #             "y": box[2][1]},
    #         "bottomright": {
    #             "x": box[2][2],
    #             "y": box[2][3]}
    #     })
    # print(start - time.time())
    #
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

@app.route("/result", methods=['POST'])
def write_results():
    r = request
    print(r.data)

    with open("mobile_results.txt", "a") as myfile:
        myfile.write(r.data + "\n")

    return Response(response="success", status=200)

if __name__ == '__main__':
    app.run(host="192.168.6.131", port=5010, threaded = True)


