# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import time

dn.set_gpu(0)
net = dn.load_net("cfg/yolo.cfg", "yolo.weights", 0)
meta = dn.load_meta("cfg/coco.data")

start = time.time()
boxes = dn.detect(net, meta, "data/dog.jpg")
print boxes[0][2][0]
print(time.time()-start)

# And then down here you could detect a lot more images like:
start = time.time()
r = dn.detect(net, meta, "data/eagle.jpg")
print r
print(time.time()-start)

start = time.time()
r = dn.detect(net, meta, "data/giraffe.jpg")
print r
print(time.time()-start)

start = time.time()
r = dn.detect(net, meta, "data/horses.jpg")
print r
print(time.time()-start)

start = time.time()
r = dn.detect(net, meta, "data/person.jpg")
print r
print(time.time()-start)




