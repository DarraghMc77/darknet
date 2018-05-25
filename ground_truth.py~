import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import cv2
import numpy
import json
import darknet as dn

dn.set_gpu(0)
net = dn.load_net("cfg/yolov2-voc.cfg", "yolov2-voc.weights", 0)
meta = dn.load_meta("cfg/voc.data")

video_loc = "/home/ale/Desktop/VC/darknet/GT-Videos/bottle.mov"
gt_txt_file = "/home/ale/Desktop/VC/darknet/GT-txt/bottle_gt.txt"

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def centeredCrop(img, new_height, new_width):

   width =  numpy.size(img,1)
   height =  numpy.size(img,0)

   left = numpy.ceil((width - new_width)/2.)
   top = numpy.ceil((height - new_height)/2.)
   right = numpy.floor((width + new_width)/2.)
   bottom = numpy.floor((height + new_height)/2.)
   cImg = img[top:bottom, left:right]
   return cImg

def cropimread(img_pre, xcrop, ycrop):
    ysize, xsize, chan = img_pre.shape
    xoff = (xsize - xcrop) // 2
    yoff = (ysize - ycrop) // 2
    img= img_pre[yoff:-yoff,xoff:-xoff]
    return img

def create_cropped_images(folder, video_location, crop_width, crop_height):
    cap = cv2.VideoCapture(video_location)
    count = 0

    # if not os.path.exists('/Gt_Images/' + folder):
    #     try:
    #         original_umask = os.umask(0)
    #         os.makedirs('/Gt_Images/' + folder, 0777)
    #     finally:
    #         os.umask(original_umask)
    #     # os.makedirs('/Gt_Images/' + folder)

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite('test.jpg', frame)

        resized_frame = cropimread(frame, crop_width, crop_height)

        cv2.imwrite('Gt_Images/' + folder + '/test{}.jpg'.format(count), resized_frame)

        count = count + 1
        cv2.imshow('frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ""


def cv_read_video(video_location, test_file):
    cap = cv2.VideoCapture(video_location)
    f = open('GT-txt/' + test_file + '.txt', 'w')

    count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        # fh = open('test.jpg'.format(count), 'w')
        # fh.write(frame)
        # fh.close()
        cv2.imwrite('test.jpg', frame)
        # continue

        # count = count + 1

        width = 352
        height = 352

        resized_frame = cropimread(frame, width, height)

        # result = tfnet.return_predict(resized_frame)

        # if len(result) > 0:
        #     for res in result:
        #         res['image'] = count
        #     print(result)
        #
        #     f.write(str(result) + '\n')

        boxes = dn.detect(net, meta, 'Gt_Images/' + test_file +'/test{}.jpg'.format(count))

        boxesInfo = list()

        for box in boxes:
            centre_x = box[2][0]
            centre_y = box[2][1]
            box_width = box[2][2]
            box_height = box[2][3]

            top_x = max(0, (centre_x - box_width / 2))
            top_y = max(0, (centre_y - box_height / 2))
            bottom_x = min(height-1, (centre_x + box_width / 2))
            bottom_y = min(height-1, (centre_y + box_height / 2))


            boxesInfo.append({
                "image": count,
                "label": box[0],
                "confidence": box[1],
                "topleft": {
                    "x": top_x,
                    "y": top_y},
                "bottomright": {
                    "x": bottom_x,
                    "y": bottom_y}
            })

        result = json.dumps(boxesInfo)

        f.write(str(result) + '\n')

        # cv2.imwrite('Gt_Images/bottle/test{}.jpg'.format(count), resized_frame)

        count = count + 1
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resized_frame = cv2.resize(frame, (300, 300))

        # pts1 = numpy.float32([[0.8152174, 0.0, 0.0], [0.0, 0.8152174, 0.0], [0.0, 0.0, 1.0]])
        # gray = cv2.warpAffine(frame, pts1, (300, 300))
        # gray = frame[pts1]

        cv2.imshow('frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    f.close()

    cap.release()
    cv2.destroyAllWindows()
    return ""

if __name__ == '__main__':
    images = create_cropped_images('monitorbottle', '/home/ale/Desktop/VC/darknet/GT-Videos/monitorbottle.mov', 352, 352)
    # video = cv_read_video('/home/ale/Desktop/VC/darknet/GT-Videos/monitorbottle.mov', 'monitorbottle')