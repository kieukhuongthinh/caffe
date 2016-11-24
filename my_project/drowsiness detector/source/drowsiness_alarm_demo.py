import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import sys
from caffe.proto import caffe_pb2
from google.protobuf import text_format

#------------NETWORK SETUP------------------
# Make sure that caffe is on the python path:
root_path = './'  # this file is expected to be in {root_path}/
os.chdir(root_path)
sys.path.insert(0, 'python')

# setup caffe mode
CPU_ONLY = True
if CPU_ONLY:
    caffe.set_mode_cpu()
else:
    caffe.set_device(0)
    caffe.set_mode_gpu()

# load labels
labelmap_file = 'ssd_net/data/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = 'ssd_net/models/SSD_300x300/deploy.prototxt'
model_weights = 'ssd_net/models/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_5700.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

#-----------WEBCAM SETUP------------
import cv2
import time

sleepy_id = 1 # copy from labelmap_voc.prototxt
window_title = 'drowsiness detector demo'
mirror_effect = True
resize_webcam = True
webcam_w = 640
webcam_h = 480
record_video = True

cap = cv2.VideoCapture(0)
if record_video:
    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 15.0, (webcam_w, webcam_h))

if cap.isOpened(): # try to get the first frame
   rval, frame = cap.read()
else:
   rval = False

frame_count = 0
ini_time = time.time() # in seconds

font = cv2.FONT_HERSHEY_TRIPLEX

skipTime = 8 # get 1 frame per 8 seconds
lastTime = time.time() - skipTime

def insertFPS(frame_count, ini_time, image):
       #------------
       txtScale = 0.8
       txtThick = 1
       txtType = cv2.CV_AA

       marginV = 10
       marginH = 5

       color = (1., 1., 1.)
       display_txt = '%s: %.1f'%('FPS: ', frame_count/(time.time()-ini_time))

       (txtW, txtH) = cv2.getTextSize(display_txt, font, txtScale, txtThick)[0]

       topleft = (webcam_w - txtW - marginH*2 - 10, webcam_h - txtH - marginV*2 - 10) # box
       textBLeft = (topleft[0] + marginH, topleft[1] + marginV + txtH)

       #--------insert transperent recs
       overlay = image.copy()
       cv2.rectangle(overlay, topleft, (topleft[0] + txtW + marginH*2, topleft[1] + txtH + marginV*2), color, thickness=-1)

       opacity = 0.4
       cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

       #------------insert text
       cv2.putText(image, display_txt, textBLeft, font, txtScale, (0, 0, 0), txtThick)

def alarmSleepingFaces((xmin, ymin, xmax, ymax), image, label_id, color, frame_count):
       #------------
       topleft = (xmin, ymin) # face box
       bottomright = (xmax, ymax) # face box
       marginV = 10
       marginH = 5
       
       txtScale = 0.8
       txtThick = 1
       txtType = cv2.CV_AA
       recThick = 2

       textBLeft = (topleft[0] + marginH, topleft[1] - recThick/2 - marginV)
       (txtW, txtH) = cv2.getTextSize(display_txt, font, txtScale, txtThick)[0]
       txtBgTLeft = (topleft[0] - recThick/2, topleft[1] - txtH - marginV*2)

       opacity = 0.4

       #--------insert face box
       overlay = image.copy()
       cv2.rectangle(overlay, topleft, bottomright, color, thickness=recThick, lineType=8, shift=0)

       #--------insert alarm signal
       if label_id == sleepy_id and frame_count % 2 == 0:
           cv2.rectangle(overlay, txtBgTLeft, (txtBgTLeft[0] + txtW + marginH*2, txtBgTLeft[1] + txtH + marginV*2), color, thickness=-1)
           cv2.putText(image, display_txt, textBLeft, font, txtScale, (1., 1., 1.), txtThick)

       cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
while True:
    rval, frame = cap.read()
    frame_count = frame_count + 1

    # resize frame to (webcam_w, webcam_h) size
    if resize_webcam:
        h = frame.shape[0]
        w = frame.shape[1]
        if (w != webcam_w and h != webcam_h):
            frame = cv2.resize(frame, (webcam_w, webcam_h))

    # flip frame vertically and get mirror effect
    if mirror_effect:
        frame = cv2.flip(frame, 1)
 
    # convert BGR to RGB, [0, 255] to [0.0, 1.0] float32
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame / 255.
    image = image.astype(np.float32)

    # put image into the net
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    now = time.time()
    if (now - lastTime) > skipTime:
       # Forward pass.
       print 'detecting...'
       start = time.time()
       detections = net.forward()['detection_out']
       end = time.time()
       lastTime = now
       print 'finish detecting in ' + str(end - start) + ' s'

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in xrange(top_conf.shape[0]):
       xmin = int(round(top_xmin[i] * image.shape[1]))
       ymin = int(round(top_ymin[i] * image.shape[0]))
       xmax = int(round(top_xmax[i] * image.shape[1]))
       ymax = int(round(top_ymax[i] * image.shape[0]))

       score = top_conf[i]

       label = int(top_label_indices[i])
       label_name = top_labels[i]
       display_txt = '%s: %.2f'%(label_name, score)
       color = colors[label]

       alarmSleepingFaces((xmin, ymin, xmax, ymax), image, label, color, frame_count)

    insertFPS(frame_count, ini_time, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if record_video:
        # write the result to video
        out.write((image*255.0).astype('u1'))
    
    cv2.imshow(window_title, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if record_video:
    out.release()
cv2.destroyAllWindows()
