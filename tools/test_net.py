import _init_paths

import caffe
import cv2
import numpy as np
import datetime
import os
import cPickle

from voc_eval_angle import voc_eval
from rpn.rbbox_transform import rbbox_transform_inv
from caffeWrapper.timer import Timer
from config import cfg
from nms.rotate_polygon_nms import rotate_gpu_nms

def get_voc_results_file_template(cls,date):

    filename = 'resultTxt.txt'
    path = os.path.join(save_prob_path, filename)
    return path

def im_detect(net, im):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
#    im = cv2.resize(im, (512,384), interpolation = cv2.INTER_CUBIC)  
    im = im.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    im = cv2.resize(im, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC) 

    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    blob = blob.transpose((0, 3, 1, 2))
    
    blobs = {'data' : blob}

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], 1.0]],dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

   # _t = {'im_detect' : Timer()}

#    _t['im_detect'].tic()
    blobs_out = net.forward(**blobs)
#    _t['im_detect'].toc()

    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes_initial = rois[:, 1:6]
    
    scores = blobs_out['cls_prob']

    box_deltas = blobs_out['bbox_pred']
    pred_boxes = rbbox_transform_inv(boxes_initial, box_deltas)

    return scores, pred_boxes, boxes_initial

file_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset')
test_file = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/ImageSets/Main/test.txt')
file_path_img = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/JPEGImages')
save_prob_path = os.path.join(cfg.ROOT_DIR, 'output')
test_prototxt = os.path.join(cfg.ROOT_DIR, 'Models/test.prototxt')

weight = os.path.join(cfg.ROOT_DIR, 'Models/Training/ASD_iter_160000.caffemodel')

thresh = 0.05
max_per_image = 100
num_classes = 2

CLASSES = ('__background__', 'ship')

with open(test_file) as f:
    image_index = [x.strip() for x in f.readlines()]

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(test_prototxt, weight, caffe.TEST)
net.name = os.path.splitext(os.path.basename(weight))[0]

num_images = len(image_index)
# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in xrange(num_images)]
             for _ in xrange(num_classes)]

# timers
_t = {'im_detect' : Timer()}

for i in xrange(num_images): 
    
    image_path = os.path.join(file_path_img, image_index[i] + '.jpg')
    im = cv2.imread(image_path)
    
    _t['im_detect'].tic()
    #_t['im_detect'].tic()
    scores, boxes, boxes_initial = im_detect(net, im)
    #_t['im_detect'].toc()

    # skip j = 0, because it's the background class
    for j in xrange(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            thresh = 0.005
            inds = np.where(scores[:, j] > thresh)[0]  
        if len(inds) == 0:
            thresh = 0.0000005
            inds = np.where(scores[:, j] > thresh)[0]      
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*5:(j+1)*5]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        
#        print cls_dets.shape
        keep = rotate_gpu_nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        all_boxes[j][i] = cls_dets
    _t['im_detect'].toc()

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in xrange(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

    print 'im_detect: {:d}/{:d} {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time)

if not os.path.exists(save_prob_path):
    os.mkdir(save_prob_path)
date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
det_file = os.path.join(save_prob_path, 'detections_ship' + '.pkl')
with open(det_file, 'wb') as f:
    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
   
for cls_ind, cls in enumerate(CLASSES):
    if cls == '__background__':
        continue
    print 'Writing {} VOC results file'.format(cls)
    filename = get_voc_results_file_template(cls,date)
    if not os.path.exists(filename):
        os.mknod(filename) 
    
    with open(filename, 'wt') as f:
        for im_ind, index in enumerate(image_index):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # the VOCdevkit expects 1-based indices
            for k in xrange(dets.shape[0]):                
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                        format(index, dets[k, -1],
                               dets[k, 0] + 1, dets[k, 1] + 1,
                               dets[k, 2], dets[k, 3], dets[k, 4]))
        
annopath = os.path.join(file_path, 'Annotations', '{:s}.xml')
imagesetfile = os.path.join(test_file)
cachedir = os.path.join(save_prob_path)
aps = []
recall = []
precision = []

# The PASCAL VOC metric changed in 2010
use_07_metric = True #True
print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')

for i, cls in enumerate(CLASSES):
    if cls == '__background__':
        continue
    #filename = get_voc_results_file_template(cls)
    rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh = 0.5,
        use_07_metric = use_07_metric)
    aps += [ap]
    recall += [rec]
    precision += [prec]
    print('AP for {} = {:.4f}'.format(cls, ap))
    with open(os.path.join(save_prob_path, 'ship_pr.pkl'), 'w') as f:
        cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
print('Mean AP = {:.4f}'.format(np.mean(aps)))
print('--------------------------------------------------------------')
print('Results computed with the **unofficial** Python eval code.')
print('Results should be very close to the official MATLAB eval code.')
print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
print('-- Thanks, The Management')
print('--------------------------------------------------------------')


















































