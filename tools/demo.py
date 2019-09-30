import _init_paths
from config import cfg
from nms.rotate_polygon_nms import rotate_gpu_nms
from caffeWrapper.timer import Timer
import numpy as np
import caffe, os, cv2
reload(cv2)
from rpn.rbbox_transform import rbbox_transform_inv


CLASSES = ('__background__', 'ship')
DATASET_DIR = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset')
img_path = os.path.join(DATASET_DIR, 'JPEGImages')
result_path = os.path.join(cfg.ROOT_DIR, 'output/results')
img_list_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/ImageSets/Main/test.txt')
img_list = np.loadtxt(img_list_path, dtype=str)

prototxt = os.path.join(cfg.ROOT_DIR, 'Models/test.prototxt')
caffemodel = os.path.join(cfg.ROOT_DIR, 'Models/Training/ASD_iter_160000.caffemodel')

caffe.set_mode_gpu()
caffe.set_device(cfg.GPU_ID)
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

for img_name in img_list:
    im_name = os.path.join(img_path, img_name + '.jpg')
    img = cv2.imread(im_name)
    
    img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC) 
    im = img.astype(np.float32, copy=True)
    im -= np.array([[[75.4916, 78.2288, 68.2117]]])
    
    timer = Timer()
    timer.tic()    
    
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    blob = blob.transpose((0, 3, 1, 2))
    
    blobs = {'data': blob}
    blobs['im_info'] = np.array([[blob.shape[2], blob.shape[3]]],dtype=np.float32)
    
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(**blobs)

    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes_inital = rois[:, 1:6]
    scores = blobs_out['cls_prob']
    box_deltas = blobs_out['bbox_pred']
    boxes = rbbox_transform_inv(boxes_inital, box_deltas)
    
    # Visualize detections for each class
    conf = 0.75
    CONF_THRESH = conf
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 5*cls_ind:5*(cls_ind + 1)] # D
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = rotate_gpu_nms(dets, NMS_THRESH) # D
        dets = dets[keep, :]
        
    dets[:, 2] = dets[:, 2] / cfg.TEST.GT_MARGIN
    dets[:, 3] = dets[:, 3] / cfg.TEST.GT_MARGIN
        
    return_bboxes = []    
    for idx in range(len(dets)):
        if (dets[idx][5] > CONF_THRESH):
            return_bboxes.append(dets[idx])
        
    result_boxes = return_bboxes
    
    timer.toc()
    print ('Detecting {}.jpg took {:.3f}s for '
            '{:d} object proposals').format(img_name, timer.average_time, boxes.shape[0])

    for idx in range(len(result_boxes)):
        cx,cy,h,w,angle, _ = result_boxes[idx]
        lt = [cx - w/2, cy - h/2,1]
        rt = [cx + w/2, cy - h/2,1]
        lb = [cx - w/2, cy + h/2,1]
        rb = [cx + w/2, cy + h/2,1]
        
        pts = []
        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)
        
        angle = -angle
        
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)
        
        M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
        M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
        M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
        rotation_matrix = M0.dot(M1).dot(M2)
        
        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        
        cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    cv2.imwrite(os.path.join(result_path, img_name + '.jpg'), img)

















































































