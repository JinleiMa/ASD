import numpy as np 
import os 
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
from PIL import Image
import xml.etree.ElementTree as xmlET
import scipy
from config import cfg
import cPickle
from collections import Counter

DATASET_DIR = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset')
img_file_txt = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/ImageSets/Main/trainval.txt')
center_region_save_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/center_region')
ship_angle_save_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset/ship_angle')

with open(img_file_txt) as f:
    gt_images = [x.strip() for x in f.readlines()]

def get_rroidb():
    
    cache_file = os.path.join(cfg.ROOT_DIR, 'dataset/cache', 'gt_roidb.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidbs = cPickle.load(fid)
        print 'gt roidb loaded from {}'.format(cache_file)
        return roidbs
        
    roidbs = []
        
    for gt_image in gt_images:
        image_file = os.path.join(DATASET_DIR, 'JPEGImages', gt_image + '.jpg')
        filename = os.path.join(DATASET_DIR, 'Annotations', gt_image + '.xml')
        tree = xmlET.parse(filename)
        img = cv2.imread(image_file)
        img_height, img_width, _ = img.shape
        scale_ratio = cfg.IMAGE_WIDTH / np.float(img_width)
        boxes = []	
        
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox_angle')
            # Make pixel indexes 0-based
            x_ctr = float(bbox.find('center_x').text) - 1
            y_ctr = float(bbox.find('center_y').text) - 1
            width = float(bbox.find('width').text)
            height = float(bbox.find('height').text)
            
            x_ctr = x_ctr * scale_ratio
            y_ctr = y_ctr * scale_ratio
            width = width * scale_ratio
            height = height * scale_ratio
            width = width * cfg.TRAIN.GT_MARGIN
            height = height * cfg.TRAIN.GT_MARGIN
            angle = float(bbox.find('angle').text)
            if height >= width:
                height, width = width, height
                angle = 90 + angle
            if angle < -45.0:
                angle = angle + 180
  
            boxes.append([x_ctr, y_ctr, height, width, angle])
            
        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)	        
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32) #text or non-text      

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1 # cls_text
            overlaps[idx, 1] = 1.0 # cls_text
        
        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_overlaps = overlaps.toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
        
        roidb = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': image_file,
            'boxes': gt_boxes,
            'flipped' : False,
            'gt_overlaps' : overlaps,
            'height': img_height,
            'width': img_width,
            'max_overlaps' : max_overlaps,
            'rotated': False
            }
        roidbs.append(roidb)

    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidbs, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb to {}'.format(cache_file)

    return roidbs


def get_probdb():
    
    cache_file = os.path.join(cfg.ROOT_DIR, 'dataset/cache', 'gt_probdb.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            probdbs = cPickle.load(fid)
        print 'gt center_region_db loaded from {}'.format(cache_file)
        return probdbs
    
    probdbs = []
    idx = 0
    for gt_image in gt_images:
      
        img_center = np.zeros((cfg.IMAGE_HEIGHT/2, cfg.IMAGE_WIDTH/2), dtype=np.uint8)
        
        image_file = os.path.join(DATASET_DIR, 'JPEGImages', gt_image + '.jpg')
        filename = os.path.join(DATASET_DIR, 'Annotations', gt_image + '.xml')
        tree = xmlET.parse(filename)
        img = cv2.imread(image_file)
        img_height, img_width, _ = img.shape
        scale_ratio = cfg.IMAGE_WIDTH / np.float(img_width)
        long_ratio = 0.125
        short_ratio = 0.75
        
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox_angle')
            # Make pixel indexes 0-based
            x_ctr = float(bbox.find('center_x').text) - 1
            y_ctr = float(bbox.find('center_y').text) - 1
            width = float(bbox.find('width').text)
            height = float(bbox.find('height').text)
            
            x_ctr = x_ctr * scale_ratio
            y_ctr = y_ctr * scale_ratio
            width = width * scale_ratio
            height = height * scale_ratio
            angle = float(bbox.find('angle').text)
            if height >= width:
                height, width = width, height
                angle = 90 + angle
            if angle < -45.0:
                angle = angle + 180
                
                
            if width > height:
              if width < 50:
                  width_center = 0.8 * width
                  height_center = 0.9 * height
              else:
                  width_center = long_ratio * width
                  height_center = short_ratio * height
            else:
                if height < 50:
                    width_center = 0.9 * width
                    height_center = 0.8 * height
                else:
                    width_center = short_ratio * width
                    height_center = long_ratio * height    
      
            rect_tuple = tuple([[y_ctr/2, x_ctr/2], [height_center/2, width_center/2], angle])        
            
            box = cv2.boxPoints(rect_tuple)
            box = np.int0(box)
            box_center = np.zeros((4, 2), np.int64)
            for i in xrange(len(box)):
                box_center[i][1] = box[i][0]
                box_center[i][0] = box[i][1]
            
            cv2.fillPoly(img_center, [box_center], 1)
        if not os.path.exists(center_region_save_path):
            os.makedirs(center_region_save_path)
        cv2.imwrite(os.path.join(center_region_save_path, gt_image + '.png'), img_center) 
        idx += 1
        print '[{}/{}] generateing and saving center region segmentation label for {}.jpg'.format(idx, len(gt_images), gt_image)

        prob = img_center
        blob_caffe = np.zeros((1, 1, prob.shape[0], prob.shape[1]),dtype=np.uint8)
        blob_caffe[0, 0, 0:prob.shape[0], 0:prob.shape[1]] = prob
        probdb = {
            'gt_prob': blob_caffe,
        }   
        
        probdbs.append(probdb)

    with open(cache_file, 'wb') as fid:
        cPickle.dump(probdbs, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb to {}'.format(cache_file)
        
    return probdbs

def angle_range(angle):
    if angle >= -45 and angle < -15:
        num = 1
    if angle >= -15 and angle < 15:
        num = 2
    if angle >= 15 and angle < 45:
        num = 3
    if angle >= 45 and angle < 75:
        num = 4
    if angle >= 75 and angle < 105:
        num = 5
    if angle >= 105 and angle <= 135:
        num = 6
    return num
    
def get_angle_label():
    
    cache_file = os.path.join(cfg.ROOT_DIR, 'dataset/cache', 'gt_AngleLabeldb.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            AngleLabeldbs = cPickle.load(fid)
        print 'gt ship angle db loaded from {}'.format(cache_file)
        return AngleLabeldbs
    
    AngleLabeldbs = []
    idx = 0
    for gt_image in gt_images:
      
        angle_label = np.zeros((cfg.IMAGE_HEIGHT/2, cfg.IMAGE_WIDTH/2), dtype=np.uint8)
        
        image_file = os.path.join(DATASET_DIR, 'JPEGImages', gt_image + '.jpg')
        filename = os.path.join(DATASET_DIR, 'Annotations', gt_image + '.xml')
        tree = xmlET.parse(filename)
        img = cv2.imread(image_file)
        img_height, img_width, _ = img.shape
        scale_ratio = cfg.IMAGE_WIDTH / np.float(img_width)
        
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox_angle')
            # Make pixel indexes 0-based
            x_ctr = float(bbox.find('center_x').text)
            y_ctr = float(bbox.find('center_y').text)
            width = float(bbox.find('width').text)
            height = float(bbox.find('height').text)
            
            x_ctr = x_ctr * scale_ratio
            y_ctr = y_ctr * scale_ratio
            width = width * scale_ratio
            height = height * scale_ratio
            angle = float(bbox.find('angle').text)
            
            width = 0.9 * width
            height = 0.9 * height
      
            rect_tuple = tuple([[y_ctr/2, x_ctr/2], [height/2, width/2], angle])        
            
            box = cv2.boxPoints(rect_tuple)
            box = np.int0(box)
            box_center = np.zeros((4, 2), np.int64)
            for i in xrange(len(box)):
                box_center[i][1] = box[i][0]
                box_center[i][0] = box[i][1]
            
            if height >= width:
                angle = 90 + angle
            if angle < -45.0:
                angle = angle + 180            
            
            num = angle_range(angle)
            cv2.fillPoly(angle_label, [box_center], num)
        
        if not os.path.exists(ship_angle_save_path):
            os.makedirs(ship_angle_save_path)
        cv2.imwrite(os.path.join(ship_angle_save_path, gt_image + '.png'), angle_label) 
        idx += 1
        print '[{}/{}] generateing and saving ship angle segmentation label for {}.jpg'.format(idx, len(gt_images), gt_image)
        
        blob_caffe = np.zeros((1, 1, angle_label.shape[0], angle_label.shape[1]), dtype=np.uint8)
        blob_caffe[0, 0, 0:angle_label.shape[0], 0:angle_label.shape[1]] = angle_label
        
        AngleLabeldb = {
            'angle_label': blob_caffe,
        }   
        
        AngleLabeldbs.append(AngleLabeldb)

    with open(cache_file, 'wb') as fid:
        cPickle.dump(AngleLabeldbs, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb to {}'.format(cache_file)
        
    return AngleLabeldbs        
    
    
    
    
#def get_angle_label():
#    
#    cache_file = os.path.join(cfg.ROOT_DIR, 'dataset/cache', 'gt_AngleLabeldb.pkl')
#    if os.path.exists(cache_file):
#        with open(cache_file, 'rb') as fid:
#            AngleLabeldbs = cPickle.load(fid)
#        print 'gt ship angle db loaded from {}'.format(cache_file)
#        return AngleLabeldbs
#    
#    AngleLabeldbs = []
#    angle_label_file_path = "/home/majinlei/data/Ship1537/angle_label"
#    for gt_image in gt_images:       
#      angle_label_file_name = os.path.join(angle_label_file_path, gt_image + '.png')
#      angle_label = cv2.imread(angle_label_file_name, 0)
#      angle_label = cv2.resize(angle_label, (cfg.IMAGE_WIDTH / 2,cfg.IMAGE_HEIGHT / 2), interpolation = cv2.INTER_NEAREST) 
#      
#      blob_caffe = np.zeros((1, 1, angle_label.shape[0], angle_label.shape[1]), dtype=np.uint8)
#      #print 'blob_caffe.shape: {}'.format(blob_caffe.shape)
#      #print 'prob.shape: {}'.format(prob.shape)
#      blob_caffe[0, 0, 0:angle_label.shape[0], 0:angle_label.shape[1]] = angle_label
#      
#      AngleLabeldb = {
#          'angle_label': blob_caffe,
#      }   
#      
#      print 'saving ship angle segmentation label for {}.jpg'.format(gt_image)
#      
#      AngleLabeldbs.append(AngleLabeldb)
#
#    with open(cache_file, 'wb') as fid:
#        cPickle.dump(AngleLabeldbs, fid, cPickle.HIGHEST_PROTOCOL)
#    print 'wrote gt roidb to {}'.format(cache_file)
#        
#    return AngleLabeldbs       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






