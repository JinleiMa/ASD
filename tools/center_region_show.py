#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:27:32 2018

@author: majinlei
"""

import os
import os.path
import numpy as np
import cv2
import _init_paths
from config import cfg

DATASET_DIR = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset')
img_path = os.path.join(DATASET_DIR, 'JPEGImages')
center_region_path = os.path.join(DATASET_DIR, 'center_region')
save_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset', 'center_region_show')

pathDir = os.listdir(center_region_path)
pathDir.sort()
for idx in xrange(len(pathDir)):  
    filename = pathDir[idx]
    labelname = os.path.splitext(filename)[0]

    img = cv2.imread(os.path.join(img_path, labelname + '.jpg'))
    img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC) 
    
    img_center = cv2.imread(os.path.join(center_region_path, labelname + '.png'), 0)
    
    center = np.where(img_center == 1)
    scale = 2
    prob_show = np.array(img, dtype=np.uint8)
    for i in xrange(len(center[0])):
        prob_show[center[0][i] * scale, center[1][i] * scale, 0] =  0
        prob_show[center[0][i] * scale, center[1][i] * scale, 1] =  0
        prob_show[center[0][i] * scale, center[1][i] * scale, 2] =  255
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, labelname + '.jpg'), prob_show) 

    print '[{}/{}] showing center region on images for {}.jpg'.format(idx, len(pathDir), labelname)

    
    
    