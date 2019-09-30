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

def angle_color(angle):
    if angle == 1:
        color = [0, 0 , 255]
    if angle == 2:
        color = [0, 255 , 0]
    if angle == 3:
        color = [255, 0 , 0]
    if angle == 4:
        color = [0, 255, 255]
    if angle == 5:
        color = [128, 128, 128]
    if angle == 6:
        color = [255, 255, 255]
    return color

DATASET_DIR = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset')
img_path = os.path.join(DATASET_DIR, 'JPEGImages')
ship_angle_path = os.path.join(DATASET_DIR, 'ship_angle')
save_path = os.path.join(cfg.ROOT_DIR, 'dataset/Ship_Dataset', 'ship_angle_show')

pathDir = os.listdir(ship_angle_path)
pathDir.sort()
for idx in xrange(len(pathDir)):  
    filename = pathDir[idx]
    labelname = os.path.splitext(filename)[0]

    img = cv2.imread(os.path.join(img_path, labelname + '.jpg'))
    img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC) 
    
    angle_label = cv2.imread(os.path.join(ship_angle_path, labelname + '.png'), 0)
    
    label = np.where(angle_label > 0.5)
    label_show = np.array(img, dtype=np.uint8)
    scale = 2
    for i in xrange(len(label[0])):
        color = angle_color(angle_label[label[0][i], label[1][i]])
        label_show[label[0][i] * scale, label[1][i] * scale, 0] =  color[0]
        label_show[label[0][i] * scale, label[1][i] * scale, 1] =  color[1]
        label_show[label[0][i] * scale, label[1][i] * scale, 2] =  color[2]    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, labelname + '.jpg'), label_show) 

    print '[{}/{}] showing ship angle on images for {}.jpg'.format(idx, len(pathDir), labelname)

    
    
    
    
    
    
    
    
    
    