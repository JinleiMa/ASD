#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import os
import argparse
import sys
import pprint
import numpy as np
import _init_paths
from config import cfg  # config mnc
from caffeWrapper.SolverWrapper import train_net
import caffe
from datasets.data_extractor import get_rroidb, get_probdb, get_angle_label # D
import google.protobuf.text_format

def parse_args():
    """ Parse input arguments
    """
    
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default = cfg.GPU_ID, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default = os.path.join(cfg.ROOT_DIR, 'Models/solver.prototxt'), type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default = cfg.iters_numbers, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=os.path.join(cfg.ROOT_DIR, 'Models/pretrained_weight/vgg16.caffemodel'), type=str)                         

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
        
    roidb = get_rroidb()
    probdb = get_probdb()
    AngleLabeldb = get_angle_label()
    print '{:d} roidb entries'.format(len(roidb))

    root = os.path.abspath('..')
    output_dir = os.path.join(cfg.ROOT_DIR, 'Models/Training')
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, probdb, AngleLabeldb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters= args.max_iters )



