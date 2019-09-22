# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import PIL
import numpy as np
import os
import cPickle
import scipy

#from imdb import get_imdb
from config import cfg
from rpn.rbbox_transform import rbbox_transform
from nms.rbbox_overlaps import rbbx_overlaps
from nms.rotate_polygon_nms import rotate_gpu_nms

###########################################################
#
# 	     Rotation Bbox target distribution
#
###########################################################

def add_rbbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'
    
    #[dx, dy, dh, dw, da]
    bbox_para_num = 5
    
    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = \
                _compute_targets(rois, max_overlaps, max_classes)

    if cfg.TRAIN.RBBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
                np.array(cfg.TRAIN.RBBOX_NORMALIZE_MEANS), (num_classes, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.RBBOX_NORMALIZE_STDS), (num_classes, 1))
    else:
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        sums = np.zeros((num_classes, bbox_para_num)) # D
        squared_sums = np.zeros((num_classes, bbox_para_num)) # D
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                if cls_inds.size > 0:
                    class_counts[cls] += cls_inds.size
                    sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                    squared_sums[cls, :] += \
                            (targets[cls_inds, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0) # ignore bg class

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
		#print roidb[im_i]['bbox_targets'][cls_inds, 1:]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    else:
        print "NOT normalizing targets"

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs

    bbox_para_num = 5

    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs [cls, dx, dy, dh, dw, da]
        return np.zeros((rois.shape[0], bbox_para_num + 1), dtype=np.float32) # D  
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    #print "rois[ex_inds, :]", rois[ex_inds, :].shape
    #print "rois[gt_inds, :]", rois[gt_inds, :].shape
    ex_gt_overlaps = rbbx_overlaps(np.ascontiguousarray(rois[ex_inds, :], dtype=np.float32), np.ascontiguousarray(rois[gt_inds, :], dtype=np.float32),cfg.GPU_ID) # D

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], bbox_para_num + 1), dtype=np.float32) # D
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = rbbox_transform(ex_rois, gt_rois) # D
    #print targets[ex_inds, 1:]
    # targets: [cls, dx, dy, dh, dw, da]
    return targets

def select_hard_examples(loss):
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), cfg.TRAIN.BATCH_SIZE)]
    # (explore more ways of selecting examples in this function; e.g., sampling)

    return hard_keep_inds

def get_ohem_minibatch(loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)

    if cfg.TRAIN.OHEM_USE_NMS:
        # Do NMS using loss for de-dup and diversity
        keep_inds = []
        nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        source_img_ids = [roi[0] for roi in rois]
        for img_id in np.unique(source_img_ids):
            for label in np.unique(labels):
                sel_indx = np.where(np.logical_and(labels == label, \
                                    source_img_ids == img_id))[0]
                if not len(sel_indx):
                    continue
                boxes = np.concatenate((rois[sel_indx, 1:],
                        loss[sel_indx][:,np.newaxis]), axis=1).astype(np.float32)
                #keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])
                keep_inds.extend(sel_indx[rotate_gpu_nms(boxes, nms_thresh)])                    

        hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)

    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs   




































