import numpy as np
import yaml

import caffe
from config import cfg
import cv2
from datasets.roidb import get_ohem_minibatch
from collections import Counter

class ProbBoxDataLayer(caffe.Layer):
    """
    Provide image, image w/h/scale, gt boxes, prob and prob info to upper layers
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(1, 3, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
        self._name_to_top_map['data'] = 0
        top[1].reshape(1, 2)
        self._name_to_top_map['im_info'] = 1
        top[2].reshape(1, 4)
        self._name_to_top_map['gt_boxes'] = 2
        top[3].reshape(1, cfg.IMAGE_HEIGHT / 2, cfg.IMAGE_WIDTH / 2)
        self._name_to_top_map['gt_prob'] = 3
        top[4].reshape(1, 1, cfg.IMAGE_HEIGHT / 2, cfg.IMAGE_WIDTH / 2)
        self._name_to_top_map['angle_label'] = 4      
        top[5].reshape(1, 1)
        self._name_to_top_map['cls_num'] = 5                    
        assert len(top) == len(self._name_to_top_map)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            if blob_name != 'cls_num':               
                top_ind = self._name_to_top_map[blob_name]
                # Reshape net's input blobs
                top[top_ind].reshape(*blob.shape)
                # Copy data into net's input blobs
                top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        top[5].reshape(1, 1)
        top[5].data[...] = blobs['cls_num']
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
    
    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
    
    def set_probdb(self, probdb):
        self._probdb = probdb
        self._shuffle_roidb_inds()
        
    def set_AngleLabeldb(self, AngleLabeldb):
        self._AngleLabeldb = AngleLabeldb
        self._shuffle_roidb_inds()    
    
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0       
        
        
    def _get_image_blob(self, roidb):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = 1  # len(roidb)
        processed_ims = []
        for i in xrange(num_images):
            im = cv2.imread(roidb['image'])
            im_resized = cv2.resize(im, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC) 
            im = im_resized
            im = im.astype(np.float32, copy=False)
            im -= cfg.PIXEL_MEANS
            processed_ims.append(im)
        # Create a blob to hold the input images
        # print "processed_ims shape is : {}".format(len(processed_ims))
        blob = np.zeros((1, im.shape[0], im.shape[1], 3),dtype=np.float32)
        blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob
    
    
    def _get_next_minibatch(self):
        """
        Return the blobs to be used for the next minibatch.
        """
        assert cfg.TRAIN.IMS_PER_BATCH == 1, 'Only single batch forwarding is supported'

        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur]
        self._cur += 1
        roidb = self._roidb[db_inds]
        
        im_blob = self._get_image_blob(roidb)

        gt_label = np.where(roidb['gt_classes'] != 0)[0]
        gt_boxes = np.hstack((roidb['boxes'][gt_label, :],
                              roidb['gt_classes'][gt_label, np.newaxis])).astype(np.float32)
        
        #####################################
        gt_boxes[:, -1] = 1
        
        blobs = {
            'data': im_blob,
            'gt_boxes': gt_boxes,
            'im_info': np.array([[im_blob.shape[2], im_blob.shape[3]]], dtype=np.float32)
        }

        probdb = self._probdb[db_inds]
        prob_list = probdb['gt_prob']
        gt_prob = np.zeros((len(prob_list), cfg.IMAGE_HEIGHT / 2, cfg.IMAGE_WIDTH / 2)) ################
        for j in xrange(len(prob_list)):
            prob = prob_list[j]
            prob = prob[0, :, :] #########################
            prob_x = prob.shape[1]
            prob_y = prob.shape[0]
            gt_prob[j, 0:prob_y, 0:prob_x] = prob
        blobs['gt_prob'] = gt_prob
             
        AngleLabeldb = self._AngleLabeldb[db_inds]
        angle_label = AngleLabeldb['angle_label']
#        print 'angle_label shape is: {}'.format(angle_label.shape)  
        blobs['angle_label'] = angle_label
              
        temp = angle_label[0, 0, :, :]
        num_angle_tuple = Counter(temp.ravel()).most_common(20)
        num_angle = len(num_angle_tuple) 
        num_temp = num_angle

        blobs['cls_num'] = num_temp                  
            
        return blobs 
    
class OHEMDataLayer(caffe.Layer):
    """Online Hard-example Mining Layer."""
    def setup(self, bottom, top):
        """Setup the OHEMDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_bottom_map = {
            'cls_prob_readonly': 0,
            'bbox_pred_readonly': 1,
            'rois': 2,
            'labels': 3}

        if cfg.TRAIN.BBOX_REG:
            self._name_to_bottom_map['bbox_targets'] = 4
            self._name_to_bottom_map['bbox_inside_weights'] = 5 ###?????????????????????????????????????????
            self._name_to_bottom_map['bbox_outside_weights'] = 6

        self._name_to_top_map = {}

#        assert cfg.TRAIN.HAS_RPN == False
        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(1, 6)
        self._name_to_top_map['rois_hard'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1)
        self._name_to_top_map['labels_hard'] = idx
        idx += 1

        if cfg.TRAIN.BBOX_REG:
            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[idx].reshape(1, self._num_classes * 5)
            self._name_to_top_map['bbox_targets_hard'] = idx
            idx += 1

            # bbox_inside_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[idx].reshape(1, self._num_classes * 5)
            self._name_to_top_map['bbox_inside_weights_hard'] = idx
            idx += 1

            top[idx].reshape(1, self._num_classes * 5)
            self._name_to_top_map['bbox_outside_weights_hard'] = idx
            idx += 1

        print 'OHEMDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        cls_prob = bottom[0].data
        bbox_pred = bottom[1].data
        rois = bottom[2].data
        labels = bottom[3].data
        if cfg.TRAIN.BBOX_REG:
            bbox_target = bottom[4].data
            bbox_inside_weights = bottom[5].data
            bbox_outside_weights = bottom[6].data
        else:
            bbox_target = None
            bbox_inside_weights = None
            bbox_outside_weights = None

        flt_min = np.finfo(float).eps
        # classification loss
        loss = [ -1 * np.log(max(x, flt_min)) \
            for x in [cls_prob[i,label] for i, label in enumerate(labels)]]

        if cfg.TRAIN.BBOX_REG:
            # bounding-box regression loss
            # d := w * (b0 - b1)
            # smoothL1(x) = 0.5 * x^2    if |x| < 1
            #               |x| - 0.5    otherwise
            def smoothL1(x):
                if abs(x) < 1:
                    return 0.5 * x * x
                else:
                    return abs(x) - 0.5

            bbox_loss = np.zeros(labels.shape[0])
            for i in np.where(labels > 0 )[0]:
                indices = np.where(bbox_inside_weights[i,:] != 0)[0]
                bbox_loss[i] = sum(bbox_outside_weights[i,indices] * [smoothL1(x) \
                    for x in bbox_inside_weights[i,indices] * (bbox_pred[i,indices] - bbox_target[i,indices])])
            loss += bbox_loss

        blobs = get_ohem_minibatch(loss, rois, labels, bbox_target, \
            bbox_inside_weights, bbox_outside_weights)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
