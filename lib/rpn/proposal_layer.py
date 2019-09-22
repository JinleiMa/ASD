import caffe
import numpy as np
import yaml

from generate_anchors import generate_anchors
from config import cfg
import numpy.random as npr
from nms.rotate_polygon_nms import rotate_gpu_nms

DEBUG = False # False True

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated anchors.
    """

    def setup(self, bottom, top):
        
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._bbox_para_num = 5 # parameter number of bbox
        
        base_size = 16
        self.base_anchor_center = 7.5 
        self.scale = 2
        self.min_num = 100 
        
        ratios = [0.25, 0.125]
        scales=np.array((2.0, 4.0, 8.0, 16.0)) # * 16 = (32, 64, 128, 256)
        angle = [0.0]
        self.anchors_base = generate_anchors(base_size, ratios, scales, angle)
        
        self.anchors_base_angle_one = generate_anchors(base_size, ratios, scales, angle = [-37.5, -30.0, -22.5])
        self.anchors_base_angle_two = generate_anchors(base_size, ratios, scales, angle = [-7.5, 0.0, 7.5])
        self.anchors_base_angle_three = generate_anchors(base_size, ratios, scales, angle = [22.5, 30.0, 37.5])
        self.anchors_base_angle_four = generate_anchors(base_size, ratios, scales, angle = [52.5, 60.0, 67.5])
        self.anchors_base_angle_five = generate_anchors(base_size, ratios, scales, angle = [82.5, 90.0, 97.5])
        self.anchors_base_angle_six = generate_anchors(base_size, ratios, scales, angle = [112.5, 120.0, 127.5])
        
        
        
        
        # rois blob: holds R regions of interest, each is a 6-tuple
        # (n, ctr_x, ctr_y, h, w, theta) specifying an image batch index n and a
        # rectangle (ctr_x, ctr_y, h, w, theta)
        top[0].reshape(1, self._bbox_para_num + 1) # D

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1, 1) # D
        
    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'  
            
        cfg_key = str('TRAIN' if self.phase == 0 else 'TEST')
#        min_size      = cfg[cfg_key].RPN_MIN_SIZE
#        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg.RPN_NMS_THRESH
                           
        
        angle_label_small = bottom[1].data[0].argmax(axis=0) 
        label_loc_small = np.where(angle_label_small > 0.5)    
        label_loc = tuple(np.array(label_loc_small) * self.scale)            
        
        angle_label = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH), dtype = np.uint8)
        angle_label[label_loc] = angle_label_small[label_loc_small]     


        prob = bottom[0].data[0].argmax(axis=0)  

        center_num_tuple = np.where(prob > 0.5)    
        center_num = len(center_num_tuple[0])
        if DEBUG:
            print 'center_num is: {}'.format(center_num) 
#            print prob
            
        prob_probability = bottom[0].data[0, 1, :, :]
        
        
        ###################################################################                       
        wid = prob.shape[1]
        order = prob.ravel().argsort()[::-1]
        if center_num >= self.min_num:
            order_random = npr.choice(order[:center_num], size=self.min_num, replace=False)
            row_center = order_random / wid
            column_center = order_random % wid
            row_center *= self.scale
            column_center *= self.scale
        else:
            order_random = prob_probability.ravel().argsort()[::-1]
            order_random = order_random[:self.min_num]
            row_center = order_random / wid
            column_center = order_random % wid
            row_center *= self.scale
            column_center *= self.scale
             
        
        ###################################################################     
        angle_label_loc = np.where(angle_label > 0.5)
        angle_label_num = len(angle_label_loc[0])
        if DEBUG:
            print 'angle_label_num is: {}'.format(angle_label_num)


        angle_index = np.zeros((len(row_center), ), dtype = np.uint8)

        for idx in xrange(len(row_center)):
            row_idx = row_center[idx]
            column_idx = column_center[idx]
            
            angle_index[idx] = angle_label[row_idx, column_idx]
        
        ################################################################### 
        if angle_label_num < 1:
            index_tuple = [row_center, column_center]    
            index_tuple = tuple(index_tuple) 
            
            K = len(row_center)
            A = self.anchors_base.shape[0]
            anchors = np.zeros((K * A, 6), dtype=np.float32)
            
            for i in xrange(len(index_tuple[0])):
                shift_y = index_tuple[0][i] - self.base_anchor_center # height row
                shift_x = index_tuple[1][i] - self.base_anchor_center # width column
                anchors[A * i : A * i + A, 0] = self.anchors_base[:, 0] + shift_x
                anchors[A * i : A * i + A, 1] = self.anchors_base[:, 1] + shift_y
                anchors[A * i : A * i + A, 2] = self.anchors_base[:, 2]  
                anchors[A * i : A * i + A, 3] = self.anchors_base[:, 3]  
                anchors[A * i : A * i + A, 4] = self.anchors_base[:, 4] 
                anchors[A * i : A * i + A, 5] = prob_probability[index_tuple[0][i] / self.scale,index_tuple[1][i] / self.scale]
                
        #############################################################################################                     
        else:
            num_match = np.sum(angle_index > 0)
            num_nomatch = np.sum(angle_index == 0)    
            base_loc = np.where(angle_index == 0)
            angle_loc = np.where(angle_index > 0)
            
            
        ############################################################################################

            A1 = self.anchors_base.shape[0]
            A2 = self.anchors_base_angle_one.shape[0]
            #K = num_match * A2 + num_nomatch * A1 + len(grid_angle_loc) * A2
            K = num_match * A2 + num_nomatch * A1                                          
            anchors = np.zeros((K, 6), dtype=np.float32)                   
                           
            for i in xrange(num_nomatch):
                shift_y = row_center[base_loc[0][i]] - self.base_anchor_center # height row
                shift_x = column_center[base_loc[0][i]] - self.base_anchor_center # width column
                anchors[A1 * i : A1 * i + A1, 0] = self.anchors_base[:, 0] + shift_x
                anchors[A1 * i : A1 * i + A1, 1] = self.anchors_base[:, 1] + shift_y
                anchors[A1 * i : A1 * i + A1, 2] = self.anchors_base[:, 2]  
                anchors[A1 * i : A1 * i + A1, 3] = self.anchors_base[:, 3]  
                anchors[A1 * i : A1 * i + A1, 4] = self.anchors_base[:, 4] 
                anchors[A1 * i : A1 * i + A1, 5] = prob_probability[row_center[base_loc[0][i]] / self.scale,column_center[base_loc[0][i]] / self.scale]
            for i in xrange(num_match):
                shift_y = row_center[angle_loc[0][i]] - self.base_anchor_center # height row
                shift_x = column_center[angle_loc[0][i]] - self.base_anchor_center # width column
                if angle_index[angle_loc[0][i]] == 1:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_one[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_one[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_one[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_one[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_one[:, 4]
                if angle_index[angle_loc[0][i]] == 2:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_two[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_two[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_two[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_two[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_two[:, 4]
                if angle_index[angle_loc[0][i]] == 3:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_three[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_three[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_three[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_three[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_three[:, 4]
                if angle_index[angle_loc[0][i]] == 4:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_four[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_four[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_four[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_four[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_four[:, 4]
                if angle_index[angle_loc[0][i]] == 5:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_five[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_five[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_five[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_five[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_five[:, 4]
                if angle_index[angle_loc[0][i]] == 6:
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 0] = self.anchors_base_angle_six[:, 0] + shift_x
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 1] = self.anchors_base_angle_six[:, 1] + shift_y
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 2] = self.anchors_base_angle_six[:, 2]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 3] = self.anchors_base_angle_six[:, 3]
                    anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 4] = self.anchors_base_angle_six[:, 4]
                anchors[A1 * num_nomatch + A2 * i : A1 * num_nomatch + A2 * i + A2, 5] = \
                         prob_probability[row_center[angle_loc[0][i]] / self.scale,column_center[angle_loc[0][i]] / self.scale]  
    
 #####################################################################################################
        
#        print 'anchors A: {}'.format(A)
#        print 'anchors K: {}'.format(K)
#        print 'index_tuple : {}'.format(index_tuple)
#        print 'anchors shape is: {}'.format(anchors.shape)
#        print 'prob min is: {}'.format(anchors[:,-1].min())

        if DEBUG:
            print 'anchors shape is: {}'.format(anchors.shape)
            
        
        proposals = anchors[:, :-1]
        
#        print 'proposals num is: {}'.format(proposals.shape)

        scores = np.zeros((anchors.shape[0], 1), dtype=np.float32)
        scores[:,0] = anchors[:, -1]
        
        if DEBUG:
            print 'initial proposals shape is: {}'.format(proposals.shape)
            print 'scores shape is: {}'.format(scores.shape)
        
        keep = rotate_gpu_nms(np.hstack((proposals, scores)).astype(np.float32), nms_thresh, cfg.GPU_ID) # D
        
        if DEBUG:
            print 'after nms proposals num is: {}'.format(len(keep))

        post_nms_topN = min(post_nms_topN, len(keep))
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        
        if DEBUG:
            print 'final proposals num is: {}'.format(proposals.shape)
        
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
        
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass




