
"""config system
"""
import numpy as np
import os.path
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Training options
__C.TRAIN = edict()
__C.TEST = edict()

# MNC/CFM mode
__C.GPU_ID = 0

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

__C.iters_numbers = 160000

__C.IMAGE_WIDTH = 512
__C.IMAGE_HEIGHT = 384

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# gt margin setting to using context information
__C.TRAIN.GT_MARGIN = 1.4
__C.TEST.GT_MARGIN = 1.4


__C.USE_GPU_NMS = True
__C.RNG_SEED = 3
__C.EPS = 1e-14
__C.PIXEL_MEANS = np.array([[[75.4916, 78.2288, 68.2117]]])



# ------- General setting ----
__C.TRAIN.IMS_PER_BATCH = 1
# Batch size for training Region CNN (not RPN)
__C.TRAIN.BATCH_SIZE = 128
# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = False
# Use flipped image for augmentation
__C.TRAIN.USE_FLIPPED = False # True

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.45

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.45
__C.TRAIN.BG_THRESH_LO = 0.0




# ------- Proposal -------
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# ------- BBOX Regression ---------
# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True #False
__C.TRAIN.BBOX_THRESH = 0.5
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# weight of smooth L1 loss
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
#__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1, 1, 1, 1)

# -------- RPN ----------
# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IO < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
# Note this is class-agnostic anchors' FG_FRACTION
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
#__C.TRAIN.RPN_NMS_THRESH = 0.7 #0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Mix anchors used for RPN and later layer
__C.TRAIN.MIX_INDEX = True

# Test option



# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
# Set this true in the yml file to specify proposed RPN
__C.TEST.HAS_RPN = True
# NMS threshold used on RPN proposals
__C.RPN_NMS_THRESH = 0.7 #0.85
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 600
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 8
__C.TEST.BBOX_REG = True


# Used for multi-scale testing, since naive implementation
# will waste a lot of on zero-padding, so we group each
# $GROUP_SCALE scales to feed in gpu. And max rois for
# each group is specified in MAX_ROIS_GPU
__C.TEST.MAX_ROIS_GPU = [2000]
__C.TEST.GROUP_SCALE = 1

# Parameters for "Online Hard-example Mining Algorithm"
__C.TRAIN.USE_OHEM = True
# For diversity and de-duplication
__C.TRAIN.OHEM_USE_NMS = True
__C.TRAIN.OHEM_NMS_THRESH = 0.7

# ---------------------------------Rotate related------------------------------------

__C.TRAIN.RPN_RBBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)

__C.TRAIN.RBBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0, 0.0)

__C.TRAIN.RBBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)

__C.TRAIN.RBBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

# We let the regression range between (-pi/4, pi/4) with uniform distribution, so the std is sqrt(pi^2 / 48)
# TODO statistic the angle distribution of dataset

__C.TRAIN.RBBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2, 1)

__C.TRAIN.R_POSITIVE_ANGLE_FILTER = 15
__C.TRAIN.R_NEGATIVE_ANGLE_FILTER = 15



# anchor setting on testing 
__C.TEST.RATIO_GROUP = []


    
def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.iteritems():
        # Since user_cfg is a sub-file of default_cfg
        if not default_cfg.has_key(key):
            raise KeyError('{} is not a valid config key'.format(key))

        if type(default_cfg[key]) is not type(val):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print 'Error under config key: {}'.format(key)
                raise
        else:
            default_cfg[key] = val


