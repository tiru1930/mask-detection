import math
import sys
import cv2
import logging 
import tensorflow as tf
import numpy as np
from config import config
from itertools import product

class dataProcessUtils(object):
    """docstring for dataProcessUtils"""
    def __init__(self):
        super(dataProcessUtils, self).__init__()
        self.arg = None

    def transform_center_to_corner(self,boxes):

        return tf.concat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), axis=1)

    def intersect(self,box_a,box_b):
        A = tf.shape(box_a)[0]
        B = tf.shape(box_b)[0]
        max_xy = tf.minimum(
            tf.broadcast_to(tf.expand_dims(box_a[:, 2:], 1), [A, B, 2]),
            tf.broadcast_to(tf.expand_dims(box_b[:, 2:], 0), [A, B, 2]))
        min_xy = tf.maximum(
            tf.broadcast_to(tf.expand_dims(box_a[:, :2], 1), [A, B, 2]),
            tf.broadcast_to(tf.expand_dims(box_b[:, :2], 0), [A, B, 2]))
        inter = tf.clip_by_value(max_xy - min_xy, 0.0, 512.0)
        return inter[:, :, 0] * inter[:, :, 1]
        

    def jaccard(self,box_a,box_b):
      
        inter = self.intersect(box_a,box_b)
        inter = self.intersect(box_a, box_b)
        area_a = tf.broadcast_to(
            tf.expand_dims(
                (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), 1),
            tf.shape(inter))  
        area_b = tf.broadcast_to(
            tf.expand_dims(
                (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), 0),
            tf.shape(inter))  
        union = area_a + area_b - inter
        return inter / union 

    def encode_bbox(self,matched, priors, variances):
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = tf.math.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return tf.concat([g_cxcy, g_wh], 1)  # [num_priors,4]
        
    def encode_tf_label(self,labels ,priors,match_thresh=0.5,variances=[0.1,0.2]):
        try:
            priors = tf.cast(priors,tf.float32)
         
            bboxs   = labels[:,:4]
            conf   = labels[:,-1]
  
            priors_boxes = self.transform_center_to_corner(priors)
            overlap_boxes = self.jaccard(bboxs,priors_boxes)
            
            best_prior_overlap_boxes = tf.reduce_max(overlap_boxes,1)
            best_prior_overlap_boxes_index = tf.argmax(overlap_boxes,1,tf.int32)

            best_truth_overlap = tf.reduce_max(overlap_boxes,0)
            best_truth_idx     = tf.argmax(overlap_boxes,0,tf.int32) 

            best_truth_overlap = tf.tensor_scatter_nd_update(
                                            best_truth_overlap, 
                                            tf.expand_dims(best_prior_overlap_boxes_index, 1),
                                            tf.ones_like(best_prior_overlap_boxes_index , tf.float32) * 2.)
            best_truth_idx = tf.tensor_scatter_nd_update(
                                            best_truth_idx, 
                                            tf.expand_dims(best_prior_overlap_boxes_index, 1),
                                            tf.range(tf.size(best_prior_overlap_boxes_index), dtype=tf.int32))

            matches_bbox = tf.gather(bboxs, best_truth_idx)

            loc_t = self.encode_bbox(matches_bbox, priors, variances)

            conf_t = tf.gather(conf, best_truth_idx)  # [num_priors]
            conf_t = tf.where(tf.less(best_truth_overlap, match_thresh), tf.zeros_like(conf_t), conf_t)

            return tf.concat([loc_t, conf_t[..., tf.newaxis]], axis=1)

        except Exception as e:
            raise e

    def generate_prior_boxes(self):

        image_sizes = config.imgae_dim
        min_sizes   = config.min_sizes
        steps       = config.steps
        clip        = config.clip

        if isinstance(image_sizes, int):
            image_sizes = (image_sizes, image_sizes)
        elif isinstance(image_sizes, list):
            image_sizes = image_sizes
        else:
            logging.info('Type error of input image size format,tuple or int. ')
            sys.exit()

        for m in range(4):
            if (steps[m] != pow(2, (m + 3))):
                logging.info("steps must be [8,16,32,64]")
                

        assert len(min_sizes) == len(steps), "anchors number didn't match the feature map layer."

        feature_maps = [[math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)] for step in steps]

        anchors = []
        num_box_fm_cell=[]
        for k, f in enumerate(feature_maps):
            num_box_fm_cell.append(len(min_sizes[k]))
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes[k]:
                    if isinstance(min_size, int):
                        min_size = (min_size, min_size)
                    elif isinstance(min_size, tuple):
                        min_size=min_size
                    else:
                        logging.info('Type error of min_sizes elements format,tuple or int. ')
                    s_kx = min_size[1] / image_sizes[1]
                    s_ky = min_size[0] / image_sizes[0]
                    cx = (j + 0.5) * steps[k] / image_sizes[1]
                    cy = (i + 0.5) * steps[k] / image_sizes[0]
                    anchors += [cx, cy, s_kx, s_ky]

        output = np.asarray(anchors).reshape([-1, 4])

        if clip:
            output = np.clip(output, 0, 1)
        return output,num_box_fm_cell

    def decode_bbox_tf(self,bbox, priors, variances =[0.1,0.2]):
  
        boxes = np.concatenate(
            (priors[:, :2] + bbox[:, :2] * variances[0] * priors[:, 2:],
             priors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def compute_nms(self,boxes, scores, nms_threshold=0.5, limit=200):
        """ Perform Non Maximum Suppression algorithm
            to eliminate boxes with high overlap
        Args:
            boxes: tensor (num_boxes, 4)
                   of format (xmin, ymin, xmax, ymax)
            scores: tensor (num_boxes,)
            nms_threshold: NMS threshold
            limit: maximum number of boxes to keep
        Returns:
            idx: indices of kept boxes
        """
        if boxes.shape[0] == 0:
            return tf.constant([], dtype=tf.int32)
        selected = [0]
        idx = tf.argsort(scores, direction='DESCENDING')
        idx = idx[:limit]
        boxes = tf.gather(boxes, idx)

        iou = self.jaccard(boxes, boxes)

        while True:
            row = iou[selected[-1]]
            next_indices = row <= nms_threshold

            # iou[:, ~next_indices] = 1.0
            iou = tf.cast(iou,tf.float32)
            iou = tf.where(
                tf.expand_dims(tf.math.logical_not(next_indices), 0),
                tf.ones_like(iou, dtype=tf.float32),
                iou)

            if not tf.math.reduce_any(next_indices):
                break

            selected.append(tf.argsort(
                tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

        return tf.gather(idx, selected)
