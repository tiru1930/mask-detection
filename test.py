from config import config
from model.network import slimModel
from utils.data_utils import dataProcessUtils
from data_processing.tf_record_process import tfrecord_processer


from tqdm import tqdm
import logging
import sys 
import tensorflow as tf 
import cv2
import numpy as np 
import time

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_prediction(predictions, priors,data_process):

    label_classes = config.class_names

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    
    boxes = data_process.decode_bbox_tf(bbox_regressions, priors,config.variances)

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]
        score_idx = cls_scores > config.score_threshold

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]
    
        nms_idx = data_process.compute_nms(cls_boxes, cls_scores, config.nms_threshold, config.max_number_keep)

        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)

        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores

def main():

    data_process            = dataProcessUtils()
    priors,number_of_cells  = data_process.generate_prior_boxes()

    TFp = tfrecord_processer(data_process,priors)
    test_dataset = TFp.tf_loader(is_test=True)

    logging.info("building model")
    sModel = slimModel(number_of_cells)
    model  = sModel.build_model()
    logging.info(model.summary())
    model.load_weights("model.h5")
    logging.info("model loading done")
    logging.info("predictions on test_dataset")

    try:
    
        for image,labels in test_dataset:
            predictions = model(image)
            boxes, classes, scores = parse_prediction(predictions, priors, data_process)
            logging.info("{} {}".format(classes,scores))
    
    except Exception as e:
        pass 



if __name__ == '__main__':
    main()