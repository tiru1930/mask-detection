from config import config
from model.network import slimModel
from utils.data_utils import dataProcessUtils


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

def show_image(img, boxes, classes, scores, img_height, img_width, prior_index, class_list):
    """
    draw bboxes and labels
    out:boxes,classes,scores
    """
    # bbox

    x1, y1, x2, y2 = int(boxes[prior_index][0] * img_width), int(boxes[prior_index][1] * img_height), \
                     int(boxes[prior_index][2] * img_width), int(boxes[prior_index][3] * img_height)
    if classes[prior_index] == 1:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # confidence

    score = "{:.4f}".format(scores[prior_index])
    class_name = class_list[classes[prior_index]]

    cv2.putText(img, '{} {}'.format(class_name, score),
                (int(boxes[prior_index][0] * img_width), int(boxes[prior_index][1] * img_height) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    

def main():

    data_process            = dataProcessUtils()
    priors,number_of_cells  = data_process.generate_prior_boxes()


    logging.info("loading model")
    sModel = slimModel(number_of_cells)
    model  = sModel.build_model()
    model.load_weights("model.h5")
    logging.info("model loading done")

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start = time.time()
    while True:

        _,frame = capture.read()
        if frame is None:
            logging.error('No camera found')

        h,w,_ = frame.shape
        img = np.float32(frame.copy())

        img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img = img / 255.0 - 0.5

        predictions = model(img[np.newaxis, ...])
        boxes, classes, scores = parse_prediction(predictions, priors, data_process)      

        for prior_index in range(len(classes)):
            show_image(frame, boxes, classes, scores, h, w, prior_index,config.class_names)
       
        fps_str = "FPS: %.2f" % (1 / (time.time() - start))
        start = time.time()
        cv2.putText(frame, fps_str, (25, 25),cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            exit()



if __name__ == '__main__':
    main()