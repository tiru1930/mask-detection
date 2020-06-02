import tensorflow as tf
from utils.data_utils import dataProcessUtils
from config import config
import logging 

class tfrecord_processer(object):
    """docstring for tfrecord_processer"""
    def __init__(self,data_process,priors):
        super(tfrecord_processer, self).__init__()

        self.variances  = config.variances
        self.batch_size = config.batch_size
        self.image_dim  = config.imgae_dim
        self.match_thresh = config.match_thresh
        self.data_process = data_process
        self.priors      = priors
        logging.info("priors shape{}".format(self.priors.shape)) 

    def tf_decode_parser(self,each_tf_record):

        features = {

            'filename': tf.io.FixedLenFeature([], tf.string),
            'classes': tf.io.VarLenFeature(tf.int64),
            'x_mins': tf.io.VarLenFeature(tf.float32),
            'y_mins': tf.io.VarLenFeature(tf.float32),
            'x_maxes': tf.io.VarLenFeature(tf.float32),
            'y_maxes': tf.io.VarLenFeature(tf.float32),
            'difficult':tf.io.VarLenFeature(tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        parsed_features = tf.io.parse_single_example(each_tf_record,features)

        image = tf.image.decode_jpeg(parsed_features['image_raw'], channels=3)

        labels = tf.sparse.to_dense(parsed_features['classes'])
        
        labels = tf.cast(labels, tf.float32)

        labels = tf.stack([
                        tf.sparse.to_dense(parsed_features['x_mins']),
                        tf.sparse.to_dense(parsed_features['y_mins']),
                        tf.sparse.to_dense(parsed_features['x_maxes']),
                        tf.sparse.to_dense(parsed_features['y_maxes']),
                        labels], axis=1)
   
        return image,labels

    def transformers(self,image,labels):

        image  = tf.cast(image,tf.float32)
        image  = (image/255.0-0.5)/1.0

        image ,labels = self.resize(image,labels)
        
        labels = self.data_process.encode_tf_label(labels,self.priors)
        return image,labels

    def resize(self,image,labels):
    
            w_f     = tf.cast(tf.shape(image)[1], tf.float32)
            h_f     = tf.cast(tf.shape(image)[0], tf.float32)
            locs    = tf.stack([labels[:, 0] / w_f,  labels[:, 1] / h_f,
                             labels[:, 2] / w_f,  labels[:, 3] / h_f] ,axis=1)
            locs    = tf.clip_by_value(locs, 0, 1.0)
            labels  = tf.concat([locs, labels[:, 4][:, tf.newaxis]], axis=1)

            img     = tf.image.resize(image, [self.image_dim[0], self.image_dim[1]])

            return img,labels
               

    def tf_loader(self,shuffle=True, repeat=True,buffer_size=10240,is_test=False):
        if is_test:
            logging.info("loading tf_recorder {}".format(config.test_tf_record_path))
            raw_dataset = tf.data.TFRecordDataset(config.test_tf_record_path)
        else:
            logging.info("loading tf_recorder {}".format(config.train_tf_record_path))
            raw_dataset = tf.data.TFRecordDataset(config.train_tf_record_path)
        raw_dataset = raw_dataset.cache()
        if repeat:
            raw_dataset = raw_dataset.repeat()
        if shuffle:
            raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)

        dataset = raw_dataset.map(self.tf_decode_parser)
        dataset = dataset.map(self.transformers)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

