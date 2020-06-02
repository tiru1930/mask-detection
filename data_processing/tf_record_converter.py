import os 
import sys
from tqdm import tqdm
import tensorflow as tf 
from absl import app, flags, logging
import xml.etree.ElementTree as ET
from config import config
import glob

class ConverttoTFRecord(object):
    """docstring for ConverttoTFRecord"""
    def __init__(self):

        super(ConverttoTFRecord, self).__init__()
        self.train_dataset_path = config.train_dataset_path
        self.test_dataset_path  = config.test_dataset_path
        self.train_tf_record_path = config.train_tf_record_path
        self.test_tf_record_path  = config.test_tf_record_path

    def processImage(self,image_path):
        try:
            image_string = tf.io.read_file(image_path)
            image_data = tf.image.decode_jpeg(image_string, channels=3)
            return image_string,image_data
        except Exception as e:
            print(image_path)
            

    def processAnnotations(self,annotation_file):
        try:
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            image_info = {}
            image_info_list = []

            file_name = root.find('filename').text

            xmin, ymin, xmax, ymax = [], [], [], []
            classes = []
            difficult = []

            for obj in root.iter('object'):
                
                label = obj.find('name').text

                if len(config.class_names) > 0 and label not in config.class_names:
                    continue
                else:
                    classes.append(config.class_names.index(label))

                if obj.find('difficult'):
                    difficult.append(int(obj.find('difficult').text))
                else:
                    difficult.append(0)

                for box in obj.findall('bndbox'):
                    xmin.append(float(box.find('xmin').text))
                    ymin.append(float(box.find('ymin').text))
                    xmax.append(float(box.find('xmax').text))
                    ymax.append(float(box.find('ymax').text))
        
            image_info['filename'] = file_name
            image_info['class'] = classes
            image_info['xmin'] = xmin
            image_info['ymin'] = ymin
            image_info['xmax'] = xmax
            image_info['ymax'] = ymax
            image_info['difficult'] = difficult

            image_info_list.append(image_info)

            return image_info_list

        except Exception as e:
            print(annotation_file)

    def make_record(self,image_string,image_info_list):
        try:

            for info in image_info_list:

                filename = info['filename']
                classes = info['class']
                xmin = info['xmin']
                ymin = info['ymin']
                xmax = info['xmax']
                ymax = info['ymax']

            if isinstance(image_string, type(tf.constant(0))):
                encoded_image = [image_string.numpy()]
            else:
                encoded_image = [image_string]

            base_name = [tf.compat.as_bytes(os.path.basename(filename))]


            example = tf.train.Example(features=tf.train.Features(feature={
                'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
                'classes':tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
                'x_mins':tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                'y_mins':tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                'x_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                'y_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
                'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
            }))

            return example
        except Exception as e:
            print(e)
            

    
    def process(self):
        
        images_file_list = glob.glob(self.train_dataset_path+"/*.jpg")

        with tf.io.TFRecordWriter(self.train_tf_record_path) as writer:
            for image_file in tqdm(images_file_list):
                try:
                    annotation_file = image_file.replace(".jpg",".xml")
                    image_string,image_data = self.processImage(image_file)
                    image_info_list = self.processAnnotations(annotation_file)
                    tf_example = self.make_record(image_string, image_info_list)
                    writer.write(tf_example.SerializeToString())
                except Exception as e:
                    print(e)
                    continue


        images_file_list = glob.glob(self.test_dataset_path+"/*.jpg")

        with tf.io.TFRecordWriter(self.test_tf_record_path) as writer:
            for image_file in tqdm(images_file_list):
                try:
                    annotation_file = image_file.replace(".jpg",".xml")
                    image_string,image_data = self.processImage(image_file)
                    image_info_list = self.processAnnotations(annotation_file)
                    tf_example = self.make_record(image_string, image_info_list)
                    writer.write(tf_example.SerializeToString())
                except Exception as e:
                    print(e)
                    continue
  
        
def main():
    cTF = ConverttoTFRecord()
    cTF.process()
    

if __name__ == '__main__':
    app.run(main)


    
