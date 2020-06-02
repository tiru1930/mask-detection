import os 


train_dataset_path = "/home/tbaggu/Desktop/face-mask-detection/face-mask-detection/data/FaceMaskDataset/train/"
test_dataset_path  = "/home/tbaggu/Desktop/face-mask-detection/face-mask-detection/data/FaceMaskDataset/val/"
train_tf_record_path = "/home/tbaggu/Desktop/face-mask-detection/mask-detection/datasets/train.tfrecord"
test_tf_record_path  = "/home/tbaggu/Desktop/face-mask-detection/mask-detection/datasets/test.tfrecord"
class_names          = ['background', 'face', 'face_mask']
batch_size           = 8 
imgae_dim            = [240, 320]
match_thresh         = 0.5
variances            = [0.1, 0.2]
min_sizes			 = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
steps				 = [8, 16, 32, 64]
clip 			     = False
base_channel         = 16 

train_dataset_len    = 6115
test_dataset_len     = 1839

resume  =  False  
epoch   =  100
init_lr =  1e-2
lr_decay_epoch = [50, 70]
lr_rate = 0.1
warmup_epoch = 5
min_lr = 1e-4

weights_decay = 5e-4
momentum = 0.9
save_freq = 1 

# # inference
score_threshold = 0.5
nms_threshold = 0.4
max_number_keep = 200