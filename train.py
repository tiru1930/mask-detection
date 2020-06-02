from data_processing.tf_record_converter import ConverttoTFRecord
from data_processing.tf_record_process import tfrecord_processer
from model.network import slimModel
from model.losses import MultiBoxLoss
from utils.data_utils import dataProcessUtils
from config import config
from utils.lr_scheduler import MultiStepWarmUpLR

from tqdm import tqdm
import logging
import sys 
import tensorflow as tf 

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def main():

    cTF = ConverttoTFRecord()
    cTF.process()

    data_process            = dataProcessUtils()
    priors,number_of_cells  = data_process.generate_prior_boxes()

    TFp = tfrecord_processer(data_process,priors)
    train_dataset = TFp.tf_loader()

    logging.info("building model")
    sModel = slimModel(number_of_cells)
    model  = sModel.build_model()
    # logging.info(model.summary())
   
    steps_per_epoch = config.train_dataset_len // config.batch_size
    logging.info("steps_per_epoch :{}".format(steps_per_epoch))

    logging.info("loss functions and optimzers intilizations")

    loss = MultiBoxLoss()

    learning_rate = MultiStepWarmUpLR(
                            initial_learning_rate=config.init_lr,
                            lr_steps=[e * steps_per_epoch for e in config.lr_decay_epoch],
                            lr_rate=config.lr_rate,
                            warmup_steps=config.warmup_epoch * steps_per_epoch,
                            min_lr=config.min_lr)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=config.momentum, nesterov=True)

    train_log_dir = 'logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)  #unused. Init for redefine network
            losses['loc'], losses['class'] = loss.multi_loss(labels, predictions)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, losses

    logging.info("Training started ")
    for epoch in tqdm(range(1,config.epoch)):

        avg_loss = 0.0
        for step, (inputs, labels) in enumerate(train_dataset.take(steps_per_epoch)):

            total_loss, losses = train_step(inputs, labels)
            avg_loss = (avg_loss * step + total_loss.numpy()) / (step + 1)
            steps =steps_per_epoch*epoch+step

            with train_summary_writer.as_default():
                tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)

            logging.info("step :{}, total_loss :{} location_loss :{} class_loss :{} ".format(step,
                                                                                        total_loss,
                                                                                        losses["loc"],
                                                                                        losses["class"]))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss/avg_loss',avg_loss,step=epoch)

        model.save_weights("model.h5")
         

if __name__ == '__main__':
    main()