import SimpleITK as sitk
import os
import tensorflow as tf
import pandas as pd
import cv2
from tqdm import tqdm

from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

import glob
import imageio
import PIL
import time

from tensorflow.keras.utils import Progbar

from models import *
from create_data import *
from utils import *


import argparse



"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow 2.1 implementation of DCGAN for 3D MRI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epochs', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--lr_g', type=float, default=5e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=5e-5, help='Discriminator learning rate')
    parser.add_argument('--data_path', type=str, default="../train.tfrecords", help='path to train data')
    parser.add_argument('--createTFrecords', type=bool, default=False, help='whether to create a TF record file')
    parser.add_argument('--rand_seed', type=int, default=42, help='tf random seed')
    parser.add_argument('--restore', type=str, default=None, help='path restore model path')

    return parser.parse_args()

@tf.function(autograph=True)
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    batch_size = int(len(images))
    noise = np.random.uniform(-1, 1, size=(batch_size, 100))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(train_type, 
          gen_lr, disc_lr,
          dataset, num_training_samples,
          epochs, batch_size,
          save_dir_path):
    
    weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0, seed=42)
    
    if train_type=='3D':
        generator=generator3d(weight_initializer=weight_initializer)
        print(generator.summary())

        discriminator=discriminator3d(weight_initializer=weight_initializer)
        print(discriminator.summary())
    
    gen_init_learning_rate = gen_lr
    gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        gen_init_learning_rate,
        decay_steps=600,
        decay_rate=0.2,
        staircase=True)
    
    disc_init_learning_rate = disc_lr
    disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        disc_init_learning_rate,
        decay_steps=1000,
        decay_rate=0.3,
        staircase=True)

    generator_optimizer=tf.keras.optimizers.Adam(gen_lr_schedule, beta_1 = 0.5)
    discriminator_optimizer=tf.keras.optimizers.Adam(disc_lr_schedule, beta_1 = 0.5)
    #discriminator_optimizer = tf.keras.optimizers.SGD(disc_lr)

    BUFFER_SIZE = 6000
  
    writer = tf.summary.create_file_writer(os.path.join(save_dir_path, 'logs'))

    checkpoint_dir = save_dir_path
    checkpoint_prefix = os.path.join("training_checkpoints", "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                              discriminator_optimizer=discriminator_optimizer,
                              generator=generator,
                              discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint,
                                      directory=checkpoint_dir,
                                      max_to_keep = 10,
                                      checkpoint_name=checkpoint_prefix)

    try:
        status = checkpoint.restore(manager.latest_checkpoint)
        restored_epoch_number = int(manager.latest_checkpoint.split("-")[-1])*5
    except:
        restored_epoch_number = 0
    print("restoring checkpoint {}".format(restored_epoch_number))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size)
    
    #global_step = restored_epoch_number*batch_size
    # keep batch size the same when restoring checkpoints
    global_step = 0
    for epoch in range(restored_epoch_number, epochs):
        print("\nepoch {}/{}".format(epoch+1,epochs))
        pb_i = Progbar(target=num_training_samples, verbose=1)
    
        for image_batch in dataset.as_numpy_iterator():
            gen_loss, disc_loss = train_step(image_batch,
                                           generator, discriminator,
                                           generator_optimizer, discriminator_optimizer)

            pb_i.add(image_batch.shape[0], values=[('gen_loss', gen_loss), ('disc_loss', disc_loss)]) #to do: average over epoch
            with writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=global_step)
                tf.summary.scalar('disc_loss', disc_loss, step=global_step)
                writer.flush()
            global_step+=1

        # Save the model every 5 epochs
        if (epoch + 1) % 20 == 0:
            manager.save()
        if (epoch + 1) % 5 == 0:
            # Produce images for the GIF as we go
            test_noise = np.random.uniform(-1, 1, size=(1, 100))
            gen_and_save_images(generator, epoch + 1, test_noise, save_dir_path, gen_loss, disc_loss, False)

    # # Generate after the final epoch
    #display.clear_output(wait=True)
    test_noise = np.random.uniform(-1, 1, size=(1, 100))
    gen_and_save_images(generator, epoch + 1, test_noise, save_dir_path, gen_loss, disc_loss, False)


    
def main(args):
    
    tf.random.set_seed(args.rand_seed)
    
    #### create file 
    if args.createTFrecords:
        create()

    ################################################################################
    ## Load Dataset

    train_filename = args.data_path
    parsed_dataset = parse_dataset(train_filename)
    num_training_examples = sum(1 for _ in tf.data.TFRecordDataset(train_filename))

    ############################## Train #############################3

    tf.keras.backend.clear_session()
    if args.restore:
        model_name = args.restore
    else:
        model_name = time.strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(model_name):
          os.mkdir(model_name)
    print("saving images in: {}".format(model_name))

    ########### Params #####################
    disc_lr = args.lr_d
    gen_lr = args.lr_g
    noise_dim = 100
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    ##################################3

    train(train_type = '3D',
          disc_lr = disc_lr,
          gen_lr = gen_lr,
          dataset=parsed_dataset,
          num_training_samples = num_training_examples, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE,
          save_dir_path = model_name)
    

if __name__ == "__main__":
    #tf.app.run(main=main)
    args = parse_args()
    if args is None:
        exit()
    main(args)