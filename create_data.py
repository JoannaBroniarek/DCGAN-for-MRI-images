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


def load_img(file_path, subject_id=None):
    """ 
    Loads and preprocesses the traning image.
    Preprocessing: 
        - omit the initial/final slices
        - resize image to a smaller resolution 64x64
        - whitening
    """
    
    #  Construct a file path to read an image from.
    if subject_id is not None:
        t1_img = os.path.join(file_path, '{}/{}_t1.nii.gz'.format(subject_id, subject_id))
    else:
        t1_img = file_path
        
    # Read the .nii image containing a brain volume with SimpleITK and get 
    # the numpy array:
    sitk_t1 = sitk.ReadImage(t1_img)
    t1 = sitk.GetArrayFromImage(sitk_t1)

    # Select the slices from 50 to 125 among the whole 155 slices to omit initial/final slices, 
    # since they convey a negligible amount of useful information and could affect training
    t1 = t1[50:125]
  
    # Resize images to 64 x 64 from 240 x 240
    t1_new = np.zeros((t1.shape[0], 64, 64))
    for i in range(t1.shape[0]):
        t1_new[i] = cv2.resize(t1[i], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    # Normalise the image to zero mean/unit std dev:
    t1 = whitening(t1_new)
  
    # Create a 4D Tensor with a dummy dimension for channels
    t1 = np.moveaxis(t1, 0, -1)
  
    return t1



###########  Building the TFRecord file #############


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _decode(example_proto):
    feature_description = {'t1': tf.io.FixedLenFeature([], tf.string)}
    features = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.parse_tensor(features['t1'], out_type=tf.float32, name=None)
    return img

def parse_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    return raw_dataset.map(_decode)


def create():
    # open the TFRecords file
    train_filename = '../train2d.tfrecords'
    writer = tf.io.TFRecordWriter(train_filename)

    # Iterate through directories from the training dataset
    dataset_path = '../data/'  # os.chdir(dataset_path)
    counter = 1
    
    for subject_id in tqdm(os.listdir(dataset_path)):

        # Load the image
        img = load_img(dataset_path, subject_id)

        # Create a feature
        feature = {'t1': _bytes_feature(tf.io.serialize_tensor(img, name=None))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        counter += 1

    writer.close()
