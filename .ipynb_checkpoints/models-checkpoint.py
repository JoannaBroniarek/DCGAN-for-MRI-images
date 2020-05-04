import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     Conv3D,
                                     Dense,
                                     Conv2DTranspose,
                                     Conv3DTranspose,
                                     Reshape,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Activation, 
                                     Flatten,
                                     Dropout)
from tensorflow.keras import Sequential



#################################################
####        Generator & Discriminator        ####
#################################################


def generator3d(img_shape=(75, 64, 64, 1),
                noise_shape = (100, ),
                kernel_size = (4, 4, 4),
                strides = (1, 2, 2),
                upsample_layers = 4,
                starting_filters = 512,
                weight_initializer = None):
    
    """
    Generator3D for a Deep Convolutional GAN.
    
    The generator uses Transposed Convolution Layers (upsampling) to generate the 3D image from random noise.
    Models starts with a Dense layer and then upsamples few times in order to reach the desired image size.
    The ELu activation was used after Dense layer and the ReLU activation after the next convolutional layers 
    except from the last one. The Batch Normalization techinique was used, as well.
    
    The number of output filters from each convolutional layer is decreased twice for each new layer, 
    except for the last layer, where the only one filter is returned. Padding was set as 'same'.
    
    Parameters
    ----------
    img_shape : {tuple of 4 elements} - 
            Shape of an input image. Default shape is (75, 64, 64, 1).
    noise_shape : {tuple} -
            Shape of a noise vector. Default shape is (100, ).
    kernel_size : {an integer or tuple/list of 3 integers} - 
            Specifies the depth, height and width of the 3D convolution window. 
            Can be a single integer to specify the same value for all spatial dimensions.. 
            Default is (4, 4, 4).
    strides : {an integer or tuple/list of 3 integers} -
            Specifies the strides of the convolution along the depth, height and width. 
            Can be a single integer to specify the same value for all spatial dimensions.
            Default is (1, 2, 2).
    upsample_layers : {int} - 
            Number of the Transposed Convolution Layers. 
            Default is 4.
    starting_filters : {integer} - 
            Dimensionality of the output space after the initial dense layer.
            Default is 512.
    weight_initializer  - Kernel initializer for a dense layer.

    """

    filters = starting_filters

    model = Sequential()
    model.add(
      Dense(np.int32(starting_filters * (img_shape[0])  * (img_shape[1] / (2 ** upsample_layers)) * (img_shape[2] / (2 ** upsample_layers))),
            input_shape=noise_shape, kernel_initializer=weight_initializer)) #use_bias=False
    model.add(BatchNormalization())
    model.add(Activation("elu"))

    model.add(Reshape(((img_shape[0]),
                     np.int((img_shape[1] // (2 ** upsample_layers))),
                     np.int((img_shape[2] // (2 ** upsample_layers))),
                     starting_filters)))

    ## 3 Hidden Convolution Layers
    for l in range(upsample_layers-1):
        filters = int(filters/2)
        model.add(Conv3DTranspose(filters, kernel_size, strides,
                                  padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        #model.add(LeakyReLU())

    ## 4th Convolution Layer
    model.add(Conv3DTranspose(1, kernel_size, strides, 
                            padding='same', use_bias=False))
#     model.add(Activation("tanh"))

    return model



def generator2d(img_shape=(64, 64, 75),
                noise_shape = (100, ),
                kernel_size = (4, 4),
                strides = (2, 2),
                upsample_layers = 4,
                starting_filters = 512):
    """
    Generator2D for a Deep Convolutional GAN.
    """
    
    filters = starting_filters
    starting_img_size = img_shape[1] //(2**upsample_layers)
  
    model = Sequential()
    model.add(Dense((starting_img_size ** 2) * starting_filters, 
                  input_shape=noise_shape, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("elu"))

    model.add(Reshape((starting_img_size, starting_img_size,filters)))

    ## 3 Hidden Convolution Layers
    for l in range(upsample_layers-1):
        filters = int(filters / 2)
        model.add(Conv2DTranspose(filters, kernel_size, strides,
                              padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("elu"))
        #model.add(LeakyReLU())
  
    ## 4th Convolution Layer
    model.add(Conv2DTranspose(img_shape[-1], kernel_size, strides, 
                            padding='same', use_bias=False))
    


def discriminator3d(input_shape=(75, 64, 64, 1),
                    kernel_size = (4, 4, 4),
                    strides = (1, 2, 2),
                    downsample_layers = 4,
                    weight_initializer = None):
    """
    Discriminator3D for a Deep Convolutional GAN.
    
    The discriminator uses Convolution Layers (downsampling) to classify the generated images as real (positive output) 
    or fake (negative values).
    The LeakyRelLU activation was used after the each convolutional layer.
    Padding was set as 'same'.
    The Dropout techinique with rate 0.2 was used for a more stable training.
    
    Parameters
    ----------
    img_shape : {tuple of 4 elements} - 
            Shape of an input image. Default shape is (75, 64, 64, 1).
    kernel_size : {an integer or tuple/list of 3 integers} - 
            Specifies the depth, height and width of the 3D convolution window. 
            Can be a single integer to specify the same value for all spatial dimensions.. 
            Default is (4, 4, 4).
    strides : {an integer or tuple/list of 3 integers} -
            Specifies the strides of the convolution along the depth, height and width. 
            Can be a single integer to specify the same value for all spatial dimensions.
            Default is (1, 2, 2).
    downsample_layers : {int} - 
            Number of the Convolution Layers. 
            Default is 4.
    weight_initializer  - Kernel initializer.

    """
    rate = 0.2
    filters = input_shape[1]
    print('input_shape', input_shape)
    model = Sequential()

    model.add(Conv3D(strides=strides,
                  kernel_size = kernel_size,
                  filters = filters,
                  input_shape=input_shape,
                  padding='same', 
                  kernel_initializer=weight_initializer))

    model.add(LeakyReLU())
    model.add(Dropout(rate=rate))

    for l in range(downsample_layers-1):
        filters = int(filters*2)
        model.add(Conv3D(strides=strides,
                         kernel_size = kernel_size,
                         filters = filters,
                         padding='same'))

        model.add(LeakyReLU())
        model.add(Dropout(rate=rate))
        

    model.add(Flatten())
    model.add(Dense(1))
    return model


def discriminator2d(input_shape=(64, 64, 75),
                    kernel_size = (4, 4),
                    strides = (2, 2),
                    upsample_layers = 4):
    """
    Discriminator2D for a Deep Convolutional GAN.
    """
    
    filters = input_shape[0]

    model = Sequential()
    model.add(Conv2D(strides=strides,
                  kernel_size = kernel_size,
                  filters = filters,
                  input_shape=input_shape,
                  padding='same'))
  
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    for l in range(upsample_layers-1):
        filters = int(filters * 2)
        model.add(Conv2D(strides=strides,
                     kernel_size = kernel_size,
                     filters = filters,
                     padding='same'))
        model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1))
    return model



##############################
####        Losses        ####
##############################
# Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, we assign a random integer in range [0.7, 1.0] for positive class
# and [0.0, 0.3] for negative class

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3

# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)
    
    op_list = []
    # invert the labels in place
    #y_np[flip_ix] = 1 - y_np[flip_ix]
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1, y[i]))
        else:
            op_list.append(y[i])
    
    outputs = tf.stack(op_list)
    return outputs


# def generator_loss(fake_output):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss(fake_output, apply_label_smoothing=True):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    if apply_label_smoothing:
        fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
        return cross_entropy(tf.ones_like(fake_output_smooth), fake_output)
    
    else:
        return cross_entropy(tf.ones_like(fake_output), fake_output)


# def discriminator_loss(real_output, fake_output):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss


def discriminator_loss(real_output, fake_output, apply_label_smoothing=True, label_noise=True):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    if label_noise and apply_label_smoothing:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        real_output_smooth = smooth_positive_labels(real_output_noise)
        fake_output_smooth = smooth_negative_labels(fake_output_noise)
        
        real_loss = cross_entropy(tf.ones_like(real_output_smooth), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output_smooth), fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    loss = fake_loss + real_loss
    return loss