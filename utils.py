import os
from matplotlib import pyplot as plt

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def gen_and_save_images(model, epoch, test_noise, save_dir_path, gen_loss, disc_loss, show=False):
    """
    Generates synthetic images while training.
    
    Parameters:
    ----------
    model - the pretrained generator model
    epoch - the number of a current epoch
    test_noise - the noise vector
    save_dir_path - The output path for images
    gen_loss, disc_loss - The geneator and discriminator losses
    show : {boolean} - If True, it plots the images immediately 
    
    """
    save_dir_path = os.path.join(save_dir_path, "images")
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
        
    preds = model(test_noise, training=False)
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Epoch: {}\nG_Loss: {:.4f} D_Loss: {:.4f}".format(epoch, gen_loss, disc_loss, fontsize=14))

    for ind, i in enumerate(list(range(7, 70, 4))):
        plt.subplot(4, 4, ind+1)
        #plt.imshow(preds[0][:,:,i], cmap='gray')
        plt.title(str(i))
        plt.imshow(preds[0][i,:,:,0], cmap='gray')
        plt.axis('off')
  
    plt.savefig(
        os.path.join(save_dir_path, 'img_epoch_{:04d}.png'.format(epoch)), bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()

    
def generate_image(model, test_noise, save=False, save_dir_path=None, show=False):
    """
    Generates new synthetic images for a given noise vector.
    This function is used out of training.
    
    Parameters:
    ----------
    model - the pretrained generator model
    test_noise - the noise vector
    save - : {boolean} - If True, it saves the images to the specified by 'save_dir_path' folder. 
    save_dir_path - The output path for images
    show : {boolean} - If True, it plots the images immediately 
    
    """
        
    preds = model(test_noise, training=False)
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("An example sequence of synthetic images", fontsize=14)

    for ind, i in enumerate(list(range(7, 70, 4))):
        plt.subplot(4, 4, ind+1)
        plt.title(str(i))
        plt.imshow(preds[0][i,:,:,0], cmap='gray')
        plt.axis('off')
    
    if save:
        save_dir_path = os.path.join(save_dir_path, "images")
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
            
        plt.savefig(
            os.path.join(save_dir_path, 'example.png'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    

def plot_image_seq(img, save=False):
    """ 
    Plot an example of a sequence of the original images 
    """
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("An example sequence of the original images", fontsize=14)

    for ind, i in enumerate(list(range(7, 70, 4))):
        plt.subplot(4, 4, ind+1)
        plt.title(str(i))
        plt.imshow(img[:,:, i], cmap='gray')
        plt.axis('off')
    
    if save:
        plt.savefig('docs/example.png', bbox_inches='tight')
        
        
def get_pretrained_model(generator):
    ''' 
    Function used in demo to load the pretrained model for a generator.
    '''
    save_dir_path = 'logs/2020-05-04_06:40:29/'
    checkpoint_dir = save_dir_path
    checkpoint_prefix = os.path.join("training_checkpoints", "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator)

    manager = tf.train.CheckpointManager(checkpoint,
                                      directory=checkpoint_dir,
                                      max_to_keep = 10,
                                      checkpoint_name=checkpoint_prefix)

    status = checkpoint.restore(manager.latest_checkpoint)