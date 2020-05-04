import os
from matplotlib import pyplot as plt

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
    