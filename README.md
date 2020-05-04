# DCGAN-for-MRI-images

Inspiration for this project was the paper: http://www.nlab.ci.i.u-tokyo.ac.jp/pdf/isbi2018.pdf

The idea was to implement and train DCGAN model on the 3D MRI images from  BRATS 2019 dataset.  

## Repository Content

* **train3D.py** -  the file to be run (with training functions and 'main' function)

* **models.py** -  contains implementation of discriminator and generator
* **create_data.py** - functions used for data preprocessing and generation of the TfRecords file 
* **utils.py** -  additional useful functions used to i.e. plotting images
* docs/ - a folder with gif / figures 

## Custom Training

```
$ python3 train3D.py
	--epochs 100
    --batch_size 16
    --lr_g 5e-4
    --lr_d 5e-5
    --rand_seed 42
```



## Sample run for 100 epochs

<img src="docs/mri.gif" style="zoom:80%;" />

-----------------

### Problems encountered while training and used GAN tricks

* Discriminator was decreasing to zero values after few epochs

* Exploding Losses

* Memory issues (tfRecords, BufferSize)

  

In order to improve the performance and make the training more stable, the following tricks were used:

* Batch Normalization
* Weight Initializer from Truncated Normal distribution
* tuning of the initial learning rates
* Decaying learning rates (ExponentialDecay) 
* trying different Activation functions



------------------------

### Requirements

TODO : add info about library versions