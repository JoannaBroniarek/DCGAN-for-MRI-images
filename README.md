# DCGAN-for-MRI-images

Inspiration for this project was the paper: http://www.nlab.ci.i.u-tokyo.ac.jp/pdf/isbi2018.pdf

The idea was to implement and train DCGAN model on the 3D MRI images from  BRATS 2019 dataset.  

# Demo

To see a short explanation of the implementation as well as to generate new images, please see the [demo notebook](https://github.com/JoannaBroniarek/DCGAN-for-MRI-images/blob/master/Demo.ipynb) 

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



------------------------

### To Do:

* requirements txt
* pre-trained model upload

