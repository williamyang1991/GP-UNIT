# Dataset Preparation

## 1. synImageNet291

**Overview**

Synthesized paired images in 291 classes.

<img src="https://user-images.githubusercontent.com/18130694/160621363-2f1bb0d3-b460-4877-ba9e-ed33b89a8194.jpg" width="48%"> <img src="https://user-images.githubusercontent.com/18130694/160621375-28bf2fe1-7488-4292-a9d7-fb14280d0682.jpg" width="48%">

**Description**

This dataset is synthesized for the training of our content encoder, containing paired data. The details of the dataset generation:

> We use the official BigGAN-deep-128 model on TF Hub 
to generate correlated images associated by random latent codes for each of the 291 domains 
including dogs, wild animals, birds and vehicles. 
Their class indexes in the original ImageNet 1000 classes are 7∼20, 22, 23, 80∼102, 104, 105, 106, 127, 128, 129, 131∼145, 151∼299, 330∼378, 
380, 382∼388, 407, 436, 468, 511, 555, 586, 609, 627, 654, 656, 675, 717,734, 751, 757, 779, 803, 817, 829, 847, 856, 864, 866, 867, 874. 
We apply truncation trick to the latent codes, and obtain 3K images with a truncation threshold of 0.5 and 3K images with a truncation threshold of 1.0. 
After filtering low quality ones, we finally obtain 655 images per domain that are linked across all domains, 600 of which are used for training. 

**Download**

synImageNet291 can be downloaded from 
[Google Drive](https://drive.google.com/file/d/1amMu_IU_W0ELGq7x2ixAMk5ZJlBUNCqL/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Nameanuztw8VXTd2QP9GSw?pwd=cvpr) (access code: cvpr)

**Content**

Unzip the downloaded **synImageNet291.zip**, which is in the following folder structure:
```
synImageNet291
|--Train
   |--7_bird
      <600 training images in the class 7 of ImageNet>
   |--8_bird
      <600 training images in the class 8 of ImageNet>
   ...
   |--874_vehicle
      <600 training images in the class 8874 of ImageNet>   
|--Test
   |--7_bird
      <55 testing images in the class 7 of ImageNet>   
   ...
   
synImageNet291_mask
|--Train
   |--7_bird
      <600 instance segmentation maps of the correponding images in synImageNet291>
   ...
|--Test
   |--7_bird
      <55 instance segmentation maps of the correponding images in synImageNet291>   
   ...
```

Images with the same filename are generated from the same latent code.
For example, `synImageNet291/Train/7_bird/0.jpg`, `synImageNet291/Train/158_dog/0.jpg` and `synImageNet291/Train/271_otheranimal/0.jpg` are paired images, 
containing three kinds of animals sharing very similar pose.


## 2. ImageNet291

**Description**

This dataset is synthesized for the training of our content encoder.
It also serves as the training and testing data for the tasks of Dog2Bird and Bird2Car in our paper.
The details of the dataset generation:

> For each domain X, we first calculate the mean style feature S_X of the images in X from **synImageNet291**. 
The style feature is defined as the channel-wise mean of the conv5_2 feature of pre-trained VGG. Then, we apply
HTC to detect and crop the object regions in the domain X of ImageNet. Small objects are filtered. The remaining
images are ranked based on the similarity between their style features and S_X . We finally select the top 800 images to
eliminate outliers, with 600 images for training and 50 images for testing.

**Download**

For [instance segmentation maps](https://drive.google.com/file/d/1iAGsFmkFAYxfwo4tXkctCKikCUjmBn85/view?usp=sharing) of the images in ImageNet291, they can be downloaded from 
[Google Drive](https://drive.google.com/drive/folders/1LPbUqpOM4oADnhRqWteTRiVrYo_U01zx?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Nameanuztw8VXTd2QP9GSw?pwd=cvpr) (access code: cvpr).
The link also provides the instance segmentation maps of the face images in CelebHA in the `ImageNet291_mask\train\1001_face\` and `ImageNet291_mask\test\1001_face\`

For the images of ImageNet291, we provide the script to build ImageNet291 from the original ImageNet training set.

* Download the original [ImageNet](https://image-net.org/download.php) dataset 
* Download the [ImageNet_To_ImageNet291.txt](https://drive.google.com/file/d/10nvdJigeKn5mA5rsPdWLu9g8uKJe0d_p/view?usp=sharing), a text file containing the filenames of the filtered training set. 
* Download the [bbox_select_txt.zip](https://drive.google.com/file/d/1PvOaLwm8-mjpCJPesPRyIA4U4SgQkvr5/view?usp=sharing), unzip it to obtain the bounding boxes of the images. 
* Specify the folder path to the training set of ImageNet in [Line 8 of generate_ImageNet291.py](./generate_ImageNet291.py#L8).
   * The folder is expected to contain 1000 subfolders of `n01440764`, `n01443537`, ...
* Specify the folder path to [bbox_select_txt](https://drive.google.com/file/d/1PvOaLwm8-mjpCJPesPRyIA4U4SgQkvr5/view?usp=sharing) in [Line 9 of generate_ImageNet291.py](./generate_ImageNet291.py#L9).
* Specify the folder path to save the images of ImageNet291 in [Line 10 of generate_ImageNet291.py](./generate_ImageNet291.py#L10).
* Specify the file path to [ImageNet_To_ImageNet291.txt](https://drive.google.com/file/d/10nvdJigeKn5mA5rsPdWLu9g8uKJe0d_p/view?usp=sharing) in [Line 11 of generate_ImageNet291.py](./generate_ImageNet291.py#L11). Run:

```python
python generate_ImageNet291.py
```

**Content**

The folder structure of ImageNet291 is the same as that of synImageNet291.
