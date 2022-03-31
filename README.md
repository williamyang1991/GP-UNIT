# GP-UNIT - Official PyTorch Implementation

<img src="https://raw.githubusercontent.com/williamyang1991/GP-UNIT/main/doc_images/results.jpg" width="96%" height="96%">

This repository provides the official PyTorch implementation for the following paper:

**Unsupervised Image-to-Image Translation with Generative Prior**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In CVPR 2022.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/gpunit/) | [**Paper**]() (coming soon)
> **Abstract:** *Unsupervised image-to-image translation aims to learn the translation between two visual domains without paired data. Despite the recent progress in image translation models, it remains challenging to build mappings between complex domains with drastic visual discrepancies. In this work, we present a novel framework, Generative Prior-guided UNsupervised Image-to-image Translation (GP-UNIT), to improve the overall quality and applicability of the translation algorithm. Our key insight is to leverage the generative prior from pre-trained class-conditional GANs (e.g., BigGAN) to learn rich content correspondences across various domains. We propose a novel coarse-to-fine scheme: we first distill the generative prior to capture a robust coarse-level content representation that can link objects at an abstract semantic level, based on which fine-level content features are adaptively learned for more accurate multi-level content correspondences. Extensive experiments demonstrate the superiority of our versatile framework over state-of-the-art methods in robust, high-quality and diversified translations, even for challenging and distant domains.*

<img src="https://raw.githubusercontent.com/williamyang1991/williamyang1991.github.io/master/images/project/CVPR2022.jpg" width="96%" height="96%">

## Updates

- [03/2022] This website is created.

## Code

- We are cleaning our code. Coming soon. 

## (1) Dataset Preparation

## (2) Inference for Latent-Guided and Exemplar-Guided Translation

### Pretrained Models

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GMGCPt1xfh0Zs82lfkQLifMZR27yANTI?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1GlCnwd_0EDHNTsQrvM0xhA?pwd=cvpr) (access code: cvpr):

| Task | Pretrained Models | 
| :--- | :--- | 
| Prior Distillation | [content encoder](https://drive.google.com/file/d/1I7_IMMheihcIR57vInof5g6R0j8wEonx/view?usp=sharing) |
| Male←→Female | generators for [male2female](https://drive.google.com/file/d/1I0xqtbuuPOhcteHH613PGX7uO3fMTf9r/view?usp=sharing) and [female2male](https://drive.google.com/file/d/19xIY0vqHVpah4UyXKgz7trWi1TdlPD_T/view?usp=sharing) |
| Dog←→Cat←→Wild| generators for [dog2cat](https://drive.google.com/file/d/1PJyk0hKClceTHRD6BORFVEpG-pl0lPFm/view?usp=sharing), [cat2dog](https://drive.google.com/file/d/1EvQRKY4AN1JxAElsOPkuqqSRZVCNL3gA/view?usp=sharing), [dog2wild](https://drive.google.com/file/d/11j0kG1EJoSLWX6AJ8kaIWp_QFAZoUdpI/view?usp=sharing), [wild2dog](https://drive.google.com/file/d/16lr7bce4qGEmUWlfKZuRatyX_OsKdNkg/view?usp=sharing), [cat2wild](https://drive.google.com/file/d/16sBFFxbc1zX2LfruMUuY7aVZekng032g/view?usp=sharing) and [wild2cat](https://drive.google.com/file/d/1LVNfpBdHPNog1_tk9rE5x_4oQUUV8uEy/view?usp=sharing) |
| Face←→Cat or Dog| generators for [face2cat](https://drive.google.com/file/d/1CKKXDaD0h6i1RFtbcHOL8Bj3P37_Cheh/view?usp=sharing), [cat2face](https://drive.google.com/file/d/1cAYIYU6JUunBRhw94cssp8cPnuD7hJgo/view?usp=sharing), [dog2face](https://drive.google.com/file/d/1OPINn14b_rwKdEO1l_2ON__ngBLSllPq/view?usp=sharing) and [face2dog](https://drive.google.com/file/d/1_RrsNvaswMuLqrUr79q8HBx9TcP_zS9y/view?usp=sharing) |
| Bird←→Dog | generators for [bird2dog](https://drive.google.com/file/d/1Nm0jAI6dxDLBUdIFPEx8qU002IUdBaTO/view?usp=sharing) and [dog2bird](https://drive.google.com/file/d/1Ud_IJTO8Ovi7T7lsMG3S9boYASemnVAB/view?usp=sharing) |
| Bird←→Car | generators for [bird2car](https://drive.google.com/file/d/1PR_AF8JnYMaXH3hKipGxUZEjgrUu_0r5/view?usp=sharing) and [car2bird](https://drive.google.com/file/d/1j1N6vMNhPt4beLDrwP-uBd-jgPoQqg4O/view?usp=sharing) |
| Face→MetFace | generator for [face2metface](https://drive.google.com/file/d/15IoEyUUY-1vnfqqVquVYMCleHVXt4jJs/view?usp=sharing) |

The saved checkpoints are under the following folder structure:
```
checkpoint
|--content_encoder.pt     % Content encoder
|--bird2car.pt            % Bird-to-Car translation model
|--bird2dog.pt            % Bird-to-Dog translation model
...
```

### Latent-Guided Translation
Translate a content image to the target domain with randomly sampled latent styles:
```python
python inference.py --generator_path PRETRAINED_GENERATOR_PATH --content_encoder_path PRETRAINED_ENCODER_PATH \ 
                    --content CONTENT_IMAGE_PATH --batch STYLE_NUMBER
```
By default, the script will use `.\checkpoint\dog2cat.pt` as PRETRAINED_GENERATOR_PATH and `.\checkpoint\content_encoder.pt` as PRETRAINED_ENCODER_PATH.

Take Dog→Cat as an example, run:
> python inference.py --content ./data/afhq/images512x512/test/dog/flickr_dog_000572.jpg --batch 6

Six results `translation_flickr_dog_000572_N.jpg` (N=0~5) are saved in the folder `.\output\`.
An corresponding overview image `translation_flickr_dog_000572_overview.jpg` is additionally saved to illustrate the input content image and the six results: 

<img src="./output/translation_flickr_dog_000572_overview.jpg">

### Exemplar-Guided Translation
Translate a content image to the target domain in the style of a style image by additionally specifying `--style`:
```python
python inference.py --generator_path PRETRAINED_GENERATOR_PATH --content_encoder_path PRETRAINED_ENCODER_PATH \ 
                    --content CONTENT_IMAGE_PATH --style STYLE_IMAGE_PATH
```

Take Dog→Cat as an example, run:
> python inference.py --content ./data/afhq/images512x512/test/dog/flickr_dog_000572.jpg --style ./data/afhq/images512x512/test/cat/flickr_cat_000418.jpg 

The result `translation_flickr_dog_000572_to_flickr_cat_000418.jpg` is saved in the folder `.\output\`.
An corresponding overview image `translation_flickr_dog_000572_to_flickr_cat_000418_overview.jpg` is additionally saved to illustrate the input content image, the style image, and the result: 

<img src="./output/translation_flickr_dog_000572_to_flickr_cat_000418_overview.jpg" width="60%">

Another example of Cat→Wild, run:
> python inference.py --generator_path ./checkpoint/cat2wild.pt --content ./data/afhq/images512x512/test/cat/flickr_cat_000418.jpg --style ./data/afhq/images512x512/test/wild/flickr_wild_001112.jpg

The overview image is as follows: 

<img src="./output/translation_flickr_cat_000418_to_flickr_wild_001112_overview.jpg" width="60%">

## Results

#### Male-to-Female: close domains

![male2female](./doc_images/5.gif)

#### Cat-to-Dog: related domains 

![cat2dog](./doc_images/3.gif)

#### Dog-to-Human and Bird-to-Dog: distant domains  

![dog2human](./doc_images/4.gif)

![bird2dog](./doc_images/2.gif)

#### Bird-to-Car: extremely distant domains for stress testing

![bird2car](./doc_images/1.gif)

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang2022Unsupervised,
  title={Unsupervised Image-to-Image Translation with Generative Prior},
  author={Yang, Shuai and Jiang, Liming and Liu, Ziwei and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}
```

## Acknowledgments

The code is developed based on [StarGAN v2](https://github.com/clovaai/stargan-v2) and [Imaginaire](https://github.com/nvlabs/imaginaire).
