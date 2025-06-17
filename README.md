# Denoising Traffic Sign Images
# A comparative study between CNN and Diffusion Models.
Youri Joosten

This repository contains the source code for my bachelor thesis.
It contains the pytorch implementation of:
- The EDM model by Karras et al. (2022). https://arxiv.org/abs/2206.00364
- The EDSR model by Lim et al. (2017). https://arxiv.org/abs/1707.02921

It also contains the script for the comparative pipeline (comparative_pipeline), this script contains functions for loading and preprocessing data, generating metrics and plotting figures.
The denoising implementation of EDM can be found in the Inference.py file, which is based on the generate.py file by Karras et al.(2022). 

The dataset and the trained model can be found and downloaded on this [google drive](https://drive.google.com/drive/folders/1EBDqZnk6EeHSiUhh-l2gkaVr_J9ZUXn3?usp=sharing).

# Abstract:
In this paper the effectiveness of Diffusion-based and Convolutional Neural Network
(CNN) based image denoising are compared in their ability to improve classification
performance and image restoration. The Elucidated Diffusion model (EDM) and the
Enhanced Deep Residual Networks model (EDSR) are compared in their ability to de-
noise traffic sign images. The restored images are evaluated on metrics such as accuracy
using the YOLOv11 classification model, inference speed and similarity metrics.

The results show that EDSR outperforms the EDM model on all metrics except on infer-
ence speed when EDM uses a low number of diffusion steps.

EDSR scores a classifaction accuracy of 96.19%, an average PSNR score of 19.9628, an
average SSIM value of 0.6994 and takes 16.57ms to denoise an image.
EDM, using 30 diffusion steps, scores a classification accuracy of only 4.78%, an average
PSNR score 6.9220, an average SSIM value of 0.2481 and takes 16.10ms to denoise an
image.

