# Codes for final project of COMS 4731 Computer Vision
- Authors: Xudong Lin, Shiyuan Huang
- Email: xudong.lin@columbia.edu, shiyuanh15@gmail.com
- Our technical report is coming soon.

# Audio-conditioned talking face generation
- In this project, we built a system which generates talking face from an audio.
- Watch our results [here](https://www.youtube.com/watch?v=aUtfPJpzuuc)
- Our system consists of three modules: audio feature extractor, face generator, talking face generator.

## Audio feature extractor
### Prerequisites
- Matlab
- MatconvNet
### Instructions
- Download the dataset VoxCeleb: [Audio](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/), [frames extracted at 1fps](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/)
- Find the pretrained model for feature extractor: [emotion feature](https://github.com/albanie/mcnCrossModalEmotions/blob/master/README.md), [identoity feature](https://github.com/a-nagrani/VGGVox/blob/master/README.md).
- run extract_identity_fc_voxceleb in matlab


## Face generator
- Note that this part is borrowed from [this reimplementation of BEGAN](https://github.com/sunshineatnoon/Paper-Implementations). 
- Many thanks to the authors. We did some modification to improve the performance and to use it as an audio-face translator.
### Prerequisites
- PyTorch
- torchvision

### Training on CelebaA
- Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), choose the `Aligh&Croped Images` zip. Unzip it and put it under `data/` directory.
- Go into folder `Data` and run `python face_detect.py`, this script will detect and crop faces and store them under `Data/64_crop/` and `Data/128_crop` folder, this detecting and cropping script is from [BEGAN-tensorflow](https://github.com/Heumi/BEGAN-tensorflow/tree/master/Data) 
- Training

  **Train on 128x128 images**
  ```
  python began.py --cuda --outf 128/ --ndf 128 --ngf 128 --gamma 0.7 --loadSize 128 --fineSize 128 --dataPath Data/128_crop/ --res 0.5
  ```
###Reproduce the FID score
- Generate images
   **For example, use the model with residual loss at 40K**
   '''
   python generate.py --netG models/celeba_res.pth --outf imgs/celeba_res
   '''
- This will generate 12800 images in the outf folder. Do the same thing for model w\o residual loss.
- Go to [here](https://github.com/mseitzer/pytorch-fid) to find the codes for FID score computation.

### Training on Voxceleb emotion features
- you may need to change the folders in dataloader depending on where you put your extracted audio features
- Training with identity features
```
  python began_voxceleb_2.py --cuda --outf 128/ --ndf 128 --ngf 128 --gamma 0.7 --loadSize 128 --fineSize 128 --dataPath $where you put the images$ --res 0.5 --metric 0.5
  ```
  - Training with emotion features
```
  python began_voxceleb_e.py --cuda --outf 128/ --ndf 128 --ngf 128 --gamma 0.7 --loadSize 128 --fineSize 128 --dataPath $where you put the images$ --res 0.5 --metric 0.5 --nz 56
  ```
## Talking face generator
### Prerequisites
- Matlab
- MatconvNet
### Instructions
- Now you have audio and image generated from it, go to [You said that](http://www.robots.ox.ac.uk/~vgg/software/yousaidthat/) to find the demo for video synthesis.



## Acknowledgement
- Thanks for all the aforementioned previouis works! We will fix the liscense issue, if there is one.
