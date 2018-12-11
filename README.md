# Codes for final project of COMS 4731 Computer Vision
Authors: Xudong Lin, Shiyuan Huang
Email: xudong.lin@columbia.edu, shiyuanh15@gmail.com
Our technical report is coming soon.

# Audio-conditioned talking face generation
In this project, we built a system which generates talking face from an audio.
Watch our results [here](https://www.youtube.com/watch?v=aUtfPJpzuuc)
Our system consists of three modules: audio feature extractor, face generator, talking face generator.

## Audio feature extractor
### Prerequisites

- PyTorch
- torchvision




## Training

- Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), choose the `Aligh&Croped Images` zip. Unzip it and put it under `data/` directory.
- Go into folder `Data` and run `python face_detect.py`, this script will detect and crop faces and store them under `Data/64_crop/` and `Data/128_crop` folder, this detecting and cropping script is from [BEGAN-tensorflow](https://github.com/Heumi/BEGAN-tensorflow/tree/master/Data) 
- Training

  **Train on 64x64 images**
  ```
  python began.py --cuda --dataPath Data/64_crop --gamma 0.4 --niter 200000
  ```

  **Train on 128x128 images**
  ```
  python began.py --cuda --outf 128/ --ndf 128 --ngf 128 --gamma 0.7 --loadSize 128 --fineSize 128 --dataPath Data/128_crop/
  ```




## Reference
1. https://github.com/sunshineatnoon/Paper-Implementations/blob/master/BEGAN/Readme.md
2. Berthelot, David, Tom Schumm, and Luke Metz. "BEGAN: Boundary Equilibrium Generative Adversarial Networks." arXiv preprint arXiv:1703.10717 (2017).
