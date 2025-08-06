# HiFusion
The official code of [*"HiFusion: An Unsupervised Infrared and Visible Image Fusion Framework With a Hierarchical Loss Function"*](https://ieeexplore.ieee.org/document/10912736)
> 📝 Published in: IEEE Transactions on Instrumentation and Measurement (TIM), 2025  
> 🧑‍💻 Author: [Kaicheng Xu](https://github.com/Kylesesa), An Wei, Congxuan Zhang, Zhen Chen, Ke Lu, Weiming Hu, and Feng Lu.


## Overview

<p align="center">
  <img src="framework.png" alt="overview" width="90%">
</p>
<p align="center">
    The overall framework of the proposed HiFusion
</p>

## Environment

```latex
Python = 3.7.16
PyTorch = 1.11.0
torchvision = 0.12.0
numpy = 1.19.5，
matplotlib = 3.5.3
opencv-python = 4.8.0.76
```
Please make sure all dependencies are installed. If certain packages are missing, please install them manually. If you are unable to set up the environment successfully, please contact me.
## Datasets
```
HiFusion (RCMCGAN_pytorch was its original name)
├── 数据集
|   ├── IR
|   ├── VIS
|   |
|   ├── test_datasets 
|   |   └── TNO
|   |        ├──ir
|   |        └──vis
|   |   ├── MSRS
|   |        ├──ir
|   |        └──vis
|   |   └── ...
|   | 
|   └── Other_datasets
|   └── ...
```

## Train 
1. Train an autoencoder using the AE.py. Training weights are automatically saved.
2. Set up the weight paths for the final AE in the main.py and start training the fusion network.

## Test
1. Chose a function to generate fused images. (generate_VIFB can peocess rgb image)
2. Set up the weight path for the fusion model (log_path), test set path (test_path) and the save path for the fused image (save_path) in the generate. py file.

## Citation

```latex
@ARTICLE{xu2025hifusion,
  author={Xu, Kaicheng and Wei, An and Zhang, Congxuan and Chen, Zhen and Lu, Ke and Hu, Weiming and Lu, Feng},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={HiFusion: An Unsupervised Infrared and Visible Image Fusion Framework With a Hierarchical Loss Function}, 
  year={2025},
  volume={74},
  number={},
  pages={1-16},
  keywords={Training;Feature extraction;Loss measurement;Image fusion;Electronic mail;Generative adversarial networks;Data mining;Visualization;Visual perception;Usability;End-to-end;hierarchical loss function;image fusion;infrared and visible images;unsupervised learning},
  doi={10.1109/TIM.2025.3548202}}
```
K. Xu et al., "HiFusion: An Unsupervised Infrared and Visible Image Fusion Framework With a Hierarchical Loss Function," in IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1-16, 2025, Art no. 5015616, doi: 10.1109/TIM.2025.3548202.


