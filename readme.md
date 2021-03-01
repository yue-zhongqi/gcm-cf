# Counterfactual Zero-Shot and Open-Set Visual Recognition

This project provides implementations for our **CVPR 2021** paper Counterfactual Zero-Shot and Open-Set Visual Recognition, where we propose a counterfactual-based binary seen/unseen classifier (GCM-CF) for Zero-Shot Learning (ZSL) and Open-Set Recognition (OSR). This repo contains

- ZSL: Strong binary seen/unseen classifier that is plug-and-play with **any** ZSL method
- ZSL: Integrations with TF-VAEGAN, RelationNet, GDAN, CADA-VAE, LisGAN, AREN
- OSR: Complete the OSR code base on MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50 with 5 fixed random seed
- OSR: Strong baseline (F1 score) of Softmax, OpenMax, CGDL
- OSR: Implementation of our GCM-CF

<br />
For technical details, please refer to:

**Counterfactual Zero-Shot and Open-Set Visual Recognition** <br />
[Zhongqi Yue*](https://www.linkedin.com/in/yue-zhongqi-37119386/?originalSubdomain=sg), [Tan Wang*](https://wangt-cn.github.io/), [Hanwang Zhang](https://www.ntu.edu.sg/home/hanwangzhang/), [Qianru Sun](https://qianrusun1015.github.io), [Xian-Sheng Hua](https://scholar.google.com/citations?user=6G-l4o0AAAAJ&hl=en) <br />
\* Equal contribution <br />
**IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021** <br />

<div align="center">
  <img src="https://github.com/Wangt-CN/gcm-cf/blob/main/osr/images/GCM_CF.png" width="600px" />
</div>

## Usage

Please refer to the README.md in ZSL and OSR folder, respectively.

## TODO

## Citation

If you find our work or the code useful, please consider cite our paper using:

```bibtex
@inproceedings{yue2021counterfactual,
  title={Counterfactual Zero-Shot and Open-Set Visual Recognition},
  author={Yue, Zhongqi and Wang, Tan and Zhang, Hanwang and Sun, Qianru and Hua, Xian-Sheng},
  booktitle= {CVPR},
  year={2021}
}
```
