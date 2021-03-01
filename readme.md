# Counterfactual Zero-Shot and Open-Set Visual Recognition

This project provides implementations for our **CVPR 2021** paper Counterfactual Zero-Shot and Open-Set Visual Recognition, where we propose a counterfactual-based binary seen/unseen classifier (GCM-CF) for Zero-Shot Learning (ZSL) and Open-Set Recognition (OSR). This repo contains

- ZSL: Strong binary seen/unseen classifier that is plug-and-play with **any** ZSL method
- ZSL: Integrations with TF-VAEGAN, RelationNet, GDAN, CADA-VAE, LisGAN, AREN
- OSR: Complete code base on MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50
- OSR: Strong baseline (F1 score) of Softmax, OpenMax, CGDL
- OSR: Implementation of our GCM-CF

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