import os

os.system('''CUDA_VISIBLE_DEVICES=0 python lvae_train.py --baseline --dataset CIFAR10 --epochs 100''')