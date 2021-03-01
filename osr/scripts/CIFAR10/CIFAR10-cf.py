import os

os.system('''CUDA_VISIBLE_DEVICES=0 python lvae_train.py --encode_z 10 --baseline --dataset CIFAR10 --epochs 100 --contrastive_loss --beta_z 6 --save_epoch 60''')
