import os

os.system('''CUDA_VISIBLE_DEVICES=2 python lvae_train.py --baseline --dataset SVHN --encode_z 10 --contrastive_loss --temperature 2''')