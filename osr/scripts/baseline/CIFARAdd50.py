import os

os.system('''CUDA_VISIBLE_DEVICES=1 python lvae_train.py --baseline --dataset CIFARAdd50 --save_epoch 60 --epochs 100''')