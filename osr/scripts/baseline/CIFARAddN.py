import os

os.system('''CUDA_VISIBLE_DEVICES=2 python lvae_train.py --baseline --dataset CIFARAddN --epochs 100 --save_epoch 65 --unseen_num 13''')
os.system('''CUDA_VISIBLE_DEVICES=2 python lvae_train.py --baseline --dataset CIFARAddN --epochs 100 --save_epoch 65 --unseen_num 18''')
