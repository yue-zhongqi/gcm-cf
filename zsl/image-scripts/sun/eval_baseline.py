
import os

os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 1 --gammaG 1 --gzsl \
--manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 \
--ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 \
--classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot /data2/xxx/Dataset/zsl/ps2/xlsa17/data \
--recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --encoder_use_y \
--save_interval 50 --clf_epoch 6 --val_interval 5 --eval --baseline''')
