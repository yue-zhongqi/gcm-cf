import os

os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
--gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --nepoch 300 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 32 --dataroot /data2/xxx/Dataset/zsl/ps2/xlsa17/data --dataset APY \
--batch_size 64 --nz 64 --latent_size 64 --attSize 64 --resSize 2048 --encoder_use_y \
--lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --save_interval 50 \
--feed_lr 0.0001 --dec_lr 0.0005 --feedback_loop 2 --a1 0.01 --a2 0.01 --val_interval 5 --clf_epoch 2 \
--additional lre-5_a1e-2_a2e-2_recon0.1_dec5e-4 \
--eval --two_stage \
--use_mask best_masks/apy/baseline \
 --u_num 300 --u_lr 0.002 --u_beta 0.9 --u_epoch 3 --u_batch_size 1000''')