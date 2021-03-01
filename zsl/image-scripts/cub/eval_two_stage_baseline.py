import os

os.system('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8  python train_tfvaegan_inductive.py --gammaD 10 --gammaG 10 --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot /data2/xxx/Dataset/zsl/ps2/xlsa17/data --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --val_interval 5 --clf_epoch 5 \
--eval --two_stage \
--use_mask best_masks/cub/baseline \
--u_num 200 --u_lr 0.0005 --u_beta 0.5 --u_epoch 9 --u_batch_size 500')