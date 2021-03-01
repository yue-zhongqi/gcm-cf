# GCM-CF ZSL

Scripts to train/evaluate the model are stored in image-scripts folder. To run the code, download Proposed Split V2 from https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/. In each image script, change --dataroot DATA_ROOT_DIR to the directory where the datasets are stored. In train_tfvaegan_inductive.py, change "/data2/xxx/Model/tfvaegan" to where you want to store the trained models.

To train the baseline model and evaluate it, run baseline.py. For example, for AWA2 dataset:

```
python image-scripts/awa2/baseline.py
```

To train the GCM-CF, run cf.py. Note that the trained cf model should be used as binary classifier, and the accuracy displayed during training is not the two-stage inference results reported in the paper. To get the two-stage results, run eval_two_stage_cf.py which uses the binary classification results pre-saved in out/ folder. Due to the upload size restriction, we are unable to provide the model files that were used to generate the binary classification results. They will be provided later in the actual release on GitHub.

To get the two-stage inference results using TF-VAEGAN as first-stage classifier, run eval_two_stage_baseline.py.