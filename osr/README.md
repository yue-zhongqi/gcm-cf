# GCM-CF OSR
### Overview

This is the official code of the **OSR** task for the paper "Counterfactual Zero-shot and Open-Set Visual Recognition". The codebase is built based on the [CGDL](https://github.com/BraveGump/CGDL-for-Open-Set-Recognition) (The code is now deleted by the original author).



### Installation

- Python 3.6
- Pytorch 1.4.0
- Torchvision
- matplotlib
- tensorboardx
- opencv-python
- scikit-learn
- R (For using the qmv.py in CGDL)

```
### For installing R
conda install r-base=3.6
pip install rpy2

Then change the `os.environ['R_HOME'] = 'xxxx/lib/R'` in qmv.py to your environment path.

R
chooseCRANmirror(graphics=F)
install.packages("mvtnorm")
```



### Running

Scripts for different datasets are stored in `scripts` folder. The dataset can be directly downloaded by running the code. 

For example, to run the baseline method for the MNIST dataset, run:

```
python scrips/baseline/MNIST.py
```

For the counterfactual training:

```
python scripts/MNIST/MNIST_cf.py
```



Note that for counterfactual testing, you need add flag `--eval --cf --use_model` to activate the counterfactual evaluation.

Some other evaluation options `--yh --use_model_gau --threshold` can be adjusted for the better performance. Please refer to the argument description in `lvae_train.py` and our paper. 



### Results

<div align="center">
  <img src="https://github.com/Wangt-CN/gcm-cf/blob/master/osr/images/Visualization.png" width="1000px" />
</div>

<div align="center">
  <img src="https://github.com/Wangt-CN/gcm-cf/blob/master/osr/images/Results.png" width="600px" />
</div>