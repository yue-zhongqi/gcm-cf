# GCM-CF OSR
This is the draft code of osr task for submission 4951. Scripts for different datasets are stored in `scripts` folder. The dataset can be directly downloaded by running the code. Remember to change the `R Home` in `qmv.py` file, which is just following the code of CGDL. Then the code can be run.

For example, for MNIST dataset, run:

```
python scrips/baseline/MNIST.py
```

Note that for cf training, you need add flag `--cf --use_model` to activate cf evaluation. The detailed model file and readme would be released on github upon acceptance.