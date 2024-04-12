# B2Opt-Learning-to-Optimize-Black-box-Optimization-with-Little-Budget

This repository is the official implementation of the source code of the paper  [B2Opt: Learning to Optimize Black-box Optimization with Little Budget](https://arxiv.org/abs/2304.11787).

# Installation

This project requires running under the Ubuntu 20.04 system, and you need to install the cuda version of pytorch >= 1.12 first. And the python package [BBOB](https://github.com/ninja-wm/BBOB/tree/main) should be installed. Our python version is 3.8.2. First, please install the dependency packages in requirements. Then, please install B2Opt as follows:

```bash
git clone git@github.com:ninja-wm/B2Opt-Learning-to-Optimize-Black-box-Optimization-with-Little-Budget.git
cd B2Opt_pkg
./install.sh
```

# Quick Start

Compared with the previous version, we have further packaged B2Opt in this version. This makes training and testing B2Opt easier. 

This tutorial can help you quickly reproduce the results of the synthetic function and BBOB in the paper.

```bash
cd exps
```

* step1)  Run the following command to view the command line interface parameter description:

```bash
python ./main.py --help
```

* step 2) You can train on TF1-TF3 by executing the following command:

```bash
python ./main.py -d 10 -expname test1 -ems 30 -ws True -popsize 100 -maxepoch 500 -lr 0.001 -mode train
```

​	Note: The parameters d, expname, ems, ws and popsize determine the architecture of B2Opt and should therefore be consistent during testing and training. If you want to customize B2Opt and train it on other tasks, please read and modify main.py.

* step 3) You can directly use the trained B2Opt to solve TF4-TF9 through the following command:

```bash
python ./main.py -d 10 -expname test1 -ems 30 -ws True -popsize 100 -lr 0.001 -mode test -target sys
```

​	Change the target parameter to "bbob" to solve the 24 functions of BBOB.

# statement

The code we provide can be run directly, but it is difficult for us to guarantee that it will execute smoothly on different platforms and different operating systems. If you encounter any problems, please submit an issue to help us improve it! Grateful!


# Citiation

If our projects have been helpful to you, please cite us below! Thanks!

```
@misc{li2023b2opt,
      title={B2Opt: Learning to Optimize Black-box Optimization with Little Budget}, 
      author={Xiaobin Li and Kai Wu and Xiaoyu Zhang and Handing Wang and Jing Liu},
      year={2023},
      eprint={2304.11787},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
