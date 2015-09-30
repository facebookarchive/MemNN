# MemN2N for bAbI tasks
This is a code used in paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)" for training MemN2N on bAbI question-answering tasks. 

## Setup
You need Matlab to run this code. We tested it on MATLAB versions R2014a and R2015a. In addition, you have to download bAbI data from [fb.ai/babi](http://fb.ai/babi). Then set `base_dir` variable in `run_babi.m` to the path where the data is stored.

## Usage
Run the following command in Matlab

    >> run_babi

This will start training on task 1. Change variable `t` to train on other tasks. To train on all tasks simultaniously, run

    >> run_babi_joint

You can try different model configurations that used in the paper by changing various options in `config_babi.m` (`config_babi_joint.m` for joint training). For example, setting 

    use_bow = true; linear_start = false;  randomize_time = 0;

you can train the simplest BOW model in the paper. 

You might notice that the performance varies lot in some tasks. In the paper, we repeat training 10 times and picked the one with lowest training error.

## References

* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.
