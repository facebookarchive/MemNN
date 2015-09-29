# MemN2N for language modeling
This code trains MemN2N model for language modeling as explained in paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)".

## Setup
This code requires [Torch7](http://torch.ch/) and its [nngraph](http://github.com/torch/nngraph), [tds](http://github.com/torch/tds) packages. Also, it uses CUDA to run on GPU for faster training. To train on Penn Treebank corpus, you should download it separately (should be formatted like [this](http://github.com/wojzaremba/lstm/tree/master/data)) and put in `data` subdirectory.

## Usage
To train a model with 6 hops and memory size of 100, run the following command

    th main.lua --show --nhop 6 --memsize 100

To see all training options, run

    th main.lua --help

which will print

    Usage: main.lua [options] 
      --gpu         GPU id to use [1]
      --edim        internal state dimension [150]
      --lindim      linear part of the state [75]
      --init_std    weight initialization std [0.05]
      --init_hid    initial internal state value [0.1]
      --sdt         initial learning rate [0.01]
      --maxgradnorm maximum gradient norm [50]
      --memsize     memory size [100]
      --nhop        number of hops [6]
      --batchsize   [128]
      --show        print progress [false]
      --load        model file to load []
      --save        path to save model []
      --epochs      [100]
      --test        enable testing [false]

Note that model performance varies from run to run. In the paper, we run each experiment 10 times and picked the one with best validation perplexity.
  
## References

* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.
