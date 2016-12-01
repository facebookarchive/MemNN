# Dialog-based Language Learning

This project contains code for the dialog-based learning MemN2N setup in the following paper: "[Dialog-based Language Learning](https://arxiv.org/abs/1604.06045)". This implementation was written by [Jiwei Li](https://web.stanford.edu/~jiweil/).

## Setup

This code requires [Torch7](http://torch.ch) and its luarocks packages cutorch, cunn, nngraph, torchx, and tds.

To get the data, from this directory first run ./setup\_data.sh to download the data (368M download, unpacks to 4.7GB).

## Usage

You can use one of the try\_\*.sh scripts as examples of how to train the model on one of the datasets.

As demonstrated there, to train run:

    th train.lua [params]

Available options are:

    -batch_size		(default 32)
    -token_size		(default 0, number of tokens)
    -dimension		(default 20, dimensionality of embedding vectors)
    -init_weight	(default 0.1, initialization weights)
    -N_hop			(default 3, number of hops)
    -lr				(default 0.01, learning rate)
    -thres			(default 40, threshold for gradient clipping)
    -iter_halve_lr	(default 20, number of iterations after which start halving learning rate)
    -task			(default 1, which task to test)
    -gpu_index		(default 1, which GPU to use)
    -policy			(default 0.5, choose from 0.01, 0.1, or 0.5 to select the rate of correct--vs random--answers)
    -N_iter			(default 20, number of iterations)
    -beta			(default true, whether to use beta for FP setting)
    -negative		(default 5, number of negative samples)
    -dataset		(default 'babi', choose from 'babi' or 'movieQA')
    -setting		(default 'RBI', choose from 'RBI', 'FP', 'IM', or 'RBI+FP')

## References

* Jason Weston, "[Dialog-based Language Learning](https://arxiv.org/abs/1604.06045)", *arXiv:1604.06045 [cs.CL]*.
