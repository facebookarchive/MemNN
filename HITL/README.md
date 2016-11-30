# Dialogue Learning With Human-in-the-Loop

This project contains code for the dialog-based learning MemN2N setup in the following paper: "[Dialogue Learning with Human-in-the-Loop](https://openreview.net/pdf?id=HJgXCV9xx)". This implementation was written by [Jiwei Li](https://web.stanford.edu/~jiweil/).

## Setup

This code requires [Torch7](http://torch.ch) and its luarocks packages cutorch, cunn, nngraph, torchx, and tds.

To get the data, from this directory first run ./setup\_data.sh to download the data (94M download, unpacks to 449M).

## Dataset
After running ./setup\_data.sh:

./data/synthetic/ contains synthetic data for simulations.

./data/TurkData/ contains human-annotated versions of [WikiMovies](http://fb.ai/babi) data.


The synthetic data includes babi ("babi1_\*") tasks and WikiMovies ("movieQA_\*") data.

The TurkData is broken down as follows:

* qa-train.txt: 66307 questions, all human-annotated, based on WikiMovies
* qa-dev.txt: 9173 questions, as above
* qa-test.txt: 7848 questions, as above
* qa-train-1k.txt, qa-train-5k, qa-train-10k, qa-train-20k: random subsets of the training full 66k questions
* qa-train-1k-1s.txt: version of the 1k subset with a specific reward label of 1 on each example
* real_rbi-p=\*-memnn-feedback-train-10k.txt: different versions of 10k feedback with different p values (see paper for more details)


## Usage

You can use one of the \*.sh scripts as examples of how to train the model on one of the datasets.

As demonstrated there, to train run:

    th online_simulate.lua [params]

Available options are:

    -batch_size		(default 32, the batch size for model training)
    -token_size		(default 0, number of tokens)
    -init_weight	(default 0.1, initialization weights)
    -N_hop			(default 3, number of hops)
    -lr				(default 0.01, learning rate)
    -thres			(default 40, threshold for gradient clipping)
    -gpu_index		(default 1, which GPU to use)
    -dataset		(default 'babi', choose from 'babi' or 'movieQA')
    -setting		(default 'RBI', choose from 'RBI' or 'FP')
    -randomness     (default 0.2, random exploration rate for epsilon greedy)
    -simulator_batch_size   (default 32, the batch size of data generation. It is different from model batch size)
    -task			(default 3, which task to test)
    -nepochs		(default 20, number of iterations)
    -negative		(default 5, number of negative samples for FP)
    -REINFORCE      (default false, where to train the REINFORCE algorithm)
    -REINFORCE_reg  (default 0.1, entropy regularizer for the REINFORCE algorithm)
    -RF_lr          (default 0.0005, learning rate used by the REINFORCE baseline)
    -log_freq       (default 200, how often we log)
    -balance        (default false, enable label balancing experience replay strategy for FP)

## References

* Jiwei Li, Alexander H. Miller, Sumit Chopra, Marc'Aurelio Ranzato and Jason Weston, "[Dialogue Learning with Human-in-the-Loop](https://openreview.net/pdf?id=HJgXCV9xx).
