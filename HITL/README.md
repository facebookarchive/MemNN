# Dialogue Learning With Human-in-the-Loop

This project contains code for the dialog-based learning MemN2N setup in the following paper: "[Dialogue Learning with Human-in-the-Loop](https://arxiv.org/abs/1611.09823)".

## Setup

This code requires [Torch7](http://torch.ch) and its luarocks packages cutorch, cunn, nngraph, torchx, and tds.

To get the synthetic data, from this directory first run ./setup\_data.sh to download the data (90M download, unpacks to 435M).

## Dataset
After running ./setup\_data.sh:

./data/ contains synthetic data for simulations.

The synthetic data includes babi ("babi1_\*") tasks and WikiMovies ("movieQA_\*") data.


We additionally have [another dataset available](https://s3.amazonaws.com/fair-data/memnn/human_in_the_loop/turk_data.tar.gz), which contains human-annotated versions of [WikiMovies](http://fb.ai/babi) data. This data is in a slightly simpler format, so the code here does not yet run on it out-of-the-box. It is a 4M download which unpacks to 15M.

The turk_data is broken down as follows:

* qa-train.txt: 66307 questions, all human-annotated, based on WikiMovies
* qa-dev.txt: 9173 questions, as above
* qa-test.txt: 7848 questions, as above
* qa-train-1k.txt, qa-train-5k, qa-train-10k, qa-train-20k: random subsets of the training full 66k questions
* real_rbi-r=\*-memnn-feedback-train-10k.txt: different versions of 10k feedback data with different r values, which represent the proportion of data with rewards. r=2 corresponds to regular supervised learning, with true labels given but no feedback.


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

* Jiwei Li, Alexander H. Miller, Sumit Chopra, Marc'Aurelio Ranzato and Jason Weston, "[Dialogue Learning with Human-in-the-Loop](https://arxiv.org/abs/1611.09823)", *arXiv:1611.09823 [cs.AI]*.
