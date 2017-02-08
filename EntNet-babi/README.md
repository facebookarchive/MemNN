# Recurrent Entity Networks

This project contains the source code for training Recurrent Entity Networks on the bAbI tasks, described in the paper "[Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969)". This implementation was written by [Mikael Henaff](http://www.mikaelhenaff.com/).

## Usage

To download the data, run

    chmod +x get_data.sh
    ./get_data.sh

To train a model on task 2 with the parameters described in the paper, run the following command

    th main.lua -task 2

To train with different hyperparameters, do something like

    th main.lua -task 5 -edim 20 -memslots 50

One can also tie the keys to the entity embeddings as follows

    th main.lua -task 5 -edim 20 -tied 1

This will create models with different numbers of memory slots depending on the number of entities in each task.

By default, the training will be repeated 10 times with different initializations. The number of runs is a hyperparameter that can be changed.
After each epoch, a .log file with the performance as well as a .model file containing the current weights will be saved to the outputs folder.

The numbers may change slightly from the ones in the paper, depending on the version of Torch being used. Here are results with the open-source version of Torch, showing the error rate on each task:

Task | EntNet (paper) | EntNet (repo)
--- | --- | ---
1: 1 supporting fact | 0 | 0
2: 2 supporting facts | 0.1 | 0.4
3: 3 supporting facts | 4.1 | 4.5
4: 2 argument relations | 0 | 0
5: 3 argument relations | 0.3 | 0.3
6: yes/no questions | 0.2 | 0
7: counting | 0 | 0
8: lists/sets | 0.5 | 0.3
9: simple negation | 0.1 | 0
10: indefinite knowledge | 0.6 | 0.1
11: basic coreference | 0.3 | 0.1
12: conjunction | 0 | 0
13: compound coreference | 1.3 | 2.1
14: time reasoning | 0 | 0
15: basic deduction | 0 | 0
16: basic induction | 0.2 | 0.2
17: positional reasoning | 0.5 | 0.6
18: size reasoning | 0.3 | 1.2
19: path finding | 2.3 | 0.4
20: agents motivation | 0 | 0
**Failed Tasks** | 0 | 0
**Mean Error** | 0.5 | 0.5

## References

* Mikael Henaff, Jason Weston, Arthur Szlam, Antoine Bordes, and Yann LeCun, "[Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969)", *arXiv:1612.03969 [cs.CL]*.
