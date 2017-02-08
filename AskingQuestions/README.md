# Learning through Dialog Interactions

This project contains code for the dialog-based learning MemN2N setup in the following paper: "[Learning through Dialogue Interactions](https://arxiv.org/abs/1612.04936)".

## Setup

This code requires [Torch7](http://torch.ch) and its luarocks packages cutorch, cunn, nngraph, torchx, and tds.

To get the data, from this directory first run ./setup\_data.sh to download the data (1.6G download, unpacks to 8.4GB. md5: a590f31654fc5525e4fc2ee4557e80dc).

## Dataset
data/movieQA_kb contains movieQA knowledge base (based on [WikiMovies](http://fb.ai/babi)) to run the simulators
data/AQ_supervised_data contains data to run the supervised settings described in the paper
data/AQ_supervised_real contains human-written versions of supervised tasks 4 and 8
data/AQ_reinforce_data contains data to run the reinforcement learning settings described in the paper

## Simulator

The Simulator described in the paper that simulates dialogues between a teacher and a student.
This transforms the movieQA data to the version used to train the supervised and reinforcement learning models.

As demonstrated there, to run the simulator:

    th runSimulator.lua [params]


Available options are:

    -mode           (default train, taking values of train|dev|test)
    -task           (default 1, the task you want to simulate, taking values of 1-9)
    -prob_correct_final_answer  (default 0.5, the policy that controls the probability that a student gives a correct final answer)
    -prob_correct_intermediate_answer   (default 0.5, the policy that controls the probability that a student asks a correct question)
    -homefolder     (default ../data/movieQA_kb, the folder to load the movieQA database)
    -junk           (default true, whether to incorporate random question-answer pairs in the conversation)
    -randomQuestionNumTotal (default 5, the number of random question-answer pairs in the conversation)
    -setting        (default AQ, taking values of AQ|QA|mix, AQ denotes the setting in which the student always asks a question, QA denotes the setting in which the student never asks a question, mix means the student asks questions for 50 percent of the time)
    -output_dir     (default "./", the output folder path)


## Supervised

The offline supervised learning settings described in the paper

to run the trainer:

    th train_supervised.lua [params]


Available options are:

    -batch_size		(default 32, the batch size for model training)
    -init_weight	(default 0.1, initialization weights)
    -N_hop			(default 3, number of hops)
    -lr				(default 0.05, learning rate)
    -thres			(default 40, threshold for gradient clipping)
    -gpu_index		(default 1, which GPU to use)
    -task           (default 1, which task to run)
    -trainSetting   (default AQ, taking values of QA|AQ|mix, which training setting to run. AQ means the student is trained on the dataset which allows asking questions, QA means not allowing asking questions, mix means allowing asking questions for 50 percent of time)
    -testSetting    (default AQ, taking values of QA|AQ|mix, which setting to test on. AQ means the student is allowed to ask questions at test time. QA means the student is not allowed to ask questions at test time. mix means the student is allowed to ask questions for 50 percent of time)
    -homefolder     (default ../data/movieQA_kb, the folder to load the movieQA database)
    -datafolder     (default ../data/AQ_supervised_data, the folder to load train/dev/test data)
    -context        (default true, whether to use vanilla MemN2N or context-based MemN2N)
    -context_num    (default 1, the number of left/right neighbors to be considered as contexts)


## Reinforce

The online reinforcement learning settings described in the paper.

to run the trainer:

    th train_RL.lua [params]


Available options are:

    -batch_size		(default 32, the batch size for model training)
    -token_size		(default 0, number of tokens)
    -init_weight	(default 0.1, initialization weights)
    -N_hop			(default 3, number of hops)
    -lr				(default 0.05, learning rate)
    -thres			(default 40, threshold for gradient clipping)
    -N_iter         (default 14, the number of total iterations to run)
    -StaringFullTraining    (default 10, the number of iterations after which to start training AskQuestion vs notAskQuestion policy)
    -gpu_index		(default 1, which GPU to use)
    -task			(default 1, which task to test)
    -REINFORCE      (default false, where to train the REINFORCE algorithm)
    -REINFORCE_reg  (default 0.1, entropy regularizer for the REINFORCE algorithm)
    -dic_file       (default ../data/movieQA_kb/movieQA.dict, the dictionary)
    -AQcost         (default 0.2, the cost of asking a question)
    -RL_setting     (default good, taking values of good|bad|medium, which type of student to test on)
    -RF_lr          (default 0.0005, learning rate used by the REINFORCE baseline)
    -readFolder     (default ../data/AQ_reinforce_data, the folder to read data from)
    -output_file    (default output.txt, the output file)
    -context        (default true, whether to use vanilla MemN2N or context-based MemN2N)
    -context_num    (default 1, the number of left/right neighbors to be considered as contexts)

## References

* Jiwei Li, Alexander H. Miller, Sumit Chopra, Marc'Aurelio Ranzato and Jason Weston, "[Learning through Dialogue Interactions](https://arxiv.org/abs/1612.04936)", *arXiv:1612.04936 [cs.CL]*.
