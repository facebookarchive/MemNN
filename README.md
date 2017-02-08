# Memory-Augmented Neural Networks

This project contains implementations of memory augmented neural networks.
This includes code in the following subdirectories:


* [MemN2N-lang-model](MemN2N-lang-model): This code trains MemN2N model for language modeling, see Section 5 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". This code is implemented in [Torch7](http://torch.ch/) (written in Lua); more documentation is given in the README in that subdirectory.


* [MemN2N-babi-matlab](MemN2N-babi-matlab): The code for the MemN2N bAbI task experiments of Section 4 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". This code is implemented in Matlab; more documentation is given in the README in that subdirectory.


* [DBLL](DBLL): Code to train MemN2N on tasks from the paper "[Dialog-based Language Learning](https://arxiv.org/abs/1604.06045)". This code is implemented in [Torch7](http://torch.ch); more documentation is given in the README in that subdirectory.


* [HITL](HITL): Code to train MemN2N on tasks from the paper "[Dialogue Learning With Human-in-the-Loop](https://arxiv.org/abs/1611.09823)". This code is implemented in [Torch7](http://torch.ch); more documentation is given in the README in that subdirectory.


* [AskingQuestions](AskingQuestions): Code to train MemN2N on tasks from the paper "[Learning through Dialogue Interactions](https://arxiv.org/abs/1612.04936)". This code is implemented in [Torch7](http://torch.ch); more documentation is given in the README in that subdirectory.


* [KVmemnn](KVmemnn): Code to train MemN2N on tasks from the paper "[Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126)". This code is implemented in [Torch7](http://torch.ch); more documentation is given in the README in that subdirectory.


* [EntNet-babi](EntNet-babi): Code to train an Entity Network on bAbI tasks, as described in the paper "[Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969)". This code is implemented in [Torch7](http://torch.ch); more documentation is given in the README in that subdirectory.


### Other 3rd party implementations
* [python-babi](https://github.com/vinhkhuc/MemN2N-babi-python): MemN2N implemenation on bAbI tasks with very nice interactive demo.
* [theano-babi](https://github.com/npow/MemN2N): MemN2N implementation in Theano for bAbI tasks.
* [tf-lang](https://github.com/carpedm20/MemN2N-tensorflow): MemN2N language model implementation in TensorFlow.
* [tf-babi](https://github.com/domluna/memn2n): Another MemN2N implementation of MemN2N in TensorFlow, but for bAbI tasks.
