Dialog-based Langauge Learning
https://arxiv.org/abs/1604.06045

From this directory, first run ./setup_data.sh to download the data (368M download, unpacks to 4.8GB).

Then, use one of the try_*.sh commands to train the model on one of the datasets.

Change the parameters of the try_*.sh commands to switch datasets (e.g. change
policy, switch tasks, etc). See parse.lua for how the parameters select a
dataset.

Note that this implementation uses the luarocks packages cutorch, cunn, nngraph, torchx, and tds.
