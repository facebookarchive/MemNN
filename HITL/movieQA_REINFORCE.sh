#!/bin/bash
th online_simulate.lua -dataset movieQA -nepochs 20 -task 3 -batch_size 32 -simulator_batch_size 32 -setting RBI -REINFORCE
