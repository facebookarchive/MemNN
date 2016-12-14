#!/bin/bash
th train_RL.lua -task 1 -RL_setting bad -lr 0.05 -batch_size 4 -N_iter 20 -StaringFullTraining 10 -AQcost 0.2  
