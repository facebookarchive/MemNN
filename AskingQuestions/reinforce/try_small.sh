#!/bin/bash

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

th train_RL.lua -task 1 -RL_setting bad -lr 0.05 -batch_size 4 -N_iter 20 -StaringFullTraining 10 -AQcost 0.2
