#!/bin/bash

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

th online_simulate.lua -dataset babi -nepochs 100 -task 3 -batch_size 1 -simulator_batch_size 1 -randomness 0.4 -setting FP
