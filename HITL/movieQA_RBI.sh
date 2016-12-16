#!/bin/bash

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

th online_simulate.lua -dataset movieQA -nepochs 20 -task 3 -batch_size 32 -simulator_batch_size 32 -randomness 0.5 -setting RBI
