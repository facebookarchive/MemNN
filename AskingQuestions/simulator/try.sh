#!/bin/bash

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

mkdir -p ../data/AQ_supervised_gen/
for task in 1 2 3 4 5 6 7 8 9
    do
    for setting in AQ QA mix
        do
        for mode in train test dev
            do
            th runSimulator.lua -mode $mode -prob_correct_final_answer 0.5 -prob_correct_intermediate_answer 0.5 -homefolder ../data/movieQA_kb -setting $setting -task $task -output_dir ../data/AQ_supervised_gen/ &
            done
        done
    done
