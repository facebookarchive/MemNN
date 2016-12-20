#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

for set in "train_1" "dev_1" "test_1"
do
    th ../../scripts/build_data.lua params \
    ./data/$set.txt ./data/torch/ ./data/torch/dict.txt "$@"
done

th ../../scripts/build_data.lua params \
./data/wiki-w=0-d=3-i-m.txt ./data/torch ./data/torch/dict.txt "$@"
