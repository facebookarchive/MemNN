#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

th build_dict_hash.lua \
./data/torch/dict-hash.txt ./data/torch/dict.txt ./data/entities_1.txt 1000000

th ../../scripts/build_hash.lua params \
"./data/torch/wiki-w=0-d=3-i-m.txt.hash" \
"./data/torch/wiki-w=0-d=3-i-m.txt.vecarray" \
dictFile="./data/torch/dict-hash.txt" \
memHashFreqCutoff=10000 \
"$@"
