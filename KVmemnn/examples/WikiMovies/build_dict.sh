#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

th ../../scripts/build_dict.lua params \
./data/wiki-w=0-d=3-i-m.txt,./data/train_1.txt \
./data/torch/dict.txt ./data/entities_1.txt "$@"
