#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

export LUA_PATH="$(pwd)/../../?.lua;;"

# use saved model file to set params
model="$./output/kvmemnn-w=0-d=3-i-m-id1-lr=0.005-eDim=100-initWeights=0.1-hops=2-neg=100-TPow=0-rots=false-ltshare=true-metric=dot.model.best_valid_model"
th -i "../../scripts/interactive.lua" model modelFilename=$model "$@"

# use options file to set params
# th -i "../../scripts/interactive.lua" params "$@"
