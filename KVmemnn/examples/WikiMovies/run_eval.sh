#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

model="./output/kvmemnn-w=0-d=3-i-m-id1-lr=0.005-eDim=100-initWeights=0.1-hops=2-neg=100-TPow=0-rots=false-ltshare=true-metric=dot.model.best_valid_model"

# evaluation specified model
export LUA_PATH="$(pwd)/../../?.lua;;"
th "../../scripts/eval.lua" model modelFilename=$model \
logEveryNSecs=1 debugMode=false allowSaving=false "$@"
