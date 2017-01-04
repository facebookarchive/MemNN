#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

model="./output/pretrained-eDim=500.model.best_valid_model"

# evaluation specified model
export LUA_PATH="$(pwd)/../../?.lua;;"
th "../../scripts/eval.lua" model modelFilename=$model \
logEveryNSecs=1 debugMode=false allowSaving=false numThreads=1 "$@"
