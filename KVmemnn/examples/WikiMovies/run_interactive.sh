#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

export LUA_PATH="$(pwd)/../../?.lua;;"

# use saved model file to set params
model="./output/pretrained-eDim=500.model.best_valid_model"
#th -i "../../scripts/interactive.lua" model modelFilename=$model "$@"

# use options file to set params
 th -i "../../scripts/interactive.lua" params "$@"
