#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

mkdir -p ./output
export LUA_PATH="$(pwd)/../../?.lua;;"
th "../../scripts/train.lua" params "$@"
