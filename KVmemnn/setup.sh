#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

gcc -O3 -std=c99 -shared -o ./library/c/libmemnn.so -fPIC ./library/c/*.c
