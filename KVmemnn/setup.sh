#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

# compile libmemnn.so
gcc -O3 -std=c99 -shared -o ./library/c/libmemnn.so -fPIC ./library/c/*.c
# link libmemnn.so to lua lib
s=$(which luarocks)
ln -s $(pwd)/library/c/libmemnn.so ${s%luarocks}/../lib/libmemnn.so
