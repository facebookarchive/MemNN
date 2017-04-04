#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

set -e

luarocks install lzmq
luarocks install threads

# compile libmemnn.so
gcc -O3 -std=c99 -shared -o ./library/c/libmemnn.so -fPIC ./library/c/*.c
# link libmemnn.so to lua lib
s=$(which luarocks)
if [ -e ${s%luarocks}/../lib/libmemnn.so ]; then
    rm ${s%luarocks}/../lib/libmemnn.so
fi
ln -s $(pwd)/library/c/libmemnn.so ${s%luarocks}/../lib/libmemnn.so
