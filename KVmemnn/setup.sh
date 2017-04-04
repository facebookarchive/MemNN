#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

luarocks show lzmq 1>/dev/null 2>/dev/null
if [ $? -ne 0 ]; then
    luarocks install lzmq
    if [ $? -ne 0 ]; then
        echo "Error: resolve then run luarocks install lzmq"
        exit 1
    fi
fi

luarocks show threads 1>/dev/null 2>/dev/null
if [ $? -ne 0 ]; then
    luarocks install threads
    if [ $? -ne 0 ]; then
        echo "Error: resolve then run luarocks install threads"
        exit 1
    fi
fi

# compile libmemnn.so
gcc -O3 -std=c99 -shared -o ./library/c/libmemnn.so -fPIC ./library/c/*.c
if [ $? -ne 0 ]; then
    echo "Error compiling memnn library"
    exit 1
fi

# link libmemnn.so to lua lib
s=$(which luarocks)
if [ -e ${s%luarocks}/../lib/libmemnn.so ]; then
    rm ${s%luarocks}/../lib/libmemnn.so
fi
ln -s $(pwd)/library/c/libmemnn.so ${s%luarocks}/../lib/libmemnn.so
if [ $? -ne 0 ]; then
    echo "Error linking memnn library to luarocks lib directory"
    exit 1
fi
