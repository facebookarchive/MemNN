#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

wget http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz \
&& tar -xzvf movieqa.tar.gz && rm movieqa.tar.gz
mkdir -p ./data/torch
