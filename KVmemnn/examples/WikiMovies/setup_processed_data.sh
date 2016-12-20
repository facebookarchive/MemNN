#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

wget "https://s3.amazonaws.com/fair-data/memnn/kvmemnn/data.tar.gz" \
&& tar -xzvf data.tar.gz && rm data.tar.gz
