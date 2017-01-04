#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

wget "https://s3.amazonaws.com/fair-data/memnn/kvmemnn/output.tar.gz" \
&& tar -xzvf output.tar.gz && rm output.tar.gz
