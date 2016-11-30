#!/bin/bash
wget "https://s3.amazonaws.com/fair-data/memnn/human_in_the_loop/data.tar.gz" \
&& tar -xzvf data.tar.gz && rm data.tar.gz
