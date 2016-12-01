#!/bin/bash
wget "https://s3.amazonaws.com/fair-data/memnn/human_in_the_loop/turk_data.tar.gz" \
&& tar -xzvf turk_data.tar.gz && rm turk_data.tar.gz
