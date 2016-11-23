#!/bin/bash
wget "https://s3.amazonaws.com/fair-data/dialog_based_language_learning/data.tar.gz" \
&& tar -xzvf data.tar.gz && rm data.tar.gz
