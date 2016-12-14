#!/bin/bash
wget "https://s3.amazonaws.com/fair-data/memnn/asking_questions/data.tar.gz" \
	&& tar -xzvf data.tar.gz && rm data.tar.gz
