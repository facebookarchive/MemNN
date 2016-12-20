#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

# generate single dictonary version of wikipedia data
# python3 wikiwindows.py --num_threads 4 --window_size 3 --double_dict 0 --inverse --movie_in_all \
# --entities ./movieqa/knowledge_source/entities.txt \
# --output_file ./data/wiki-w=3-d=0-i-m.txt \
# ./movieqa/knowledge_source/wiki.txt

# generate double dictionary version of wikipedia data
python3 wikiwindows.py --num_threads 4 --window_size 0 --double_dict 3 --inverse --movie_in_all \
--entities ./movieqa/knowledge_source/entities.txt \
--output_file ./data/wiki-w=0-d=3-i-m.txt \
./movieqa/knowledge_source/wiki.txt
