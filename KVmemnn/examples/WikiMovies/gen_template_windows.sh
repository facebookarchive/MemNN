#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

for type in "all_conj=0.0_coref=0.0" "all_conj=0.5_coref=0.8" "one_conj=0.0_coref=0.0" "one_conj=0.0_coref=1.0" "one_conj=0.7_coref=0.0"; do
    # generate single dictonary version of synthetic data
    # python3 wikiwindows.py --num_threads 1 --window_size 3 --double_dict 0 --inverse --movie_in_all \
    # --entities ./movieqa/knowledge_source/entities.txt \
    # --output_file ./data/$type-w=3-d=0-i-m.txt \
    # ./movieqa/knowledge_source/wiki_entities/synthetic/movie_statements_entities=wiki-entities_templates=$type.txt

    # generate double dictionary version of synthetic data
    python3 wikiwindows.py --num_threads 1 --window_size 0 --double_dict 20 --inverse --movie_in_all \
    --entities ./movieqa/knowledge_source/entities.txt \
    --output_file ./data/$type-w=0-d=20-i-m.txt \
    ./movieqa/knowledge_source/wiki_entities/synthetic/movie_statements_entities=wiki-entities_templates=$type.txt &
done
