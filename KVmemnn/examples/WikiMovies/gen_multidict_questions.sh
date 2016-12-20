#!/bin/bash
# Copyright 2004-present Facebook. All Rights Reserved.

# note: this takes a long time to run

# just symlink for single dictionary version of questions
# ln -s ./movieqa/questions/wiki_entities/wiki-entities_qa_train.txt ./data/train.txt
# ln -s ./movieqa/questions/wiki_entities/wiki-entities_qa_dev.txt ./data/dev.txt
# ln -s ./movieqa/questions/wiki_entities/wiki-entities_qa_test.txt ./data/test.txt

# generate double dictionary version of questions
python3 query_multidict.py --num_threads 2 \
--entities ./movieqa/knowledge_source/entities.txt \
--output_file ./data/train_1.txt \
./movieqa/questions/wiki_entities/wiki-entities_qa_train.txt &

python3 query_multidict.py --num_threads 1 \
--entities ./movieqa/knowledge_source/entities.txt \
--output_file ./data/dev_1.txt \
./movieqa/questions/wiki_entities/wiki-entities_qa_dev.txt &

python3 query_multidict.py --num_threads 1 \
--entities ./movieqa/knowledge_source/entities.txt \
--output_file ./data/test_1.txt \
./movieqa/questions/wiki_entities/wiki-entities_qa_test.txt &

awk '{ print $0 ; print "1:"$0 }' ./movieqa/knowledge_source/entities.txt > ./data/entities_1.txt
