#!/bin/bash

mkdir -p data/

file=tasks_1-20_v1-2.tar.gz
wget http://www.thespermwhale.com/jaseweston/babi/$file
mv $file data/
cd data
tar -xvf $file
rm $file
cd ..