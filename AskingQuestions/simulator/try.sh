#!/bin/bash
mkdir -p ../data/AQ_supervised_gen/
for task in 1 2 3 4 5 6 7 8 9
    do
    for setting in AQ QA mix
        do
        for mode in train test dev
            do
            th runSimulator.lua -mode $mode -prob_correct_final_answer 0.5 -prob_correct_intermediate_answer 0.5 -homefolder ../data/movieQA_kb -setting $setting -task $task -output_dir ../data/AQ_supervised_gen/ &
            done
        done
    done
