#!/bin/bash
mkdir -p ../data/AQ_supervised_gen/
for task in 1
    do
    for setting in AQ
        do
        for mode in dev
            do
            th runSimulator.lua -mode $mode -prob_correct_final_answer 0.5 -prob_correct_intermediate_answer 0.5 -homefolder ../data/movieQA_kb -setting $setting -task $task -output_dir ../data/AQ_supervised_gen/
            done
        done
    done
