-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local stringx = require('pl.stringx')
local tds = require('tds')
local cmd = torch.CmdLine()
-- parameters for memory nets
cmd:option("-batch_size",32,"batch size")
cmd:option("-token_size",0,"number of tokens")
cmd:option("-init_weight",0.01, "initialization weight")
cmd:option("-N_hop",3,"number of MemmNet hops")
cmd:option("-nepochs",20, "number of epochs")
cmd:option("-thres",40,"threshold for gradient clipping")
cmd:option("-negative",5,"number of negative samples");
cmd:option("-gpu_index",1,"the index of GPU to use")
cmd:option("-dataset","movieQA","the dataset to use, whether"
    .."it is babi or movieQA")
cmd:option("-setting","RBI","the model setting")
cmd:option("-task",3,"task 2,3,4,6")
-- parameters for movie dataset
cmd:option("-randomness",0.2,"-random exploration rate")
cmd:option("-simulator_batch_size",32,"simulator batch size")
cmd:option("-REINFORCE",false, "whether to enable REINFORCE for training")
cmd:option("-REINFORCE_reg", 0.1, "entropy regularizer for the REINFORCE algorithm")
cmd:option("-RF_lr", 0.0005, "lr used by REINFORCE baseline (multiplied by lr)")
cmd:option("-log_freq", 200, "how often we log")
cmd:option("-balance",false,"enable label balancing for FP")

local babi_name_match={}

local Tasks={
    "rl1_pure_imitation",
    "rl2_pos_neg",
    "rl3_with_ans",
    "rl4_with_hints",
    "rl5_told_sf",
    "rl6_only_some_rewards",
    "rl7_no_feedback",
    "rl8_imitation_plus_rl",
    "rl9_ask_for_answer",
    "rl10_ask_for_sf",
}

local params= cmd:parse(arg)

params.tasks=tds.hash()
if params.setting=="RBI+FP" then
    params.policyGrad=true;
    params.FP=true;
elseif params.setting=="RBI" then
    params.policyGrad=true;
    params.FP=false;
elseif params.setting=="FP"then
    params.FP=true;
    params.policyGrad=false
    params.N_hop=1
elseif params.setting=="IM" then
    params.policyGrad=false;
    params.FP=false
end

if params.dataset=="movieQA" then
    params.dic_file="./data/movieQA.dict"
    params.trainData="./data/movieQA_"..Tasks[params.task].."_train.txt"
    params.devData="./data/movieQA_"..Tasks[params.task].."_dev.txt"
    params.testData="./data/movieQA_"..Tasks[params.task].."_test.txt"
    params.IncorrectResponse="./data/movieQA_"..Tasks[params.task].."_incorrect_feedback"
    params.dimension=50;
    if params.setting=="RBI" then
        params.lr=0.2;
    else params.lr=0.05
    end
else
    params.dic_file="./data/babi.dict"
    params.trainData="./data/babi1_"..Tasks[params.task].."_train.txt"
    params.devData="./data/babi1_"..Tasks[params.task].."_dev.txt"
    params.testData="./data/babi1_"..Tasks[params.task].."_test.txt"
    params.IncorrectResponse="./data/babi1_"..Tasks[params.task].."_incorrect_feedback"
    params.dimension=20;
    params.lr=0.01
end

return params
