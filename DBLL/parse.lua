-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-batch_size",32,"batch size")
cmd:option("-token_size",0,"number of tokens")
cmd:option("-dimension",20,"vector dimensionality")
cmd:option("-init_weight",0.1, "initialization weight")
cmd:option("-N_hop",3,"number of MemmNet hops")
cmd:option("-lr",0.01, "initial learning rate")
cmd:option("-thres",40,"threshold for gradient clipping")
cmd:option("-iter_halve_lr",20,"number of iterations after which"
    .."start halving learning rate")
cmd:option("-task",1,"")
cmd:option("-gpu_index",1,"")
cmd:option("-policy",0.5,"the rate of correct vs random answers, taking values 0.01, 0.1 or 0.5")
cmd:option("-N_iter",20,"total number of iteration to run")
cmd:option("-dic_file","/mnt/vol/gfsai-east/ai-group/users/jiwei/movieQA/movieQA_dict.txt",
    "path to the dictionary file")
cmd:option("-beta",true,"whether use beta for FP");
cmd:option("-negative",5,"number of negative samples");
cmd:option("-dataset","babi","the dataset to use, whether"
    .."it is babi or movieQA")
cmd:option("-setting","RBI","the model setting, taking values RBI, FP, IM or RBI+FP")

local params= cmd:parse(arg)
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

if params.setting=="RBI+FP" then
    params.policyGrad=true;
    params.FP=true;
    params.N_hop=2
elseif params.setting=="RBI" then
    params.policyGrad=true;
    params.FP=false;
    params.N_hop=3
elseif params.setting=="FP"then
    params.FP=true;
    params.policyGrad=false
    params.N_hop=1
elseif params.setting=="IM" then
    params.policyGrad=false;
    params.FP=false
end

if params.dataset=="movieQA" then
    params.dic_file="./data/movieQA/dict.txt"
    params.trainData="./data/movieQA/movieQA_p"
        ..params.policy.."_"..Tasks[params.task]..'_train.txt'
    params.devData="./data/movieQA/movieQA_dev.txt"
    params.testData="./data/movieQA/movieQA_test.txt"
    params.dimension=50;
    if params.setting=="RBI" or params.setting=="IM" then
        params.lr=0.05;
    elseif params.setting=="RBI+FP" then
        params.lr=0.025;
    elseif params.setting=="FP"then
        params.lr=0.01
    end
elseif params.dataset=="babi" then
    params.dic_file="./data/babi/dict.txt"
    params.trainData="./data/babi/babi1_p"
        ..params.policy.."_"..Tasks[params.task]..'_train.txt'
    params.devData="./data/babi/babi1_p"
        ..params.policy.."_"..Tasks[params.task]..'_dev.txt'
    params.testData="./data/babi/babi1_p"
        ..params.policy.."_"..Tasks[params.task]..'_test.txt'
    params.lr=0.01
end
print(params)
return params;
