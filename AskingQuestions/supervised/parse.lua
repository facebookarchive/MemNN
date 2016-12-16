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
cmd:option("-dimension",32,"vector dimensionality")
cmd:option("-init_weight",0.1, "initialization weight")
cmd:option("-N_hop",3,"number of MemmNet hops")
cmd:option("-lr",0.05, "initial learning rate")
cmd:option("-thres",40,"threshold for gradient clipping")
cmd:option("-iter_halve_lr",20,"number of iterations after which"
    .."start halving learning rate")
cmd:option("-task",1,"")
cmd:option("-gpu_index",1,"")
cmd:option("-trainSetting","QA","training setting, AQ, QA or mix")
cmd:option("-testSetting","QA","test setting, AQ, QA")
cmd:option("-N_iter",10,"total number of iteration to run")
cmd:option("-homefolder","../data/movieQA_kb","")
cmd:option("-datafolder","../data/AQ_supervised_data","")
cmd:option("-context",true,"vanilla model or context based model")
cmd:option("-context_num",1,"the number of neighbors to be considered as context")


local params= cmd:parse(arg)
params.dic_file=params.homefolder.."/movieQA.dict"
params.trainData=params.datafolder.."/Task"..params.task.."_"..params.trainSetting.."_train.txt"
params.devData=params.datafolder.."/Task"..params.task.."_"..params.testSetting.."_dev.txt"
params.testData=params.datafolder.."/Task"..params.task.."_"..params.testSetting.."_test.txt"

print(params)
return params;
