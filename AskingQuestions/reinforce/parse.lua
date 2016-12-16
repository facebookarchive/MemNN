-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-batch_size",64,"batch size")
cmd:option("-token_size",0,"number of tokens")
cmd:option("-dimension",32,"vector dimensionality")
cmd:option("-init_weight",0.01, "initialization weight")
cmd:option("-N_hop",3,"number of MemmNet hops")
cmd:option("-lr",0.05, "initial learning rate")
cmd:option("-thres",40,"threshold for gradient clipping")
cmd:option("-iter_halve_lr",20,"number of iterations after which"
    .."start halving learning rate")
cmd:option("-N_iter",14,"total number of iteration to run")
cmd:option("-StaringFullTraining",10,"default 10, the number of iterations after which to start training AskQuestion vs notAskQuestion policy")
cmd:option("-task",1,"task")
cmd:option("-dic_file","../data/movieQA_kb/movieQA.dict","")
cmd:option("-AQcost",0.2,"the cost of asking a question")
cmd:option("-RL_setting","good","")
cmd:option("-REINFORCE_reg",0.1,"")
cmd:option("-RF_lr",0.0005, "lr used by REINFORCE baseline (multiplied by lr)")
cmd:option("-REINFORCE_reg", 0.1, "entropy regularizer for the REINFORCE algorithm")
cmd:option("-readFolder","../data/AQ_reinforce_data","")
cmd:option("-output_file","output.txt","output file")
cmd:option("-context","true","use the context model")
cmd:option("-context_num",1,"")
local params= cmd:parse(arg)

params.good_AQ_trainData=params.readFolder.."/task"..params.task.."_good_AQ_train.txt"
params.good_QA_trainData=params.readFolder.."/task"..params.task.."_good_QA_train.txt"
params.bad_AQ_trainData=params.readFolder.."/task"..params.task.."_bad_AQ_train.txt"
params.bad_QA_trainData=params.readFolder.."/task"..params.task.."_bad_QA_train.txt"
params.good_AQ_devData=params.readFolder.."/task"..params.task.."_good_AQ_dev.txt"
params.good_QA_devData=params.readFolder.."/task"..params.task.."_good_QA_dev.txt"
params.bad_AQ_devData=params.readFolder.."/task"..params.task.."_bad_AQ_dev.txt"
params.bad_QA_devData=params.readFolder.."/task"..params.task.."_bad_QA_dev.txt"
params.good_AQ_testData=params.readFolder.."/task"..params.task.."_good_AQ_test.txt"
params.good_QA_testData=params.readFolder.."/task"..params.task.."_good_QA_test.txt"
params.bad_AQ_testData=params.readFolder.."/task"..params.task.."_bad_AQ_test.txt"
params.bad_QA_testData=params.readFolder.."/task"..params.task.."_bad_QA_test.txt"

print(params)
return  params
