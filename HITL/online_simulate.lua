-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require "torch"
require "cunn"
require "cutorch"
require "nngraph"
torch.manualSeed(1111)
cutorch.manualSeed(1111)
local stringx = require('pl.stringx')
local params = require("parse")
model = require("online_memmnet")
local tds = require('tds')
cutorch.setDevice(params.gpu_index)
model:Initial(params)
local setting_string=params.dataset.."_"
    .."task"..params.task
    ..'_model'..params.setting
    .."_sbsz"..params.simulator_batch_size
    .."_mbsz"..params.batch_size
if params.REINFORCE then
    setting_string = setting_string .. '_reg' .. params.REINFORCE_reg ..
        "_RFlr" .. params.RF_lr .. "_REINFORCE"
    --assert(params.simulator_batch_size == params.batch_size)
elseif params.setting=="RBI" then
    setting_string = setting_string ..
        "_eps" .. params.randomness .. '_lf0' ..
        (params.learn_from_0 and 'y' or 'n')
elseif params.setting=="FP" then
    setting_string = setting_string ..
        "_eps" .. params.randomness .. 'balance' ..
        (params.balance and 'y' or 'n').."Nhop"..params.N_hop
end

local output_file = setting_string .. ".result"
local score_history={}
local response_string_index={}
local response_index_string={}

print("setting_string")
print(setting_string)
print("output_file  "..output_file)

print('Configuration')
print(params)
print('Saving at:')
print(output_file)
local fw=io.open(output_file,"w")
print('There are ' .. #model.Data.trainData .. ' questions in the training set')
local num_updates_per_epoch = math.ceil(#model.Data.trainData / params.batch_size)
local total_updates = num_updates_per_epoch * params.nepochs

--assert(math.floor(params.simulator_batch_size / params.batch_size) *
--           params.batch_size == params.simulator_batch_size)
print('Number of updates per epoch ' .. num_updates_per_epoch)
local n_update = 0
local bsz_ratio = params.simulator_batch_size / params.batch_size

local score_history={}
local response_index_string=tds.hash()
response_index_string.count=0

local instance_index=0
for iter = 1, math.ceil(total_updates / bsz_ratio) do
    local story={}
    local simulator_batch={}
    for i=1, bsz_ratio do
        simulator_batch[i]={};
        local predictions={};
        local current_story={}
        local start_index=torch.random(#model.Data.trainData)
        while start_index+params.batch_size>#model.Data.trainData do
            start_index=torch.random(#model.Data.trainData)
        end
        for j=1,params.batch_size do
            start_index=start_index+1;
            local instance=model.Data.trainData[start_index];
            simulator_batch[i][j]=instance;
        end
        model:prepareData(simulator_batch[i]);
        local batch_pred=model:Forward()
        for j=1,params.batch_size do
            answer, correctness =
                model.Data:MakePrediction(batch_pred[j],
                                          params.randomness,simulator_batch[i][j])
            --collect answers and whether they are correct or incorrect
            simulator_batch[i][j].answer=torch.Tensor(1):fill(answer);
            if correctness then
                if params.FP then
                    simulator_batch[i][j].response=simulator_batch[i][j].PosResponse;
                    --using the teacher's response to correct answer
                end
                if params.task==6 then
                    --for task 6, reward is only given for 50 percent of the time
                    if math.random()<0.5 then
                        simulator_batch[i][j].r=torch.Tensor({1})
                    else
                        simulator_batch[i][j].r=torch.Tensor({0})
                    end
                else
                    simulator_batch[i][j].r=torch.Tensor({1})
                end
            else
                if params.FP then
                    simulator_batch[i][j].response=simulator_batch[i][j].NegResponse;
                    --using the teacher's response to incorrect answer
                end
                simulator_batch[i][j].r=torch.Tensor({0})
            end
            if params.balance then
                assert(param.dataset=="babi")
                --the balancing strategy for FP. only applies to babi set
                local response_string=model.Data:printVector(simulator_batch[i][j].response);
                if response_index_string[response_string]==nil then
                    response_index_string.count=response_index_string.count+1
                    response_index_string[response_string]=response_index_string.count;
                    response_index_string[response_index_string.count]=response_string;
                end
                local response_index=response_index_string[response_string];
                if score_history[response_index]==nil then
                    score_history[response_index]={};
                end
                local index=#score_history[response_index]+1;
                score_history[response_index][index]={}
                for i,v in pairs(simulator_batch[i][j])do
                    score_history[response_index][index][i]=v
                end
            end
        end
    end
    for i=1, bsz_ratio do
        local current_batch;
        if params.balance and params.FP then
            current_batch={};
            for i=1,params.batch_size do
                local response_index=torch.random(#score_history);
                local instance=score_history[response_index][torch.random(#score_history[response_index])]
                --model.Data:printInstance(instance)
                current_batch[i]=instance;
            end
        else current_batch=simulator_batch[i];
        end
        local predictions={};
        local current_story={}
        local number_correct = 0
        for j=1,params.batch_size do
            number_correct = number_correct + current_batch[j].r[1]
        end
        if not ((params.setting == "RBI" and not params.REINFORCE) and number_correct == 0) then
            model:batch_train(current_batch)
        end
        n_update = n_update + 1
        if (n_update - 1) % params.log_freq == 0 then
            local valid_acc = model:test("dev")
            local test_acc = model:test("test")
            if params.write then
                fw:write(n_update .. " " .. valid_acc .. " "..test_acc.."\n")
            end
            print(string.format('iter %6d, valid_acc %2.5f',
                                n_update, valid_acc, test_acc))
        end
        if (n_update % num_updates_per_epoch == 0) then
            print('end of epoch ' .. n_update / num_updates_per_epoch)
        end
    end
end
fw:close()
