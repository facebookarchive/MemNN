-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('torch')
local tds = require('tds')
local stringx = require('pl.stringx')
local base_data = require('../supervised/data')
local RL_data={}
setmetatable(RL_data,{ __index = base_data })

function RL_data:process_data(params)
    self.params=params;
    self:load_dict()
    self.mode="train"
    self.good_AQ_train_lines=self:load_lines(params.good_AQ_trainData)
    --dataset for good student asking questions
    self.good_AQ_train_index=0
    self.good_QA_train_lines=self:load_lines(params.good_QA_trainData)
    --dataset for good student not asking questions
    self.good_QA_train_index=0
    self.mode="dev"
    self.good_AQ_dev_lines=self:load_lines(params.good_AQ_devData)
    self.good_AQ_dev_index=0
    self.good_QA_dev_lines=self:load_lines(params.good_QA_devData)
    self.good_QA_dev_index=0
    self.mode="test"
    self.good_AQ_test_lines=self:load_lines(params.good_AQ_testData)
    self.good_AQ_test_index=0
    self.good_QA_test_lines=self:load_lines(params.good_QA_testData)
    self.good_QA_test_index=0
    self.mode="train"
    self.bad_AQ_train_lines=self:load_lines(params.bad_AQ_trainData)
    --dataset for bad student asking questions
    self.bad_AQ_train_index=0
    self.bad_QA_train_lines=self:load_lines(params.bad_QA_trainData)
    --dataset for bad student not asking questions
    self.bad_QA_train_index=0
    self.mode="dev"
    self.bad_AQ_dev_lines=self:load_lines(params.bad_AQ_devData)
    self.bad_AQ_dev_index=0
    self.bad_QA_dev_lines=self:load_lines(params.bad_QA_devData)
    self.bad_QA_dev_index=0
    self.mode="test"
    self.bad_AQ_test_lines=self:load_lines(params.bad_AQ_testData)
    self.bad_AQ_test_index=0
    self.bad_QA_test_lines=self:load_lines(params.bad_QA_testData)
    self.bad_QA_test_index=0
end

function RL_data:load_lines(filename)
    local f=io.open(filename);
    if f == nil then
        error("cannot load file: " .. filename)
    end
    local lines={};
    while true do
        local s=f:read("*line")
        if s == nil then break; end
        lines[#lines+1]=self:process_string(s);
    end
    f:close();
    return lines
end


function RL_data:GetBatch(counter,lines)
    local finish=0
    local dataset={}
    local current_lines={};
    while true do
        counter=counter+1
        local line=lines[counter];
        local line=self:process_string(line);
        current_lines[#current_lines+1]=line
        if counter~=#lines then
            local next_line=lines[counter+1];
            local t=next_line:find(" ")
            local next_line_index=tonumber(next_line:sub(1,t-1));
            if next_line_index==1 then
                local instance=self:Lines2instanceMovieQA(current_lines)
                dataset[#dataset+1]=instance
                current_lines={}
                if #dataset==self.params.batch_size then
                    break;
                end
            end
        else
            finish=1;counter=0;
            break
        end
    end
    return dataset,finish,counter
end

return RL_data
