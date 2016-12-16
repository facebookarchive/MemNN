-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('torch')
local stringx=require("pl.stringx")
local base_memmnet
base_memmnet=require("../supervised/memmnet")
local RL_memmnet = {}
setmetatable(RL_memmnet,{ __index = base_memmnet })
RL_memmnet.Data=require("../reinforce/RL_data")

function RL_memmnet:MeMM_vector()
    local inputs={}
    local memory
    memory=nn.Identity()()
    table.insert(inputs,memory)
    local question=nn.Identity()()
    table.insert(inputs,question)
    local context_mask=nn.Identity()()
    table.insert(inputs,context_mask)
    local u=question;
    local store={};
    local sen_atten;
    for i=1,self.params.N_hop do
        if i==1 then
            u=nn.Replicate(1,3)(u);
        else
            u=nn.Replicate(1,3)(store[i-1]);
        end
        sen_atten=nn.MM()({memory,u});
        sen_atten=nn.Sum(3)(sen_atten);
        sen_atten=nn.SoftMax()(sen_atten);
        sen_atten=nn.CMulTable()({sen_atten,context_mask});
        sen_atten=nn.Normalize(1)(sen_atten);
        sen_atten=nn.Replicate(1,2)(sen_atten);
        local attent_vect=nn.MM()({sen_atten,memory});
        attent_vect=nn.Sum(2)(attent_vect)
        store[i]=nn.CAddTable()({attent_vect,u});
    end
    return nn.gModule(inputs,{store[self.params.N_hop]}):cuda()
end

function RL_memmnet:MeMM_Binary()
    local inputs={}
    local memory=nn.Identity()()
    table.insert(inputs,memory)
    local question=nn.Identity()()
    table.insert(inputs,question)
    local context_mask=nn.Identity()()
    table.insert(inputs,context_mask)
    local memm=self:MeMM_vector()
    local vector=memm(inputs);
    local Linear=nn.Linear(self.params.dimension,2)
    Linear:reset(self.params.init_weight)
    local transform=Linear(vector)
    local LogP=nn.LogSoftMax()(transform)
    return nn.gModule(inputs,{LogP}):cuda()
end

function RL_memmnet:BinaryLookUpTable_()
    local inputs={};
    local context=nn.Identity()();
    table.insert(inputs,context)
    local question=nn.Identity()();
    table.insert(inputs,question)
    local context_table=self.word_table:clone();
    context_v=nn.Sum(2)(context_table(context));
    local question_table=
        context_table:clone('weight','bias');
    local question_v=nn.Sum(2)(question_table(question));
    local module=nn.gModule(inputs,{context_v,question_v});
    return module:cuda()
end

function RL_memmnet:Initial(params_)
    self.params=params_;
    self.f=io.open(self.params.output_file,"w")
    self.Data:process_data(self.params)
    self.params.token_size=200000;
    self.word_table=
        nn.LookupTable(self.params.token_size,self.params.dimension):cuda();
    self.word_table:reset(self.params.init_weight)
    self.word_table.weight[1]:zero()
    self.AQLookUpTable=self:LookUpTable_();
    self.QALookUpTable=self.AQLookUpTable:clone('weight','bias')
    self.BinaryLookUpTable=self:BinaryLookUpTable_();
    self.MeMMBinary=self:MeMM_Binary()
    self.MeMM=self:MeMM_();
    self.Modules={}
    self.Modules[1]=self.AQLookUpTable
    self.Modules[2]=self.QALookUpTable
    self.Modules[3]=self.BinaryLookUpTable
    self.current_lr=self.params.lr;
    self.baseline = nn.Linear(self.params.dimension, 1)
    self.baseline.weight:fill(0)
    self.baseline.bias:fill(0)
    self.baseline:cuda()
    self.mse = nn.MSECriterion()
    self.mse:cuda()
    self.total_reward_value=0;
    self.total_reward_num=0;
    self.regularizer = nn.Sequential()
    local tmp = nn.ConcatTable()
    tmp:add(nn.Identity())
    tmp:add(nn.Exp())
    self.regularizer:add(tmp)
    self.regularizer:add(nn.CMulTable()) -- p log p
    self.regularizer:add(nn.Sum(1, 1))
    self.regularizer:add(nn.Sum(1))
    self.regularizer:cuda()
end

function RL_memmnet:test()
    self.total_scores=0
    self.total_instance=0
    self.total_question_ask=0
    self.total_right_instance=0;
    local cost_QA=0;
    local cost_AQ=0;
    self.index=0
    while true do
        local finish=self:collectBatch()
        if finish==1 then break end
        self:Forward_Policy_AQorQA()
        self:Policy_Answer()
    end
    print("AQ percentage "..self.total_question_ask/self.total_instance)
    self.f:write("AQ percentage "..self.total_question_ask/self.total_instance.."\n")
    print(self.total_right_instance,self.total_instance)
    print("accuray "..self.total_right_instance/self.total_instance)
    print(self.total_right_instance,self.total_instance)
    self.f:write("accuray "..self.total_right_instance/self.total_instance.."\n")
    print("RL score "..self.total_scores/self.total_instance)
    self.f:write("RL score "..self.total_scores/self.total_instance.."\n")
    return self.total_scores/self.total_instance
end

function RL_memmnet:collectBatch()
    local finish=0;
    local good_AQ_lines,good_QA_lines,bad_AQ_lines,bad_QA_lines
    self.current_batch_AQ={}
    self.current_batch_QA={}
    if self.flag=="train" then
        good_AQ_lines,finish,self.Data.good_AQ_train_index=self.Data:GetBatch(self.Data.good_AQ_train_index,self.Data.good_AQ_train_lines)
        --good student asking questions
        good_QA_lines,finish,self.Data.good_QA_train_index=self.Data:GetBatch(self.Data.good_QA_train_index,self.Data.good_QA_train_lines)
        --good student not asking questions
        bad_AQ_lines,finish,self.Data.bad_AQ_train_index=self.Data:GetBatch(self.Data.bad_AQ_train_index,self.Data.bad_AQ_train_lines)
        --bad student asking questions
        bad_QA_lines,finish,self.Data.bad_QA_train_index=self.Data:GetBatch(self.Data.bad_QA_train_index,self.Data.bad_QA_train_lines);
        --bad student not asking questions

    elseif self.flag=="dev" then
        good_AQ_lines,finish,self.Data.good_AQ_dev_index=self.Data:GetBatch(self.Data.good_AQ_dev_index,self.Data.good_AQ_dev_lines)
        good_QA_lines,finish,self.Data.good_QA_dev_index=self.Data:GetBatch(self.Data.good_QA_dev_index,self.Data.good_QA_dev_lines)
        bad_AQ_lines,finish,self.Data.bad_AQ_dev_index=self.Data:GetBatch(self.Data.bad_AQ_dev_index,self.Data.bad_AQ_dev_lines)
        bad_QA_lines,finish,self.Data.bad_QA_dev_index=self.Data:GetBatch(self.Data.bad_QA_dev_index,self.Data.bad_QA_dev_lines);

    elseif self.flag=="test" then
        good_AQ_lines,finish,self.Data.good_AQ_test_index=self.Data:GetBatch(self.Data.good_AQ_test_index,self.Data.good_AQ_test_lines)
        good_QA_lines,finish,self.Data.good_QA_test_index=self.Data:GetBatch(self.Data.good_QA_test_index,self.Data.good_QA_test_lines)
        bad_AQ_lines,finish,self.Data.bad_AQ_test_index=self.Data:GetBatch(self.Data.bad_AQ_test_index,self.Data.bad_AQ_test_lines)
        bad_QA_lines,finish,self.Data.bad_QA_test_index=self.Data:GetBatch(self.Data.bad_QA_test_index,self.Data.bad_QA_test_lines);
    end
    if self.params.RL_setting=="good" then
        self.current_batch_AQ=good_AQ_lines;
        self.current_batch_QA=good_QA_lines
    elseif self.params.RL_setting=="bad" then
        self.current_batch_AQ=bad_AQ_lines;
        self.current_batch_QA=bad_QA_lines;
    elseif self.params.RL_setting=="medium" then
        for i=1,self.params.batch_size do
            if i%2==0 then
                self.current_batch_AQ[#self.current_batch_AQ+1]=good_AQ_lines[i];
                self.current_batch_QA[#self.current_batch_QA+1]=good_QA_lines[i];
            else
                self.current_batch_AQ[#self.current_batch_AQ+1]=bad_AQ_lines[i];
                self.current_batch_QA[#self.current_batch_QA+1]=bad_QA_lines[i];
            end
        end
    end
    return finish;
end

function RL_memmnet:PrepareField(oldname,newname,Dataset)
    local max_length=-1;
    for i=1,#Dataset do
        if Dataset[i][oldname]:size(1)>max_length then
            max_length=Dataset[i][oldname]:size(1)
        end
    end
    self[newname]=torch.Tensor();
    for i=1,#Dataset do
        local ex=Dataset[i];
        local v=torch.Tensor(1,max_length);
        v:sub(1,1,1,ex[oldname]:size(1)):copy(ex[oldname])
        if i==1 then
            self[newname]=v;
        else
            self[newname]=torch.cat(self[newname],v,1);
        end
    end
    self[newname]=self[newname]:cuda()
end

function RL_memmnet:GeneralPrepareContextVector(old_context_name,new_context_name,new_mask_name,Dataset)
    context_length=-100
    context_num=-100
    for i,instance in pairs(Dataset)do
        if #instance[old_context_name]>context_num then
            context_num=#instance[old_context_name]
        end
        for j,v in pairs(instance[old_context_name])do
            if v:size(1)>context_length then
                context_length=v:size(1)
            end
        end
    end
    self[new_mask_name]=torch.Tensor(#Dataset,context_num):fill(0):cuda()
    for i,instance in pairs(Dataset)do
        self[new_mask_name]:sub(i,i,1,#instance[old_context_name]):fill(1);
        local context_Mat=torch.Tensor(context_num,context_length):fill(1):cuda()
        for j,v in pairs(Dataset[i][old_context_name])do
            context_Mat:sub(j,j,1,v:size(1)):copy(v);
        end
        if i==1 then
            self[new_context_name]=context_Mat;
        else
            self[new_context_name]=torch.cat(self[new_context_name],context_Mat,1);
        end
    end
end

function RL_memmnet:Forward_Policy_AQorQA()
    --the REINFORCE model for asking vs not asking questions
    self:GeneralPrepareContextVector("kb_x","kb_word","kb_mask",self.current_batch_QA)
    self:PrepareField("question","query_word_",self.current_batch_QA);
    self.n_instance=#self.current_batch_QA
    local vector_output=self.BinaryLookUpTable:forward({
        self.kb_word,self.query_word_});
    self.kb_v=vector_output[1];
    self.binary_query_v=vector_output[2];
    self.kb_v=self:Reshape2Dto3D(self.kb_v,self.n_instance)
    self.binaryP=self.MeMMBinary:forward({self.kb_v,self.binary_query_v,self.kb_mask});
end

function RL_memmnet:Backward_Policy_AQorQA()
    local d_binary_output=self.MeMMBinary:backward({self.kb_v,self.binary_query_v,self.kb_mask},
        self.d_binary_pred)
    local d_kb_v=d_binary_output[1];
    local d_binary_query_v=d_binary_output[2]
    d_kb_v=self:Reshape3Dto2D(d_kb_v);
    self.BinaryLookUpTable:backward({self.kb_word,self.query_word_},{d_kb_v,d_binary_query_v})
end

function RL_memmnet:Policy_Answer()
    --the REINFORCE model for the final answer
    if self.flag=="train" then
        self.binaryDecision=torch.multinomial(torch.exp(self.binaryP),1);
    elseif self.flag=="test" then
        if self.test_mode=="standard" then
            self.binaryDecision=torch.multinomial(torch.exp(self.binaryP),1);
            --_,self.binaryDecision=torch.max(self.binaryP,2)
        else
            self.binaryDecision=torch.Tensor(self.binaryP:size(1),1):fill(0)
            if self.test_mode=="half" then
                for i=1,self.binaryDecision:size(1) do
                    if math.random()<0.5 then
                        self.binaryDecision[i][1]=1;
                    else self.binaryDecision[i][1]=2;
                    end
                end
            elseif self.test_mode=="AQ" then
                self.binaryDecision:fill(2);
            elseif self.test_mode=="QA" then
                self.binaryDecision:fill(1);
            end
        end
    end
    self.select_current_batch_AQ={}
    self.select_current_batch_QA={}
    self.QA_map={};
    self.AQ_map={}
    for i=1,self.binaryDecision:size(1) do
        self.total_instance=self.total_instance+1
        if self.binaryDecision[i][1]==1 then
            self.select_current_batch_QA[#self.select_current_batch_QA+1]=self.current_batch_QA[i];
            self.QA_map[#self.QA_map+1]=i
        else
            self.select_current_batch_AQ[#self.select_current_batch_AQ+1]=self.current_batch_AQ[i];
            self.AQ_map[#self.AQ_map+1]=i;
            self.total_question_ask=self.total_question_ask+1
        end
    end
    self.d_binary_pred=torch.Tensor(self.binaryP:size()):fill(0):cuda()
    if #self.select_current_batch_QA~=0 then
        --update not-asking-question policy
        self.reward_vector_QA=torch.Tensor(#self.select_current_batch_QA):zero():cuda();
        self.n_instance=#self.select_current_batch_QA;
        self:prepareData(self.select_current_batch_QA);
        local batch_pred=self:Forward(self.QALookUpTable)
        local answer_prediction
        if self.flag=="train" then answer_prediction=torch.multinomial(torch.exp(batch_pred),1)
        elseif self.flag=="test" then _,answer_prediction=torch.max(batch_pred,2);
        end
        for i=1,self.n_instance do
            local answer_predict=answer_prediction[i][1];
            local answer_reward;
            local AQ_reward
            local current_instance=self.select_current_batch_QA[i]
            if current_instance.answers[current_instance.AnswerCandidate[answer_predict]]~=nil  then
                answer_reward=1
                self.total_right_instance=self.total_right_instance+1
            else answer_reward=-1
            end
            if self.iter<self.params.StaringFullTraining then
                AQ_reward=0;
            else
                if self.binaryDecision[self.QA_map[i]][1]==1 then
                    AQ_reward=0;
                else
                    error("QA_map mismatch")
                end
            end
            self.reward_vector_QA[i]=answer_reward-AQ_reward;
            self.total_scores=self.total_scores+self.reward_vector_QA[i]
            self.total_reward_value=self.total_reward_value+self.reward_vector_QA[i]
            self.total_reward_num=self.total_reward_num+1;
        end
        self.d_batch_pred=torch.Tensor(batch_pred:size()):fill(0):cuda();
        self.baseline_v=self:getHopVec()
        baseline = self.baseline:forward(self.baseline_v)
        for i=1,self.n_instance do
            self.d_batch_pred[i][answer_prediction[i][1]]=baseline[i]-self.reward_vector_QA[i]
            self.d_binary_pred[self.QA_map[i]][self.binaryDecision[self.QA_map[i]][1]]=baseline[i]-self.reward_vector_QA[i]
        end
        if self.flag=="train" and self.train_mode=="update_final" then
            self:Backward(self.d_batch_pred,self.QALookUpTable)
        end
        self.baseline:zeroGradParameters()
        local baseline_error = self.mse:forward(baseline:view(self.n_instance),self.reward_vector_QA);
        self.mse:backward(baseline,self.reward_vector_QA);
        self.baseline:backward(self.baseline_v,self.mse.gradInput)
        self.baseline:updateParameters(self.params.RF_lr)
    end
    if #self.select_current_batch_AQ~=0 then
        --update asking-question policy
        self.reward_vector_AQ=torch.Tensor(#self.select_current_batch_AQ):zero():cuda();
        self.n_instance=#self.select_current_batch_AQ;
        self:prepareData(self.select_current_batch_AQ);
        local batch_pred=self:Forward(self.AQLookUpTable)
        local answer_prediction;
        if self.flag=="train" then answer_prediction=torch.multinomial(torch.exp(batch_pred),1)
        elseif self.flag=="test" then _,answer_prediction=torch.max(batch_pred,2);
        end
        for i=1,self.n_instance do
            local answer_predict=answer_prediction[i][1];
            local answer_reward;
            local AQ_reward
            local current_instance=self.select_current_batch_AQ[i]
            if current_instance.answers[current_instance.AnswerCandidate[answer_predict]]~=nil  then
                answer_reward=1
                self.total_right_instance=self.total_right_instance+1
            else answer_reward=-1
            end
            if self.iter<self.params.StaringFullTraining then
                AQ_reward=0;
                --for the first a few iterations, ignore the policy for asking-question vs not-asking-question
            else
                if self.binaryDecision[self.AQ_map[i]][1]==1 then
                    error("AQ_map mismatch")
                else AQ_reward=self.params.AQcost
                end
            end
            self.reward_vector_AQ[i]=answer_reward-AQ_reward;
            self.total_scores=self.total_scores+self.reward_vector_AQ[i]
            self.total_reward_value=self.total_reward_value+self.reward_vector_AQ[i]
            self.total_reward_num=self.total_reward_num+1;
        end
        self.d_batch_pred=torch.Tensor(batch_pred:size()):fill(0):cuda();
        self.baseline_v=self:getHopVec()
        local baseline = self.baseline:forward(self.baseline_v)
        for i=1,self.n_instance do
            self.d_batch_pred[i][answer_prediction[i][1]]=baseline[i]-self.reward_vector_AQ[i]
            self.d_binary_pred[self.AQ_map[i]][self.binaryDecision[self.AQ_map[i]][1]]=baseline[i]-self.reward_vector_AQ[i]
        end
        if self.flag=="train" and self.train_mode=="update_final" then
            self:Backward(self.d_batch_pred,self.AQLookUpTable)
        end
        local baseline_error = self.mse:forward(baseline:view(self.n_instance),self.reward_vector_AQ);
        self.baseline:zeroGradParameters()
        self.mse:backward(baseline,self.reward_vector_AQ);
        self.baseline:backward(self.baseline_v,self.mse.gradInput)
        self.baseline:updateParameters(self.params.RF_lr)
    end
    self.regularizer:forward(self.binaryP);
    self.regularizer:backward(self.binaryP,torch.CudaTensor{self.params.REINFORCE_reg});
    self.d_binary_pred:add(self.regularizer.gradInput)
    --adding entropy to regularize the asking-question/not-asking-question policy
end

function RL_memmnet:train()
    self.base_line_value=0;
    self.base_line_num=0;
    self.iter=0
    local update=0;
    self.train_mode="update_final"
    while true do
        self.iter=self.iter+1
        self.total_scores=0
        self.total_instance=0
        self.total_question_ask=0
        self.total_right_instance=0;
        self.index=0;
        while true do
            for i=1,#self.Modules do
                self.Modules[i]:zeroGradParameters()
            end
            self.flag="train"
            local finish=self:collectBatch()
            if finish==1 then
                break;
            end
            update=update+1;
            if self.iter<self.params.StaringFullTraining then
                self.binaryP=torch.Tensor(self.params.batch_size,2):fill(0.5):cuda()
                --equal possibility for AQ and QA for the first a few iterations so that both policies
                --can get learned
            else
                self:Forward_Policy_AQorQA()
            end
            self:Policy_Answer()
            if self.iter>=self.params.StaringFullTraining then
                --do not update asking-question vs not-asking-question policy for the first a few iterations
               self:Backward_Policy_AQorQA()
            end
            self:update()
        end
        print("iter  "..self.iter)
        self.f:write("iter  "..self.iter.."\n")
        self.flag="test"
        if self.iter<self.params.StaringFullTraining then
            self.test_mode="half"
        else self.test_mode="standard"
            self.train_mode="not_update_final"
        end
        self:test()
        if self.iter==self.params.StaringFullTraining-1 then
            print("only AQ acc")
            self.test_mode="AQ"
            --test when the student always asks a question
            self:test()
            print("only QA acc")
            --test when the student never asks a question
            self.test_mode="QA"
            self:test()
        end
        if self.iter==self.params.N_iter then
            break;
        end
    end
    self.f:close()
end

return RL_memmnet
