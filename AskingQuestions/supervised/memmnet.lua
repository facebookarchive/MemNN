-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require "cutorch"
require "nngraph"
require "torch"
local memmnet={};
memmnet.Data=require("../supervised/data")

function memmnet:MeMM_()
    local inputs={}
    local memory
    memory=nn.Identity()()
    table.insert(inputs,memory)
    local question=nn.Identity()()
    table.insert(inputs,question)
    local answer_v=nn.Identity()()
    table.insert(inputs,answer_v)
    local context_mask=nn.Identity()()
    local context_mask_p = nn.MulConstant(1e8)(nn.AddConstant(-1)(context_mask))
    table.insert(inputs,context_mask)
    local word_mask,word_mask_p
    word_mask=nn.Identity()()
    word_mask_p = nn.MulConstant(1e8)(nn.AddConstant(-1)(word_mask))
    table.insert(inputs,word_mask)
    local u=question;
    local store={};
    local sen_atten;
    local attens={}
    for i=1,self.params.N_hop do
        if i==1 then
            u=nn.Replicate(1,3)(u);
        else
            u=nn.Replicate(1,3)(store[i-1]);
        end
        sen_atten=nn.MM()({memory,u});
        sen_atten=nn.Sum(3)(sen_atten);
        sen_atten = nn.CAddTable(){sen_atten, context_mask_p}
        sen_atten=nn.SoftMax()(sen_atten);
        table.insert(attens,sen_atten);
        sen_atten=nn.Replicate(1,2)(sen_atten);
        local attent_vect=nn.MM()({sen_atten,memory});
        attent_vect=nn.Sum(2)(attent_vect)
        store[i]=nn.CAddTable()({attent_vect,u}):annotate{name = 'hopvec' .. i}
    end
    local atten;
    local expand=nn.Replicate(1,3)(store[self.params.N_hop]);
    atten=nn.MM()({answer_v,expand});
    atten=nn.Sum(3)(atten);
    atten = nn.CAddTable(){atten, word_mask_p}
    pred=nn.LogSoftMax()(atten);
    local outputs={}
    table.insert(outputs,pred)
    local pred_response;
    local module=nn.gModule(inputs,outputs);
    return module:cuda();
end

function memmnet:getHopVec()
    local found = nil
    for _, node in pairs(self.MeMM.forwardnodes) do
        if node.data.annotations.name == 'hopvec'..self.params.N_hop then
            found = node.data.module.output
        end
    end
    return found
end

function memmnet:LookUpTable_()
    local inputs={};
    local context_word=nn.Identity()();
    table.insert(inputs,context_word)
    local question_word=nn.Identity()();
    table.insert(inputs,question_word)
    local question_table=
        self.word_table:clone('weight','bias');
    local question_v=nn.Sum(2)(question_table(question_word));
    local answer_word=nn.Identity()();
    local answer_table=self.word_table:clone('weight','bias');
    table.insert(inputs,answer_word)
    local answer_v=nn.Sum(2)(answer_table(answer_word))
    local context_table=self.word_table:clone('weight','bias');
    local context_v=nn.Sum(2)(context_table(context_word));
    local context_length
    if self.params.context then
        context_length=nn.Identity()();
        table.insert(inputs,context_length);
        answer_v=nn.CMulTable()({answer_v,context_length})
        --normalize the context vector by the number of words
    end
    local module=nn.gModule(inputs,{context_v,question_v,answer_v});
    return module:cuda();
end

function memmnet:PrepareNegativeResponse(Dataset)
    for i,instance in pairs(Dataset)do
        if instance.response:size(1)>self.Data.MaxResponseLength then
            self.Data.MaxResponseLength=instance.response:size(1);
        end
    end
    self.response_word=torch.Tensor(#Dataset*(1+self.params.negative),self.Data.MaxResponseLength):fill(1)
    for i,ex in pairs(Dataset)do
        local line_index=(i-1)*(1+self.params.negative)+1
        self.response_word:sub(line_index,line_index,1,ex.response:size(1)):copy(ex.response)
        for j=1,self.params.negative do
            local line_index=(i-1)*(1+self.params.negative)+1+j;
            local negative_response=self.Data.responsePool[torch.random(#self.Data.responsePool)];
            self.response_word:sub(line_index,line_index,1,negative_response:size(1)):copy(negative_response)
        end
        self.Data.responsePool[torch.random(#self.Data.responsePool)]=ex.response;
    end
    self.response_word=self.response_word:cuda()
end

function memmnet:prepareAnswer(Dataset)
    assert(self.params.context~=nil)
    if not self.params.context then
        --vanilla model
        local max_length=-1;
        for i=1,#Dataset do
            local ex=Dataset[i]
            local length=#Dataset[i].AnswerCandidate;
            if length>max_length then
                max_length=length;
            end
        end
        self.answer_word=torch.Tensor(
            #Dataset,max_length):fill(1):cuda();
        self.answer_mask=torch.Tensor(
            #Dataset,max_length):fill(0):cuda();
        for i=1,#Dataset do
            local ex=Dataset[i]
            self.answer_word[{{i},{1,#ex.AnswerCandidate}}]
            :copy(torch.Tensor(ex.AnswerCandidate));
            self.answer_mask[{{i},{1,#ex.AnswerCandidate}}]:fill(1);
        end
        self.answer_word=torch.reshape(self.answer_word,
            self.answer_word:size(1)*self.answer_word:size(2),1);
    else
        local answer_context={};
        for i,ex in pairs(Dataset)do
            answer_context[i]=self.Data:GetAnswerContext(ex)
        end
        local max_length=-1;
        local max_sen_length=0;
        for i=1,#Dataset do
            local ex=Dataset[i]
            local length=#answer_context[i]
            if length>max_length then
                max_length=length;
            end
            for j,v in pairs(answer_context[i])do
                if answer_context[i][j]:size(1)>max_sen_length then
                    max_sen_length=answer_context[i][j]:size(1)
                end
            end
        end
        self.answer_word=torch.Tensor(
            #Dataset,max_length,max_sen_length):fill(1):cuda();
        self.answer_mask=torch.Tensor(
            #Dataset,max_length):fill(0):cuda();
        self.context_length={}
        for i=1,#Dataset do
            local ex=Dataset[i]
            for j=1,#answer_context[i] do
                self.answer_word[{{i},{j},{1,answer_context[i][j]:size(1)}}]:copy(
                answer_context[i][j])
                self.context_length[#self.context_length+1]=1/answer_context[i][j]:size(1)
            end
            for j=#answer_context[i]+1,max_length do
                self.context_length[#self.context_length+1]=1;
            end
            self.answer_mask[{{i},{1,#answer_context[i]}}]:fill(1)
        end
        self.context_length=torch.Tensor(self.context_length)
            :view(#self.context_length,1)
            :expand(#self.context_length,self.params.dimension):cuda()
        self.answer_word=self:Reshape3Dto2D(self.answer_word)
    end
end

function memmnet:PrepareContextVector(Dataset)
    local context_length=-100
    local context_num=-100
    for i,instance in pairs(Dataset)do
        if #instance["hist_x"]>context_num then
            context_num=#instance["hist_x"]
        end
        for j,v in pairs(instance["hist_x"])do
            if v:size(1)>context_length then
                context_length=v:size(1)
            end
        end
    end
    self.context_mask=torch.Tensor(#Dataset,context_num):fill(0):cuda()
    for i,instance in pairs(Dataset)do
        self.context_mask:sub(i,i,1,#instance.hist_x):fill(1);
        local context_Mat=torch.Tensor(context_num,context_length):fill(1):cuda()
        for j,v in pairs(Dataset[i].hist_x)do
            context_Mat:sub(j,j,1,v:size(1)):copy(v);
        end
        if i==1 then
            self.context_word=context_Mat;
        else
            self.context_word=torch.cat(self.context_word,context_Mat,1);
        end
    end
end

function memmnet:PrepareQues(Dataset)
    local max_length=-1;
    for i,instance in pairs(Dataset)do
        if instance.question:size(1)>max_length then
            max_length=instance.question:size(1)
        end
    end
    self.query_word=torch.Tensor(#Dataset,max_length):fill(1);
    for i,instance in pairs(Dataset)do
        self.query_word:sub(i,i,1,instance.question:size(1)):copy(instance.question);
    end
    self.query_word=self.query_word:cuda()
end

function memmnet:prepareData(Dataset)
    self:PrepareContextVector(Dataset)
    self:PrepareQues(Dataset)
    self.reward_vector=torch.Tensor(#Dataset):cuda();
    for i,instance in pairs(Dataset)do
        self.reward_vector[i]=instance.r[1]
    end
    self:prepareAnswer(Dataset)
end

function memmnet:Reshape2Dto3D(vector,n1)
    return torch.reshape(vector,
        n1,vector:size(1)/n1,
        self.params.dimension);
end

function memmnet:Reshape3Dto2D(vector)
    return torch.reshape(vector,
        vector:size(1)*vector:size(2),
        vector:size(3))
end

function memmnet:Forward(LookUpTable)
    local vector_output
    if self.params.context then
        vector_output=LookUpTable:forward({
            self.context_word,self.query_word,self.answer_word,self.context_length})
    else vector_output=LookUpTable:forward({
            self.context_word,self.query_word,self.answer_word})
    end
    self.context_v=vector_output[1]
    self.question_v=vector_output[2]
    self.answer_v=vector_output[3]
    self.response_v=vector_output[4]
    self.beta_v=vector_output[5]
    self.context_v=self:Reshape2Dto3D(self.context_v,self.n_instance)
    self.answer_v=self:Reshape2Dto3D(self.answer_v,self.n_instance)
    local output=self.MeMM:forward({
    self.context_v,self.question_v,self.answer_v,
    self.context_mask,self.answer_mask});
    local pred=output
    return pred,pred_response
end

function memmnet:Backward(d_pred,LookUpTable)
    local grad_inputs;
    local d_context_v,d_question_v,d_answer_v
    grad_inputs=self.MeMM:backward({
        self.context_v,self.question_v,self.answer_v,
        self.context_mask,self.answer_mask},d_pred);
    local d_context_v=grad_inputs[1];
    local d_question_v=grad_inputs[2];
    local d_answer_v=grad_inputs[3];
    d_context_v=self:Reshape3Dto2D(d_context_v)
    d_answer_v=self:Reshape3Dto2D(d_answer_v)
    if self.params.context then
        LookUpTable:backward({self.context_word,self.query_word,self.answer_word},
        {d_context_v,d_question_v,d_answer_v})
    else
        LookUpTable:backward({
            self.context_word,self.query_word,self.answer_word},
        {d_context_v,d_question_v,d_answer_v})
    end
end

function memmnet:test(file)
    self.total_instance_RBI=0
    local options_in_total=0;
    local batch_data_;
    if file=="dev" then
        batch_data_=self.Data.devData;
        self.model_flag="dev";
    elseif file=="test" then
        self.model_flag="test"
        batch_data_=self.Data.testData;
    end
    local right=0;
    local output_f
    for i=1,torch.floor(#batch_data_/self.params.batch_size) do
        self.total_instance_RBI=self.total_instance_RBI+self.params.batch_size
        local Begin=(i-1)*self.params.batch_size+1;
        local End=i*self.params.batch_size;
        if End>#batch_data_ then
            End=#batch_data_;
        end
        batch_data={}
        for j=Begin,End do
            batch_data[j-Begin+1]=batch_data_[j];
        end
        self.n_instance=#batch_data
        self:prepareData(batch_data)
        local pred,_=self:Forward(self.LookUpTable)
        local max_p,max_index=torch.max(pred,2);
        for j=1,self.params.batch_size do
            local AnswerCandidate=batch_data[j].AnswerCandidate;
            local predict_index=max_index[j][1]
            if not (#batch_data[j].answers==1 and batch_data[j].answers[1]==1)
                and batch_data[j].answers[AnswerCandidate[predict_index]]~=nil then
                right=right+1;
            end
        end
    end
    self.model_flag="train"
    return right/self.total_instance_RBI;
end

function memmnet:Initial(params_)
    self.params=params_;
    self.Data:process_data(self.params)
    self.params.token_size=200000
    self.word_table=
        nn.LookupTable(self.params.token_size,self.params.dimension):cuda();
    self.word_table:reset(self.params.init_weight)
    self.word_table.weight[1]:zero()
    --dummy token, always zero
    self.LookUpTable=self:LookUpTable_();
    self.Modules={}
    self.Modules[#self.Modules+1]=self.LookUpTable;
    self.MeMM=self:MeMM_();
    self.n_instance=self.params.batch_size;
    self.current_lr=self.params.lr;
end

function memmnet:batch_train(batch_data)
    for i=1,#self.Modules do
        self.Modules[i]:zeroGradParameters()
    end
    self.n_instance=#batch_data
    self:prepareData(batch_data);
    local pred,pred_response=self:Forward(self.LookUpTable)
    local d_pred=torch.Tensor(pred:size()):fill(0):cuda();
    for i=1,self.reward_vector:size(1) do
        local right_answer_index=batch_data[i].AnswerCandidateReverse[batch_data[i].answer[1]]
        if right_answer_index~=nil then
            if self.reward_vector[i] == 1 then
                d_pred[i][right_answer_index] = -1
            end
        end
    end
    self:Backward(d_pred,self.LookUpTable)
    self:update()
end
function memmnet:update()
    local lr=self.current_lr;
    local grad_norm=0;
    for i=1,#self.Modules do
        local p,dp=self.Modules[i]:parameters()
        for j,m in pairs(dp) do
            grad_norm=grad_norm+m:norm()^2;
        end
    end
    grad_norm=grad_norm^0.5;
    if grad_norm>self.params.thres then
        lr=lr*self.params.thres/grad_norm;
    end
    for i=1,#self.Modules do
        self.Modules[i]:updateParameters(lr);
    end
    self.word_table.weight[1]:zero()
    for i,v in pairs(self.Modules[1].modules) do
        if v.weight~=nil then
            v.weight[1]:zero();
        end
    end
end

function memmnet:train()
    local timer=torch.Timer()
    self.iter=0;
    local best_dev_acc=-10;
    local final_test_acc=0;
    self.Data.trainData=self.Data:sortData(self.Data.trainData,"hist_x")
    print(#self.Data.trainData)
    while true do
        self.model_flag="train"
        self.iter=self.iter+1;
        print("iter "..self.iter)
        if self.iter==self.params.N_iter then
            break;
        end
        if self.iter%self.params.iter_halve_lr==0 then
            if self.params.dataset=="babi" then
                self.current_lr=self.current_lr/2
                print(self.current_lr)
            end
        end
        local time1=timer:time().real;
        for k=1,math.floor(#self.Data.trainData/self.params.batch_size) do
            local batch_data={}
            local start_index=torch.random(#self.Data.trainData)
            while start_index+self.params.batch_size>=#self.Data.trainData do
                start_index=torch.random(#self.Data.trainData)
            end
            for i=start_index,start_index+self.params.batch_size-1 do
                batch_data[#batch_data+1]=self.Data.trainData[i];
            end
            self:batch_train(batch_data)
        end
        local time2=timer:time().real;
        local acc_dev=self:test("dev")
        print("acc_dev  "..acc_dev)
        if acc_dev>=best_dev_acc then
            best_dev_acc=acc_dev;
            local acc_test=self:test("test")
            final_test_acc=acc_test;
        end
        local acc_test=self:test("test")
    end
    print("test_acc "..final_test_acc)
    return final_test_acc,best_dev_acc;
end
return memmnet
