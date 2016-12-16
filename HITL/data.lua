-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tds = require('tds')
require "torchx"
local stringx = require('pl.stringx')
local Data={}

function Data:load_dict()
    local fname=self.params.dic_file;
    local f = io.open(fname)
    if f == nil then
        error("cannot load dictionary: " .. fname)
    end
    local cnt = 0
    self.dict = tds.hash()
    self.frequency={}
    while true do
        local s = f:read("*line")
        if s == nil then break; end
        local i1 = s:find('\t')
        local s1 = s:sub(1, i1 - 1)
        cnt = cnt + 1
        self.dict[cnt] = s1
        self.dict[s1] = cnt
        self.frequency[cnt]=tonumber(s:sub(i1+1,-1))
    end
    if self.params.dataset=="babi" then
        self:getAnswerCandidate()
    end
    self.dict.size=cnt;
    f:close()
end


function Data:MakePrediction(pred,randomness,instance)
    local prediction_index;
    if self.params.REINFORCE then
        -- REINFORCE: sample from multinomial
        self.prob = (self.prob == nil)
            and torch.CudaTensor(pred:size())
            or self.prob:resizeAs(pred)
        self.prob:copy(pred)
        self.prob:exp()
        prediction_index = torch.multinomial(self.prob, 1)
        prediction_index = prediction_index[1]
    else
        if math.random()<randomness then
            if not self.params.get_all_candidates then
                prediction_index=torch.random(#instance.AnswerCandidate)
            else prediction_index=torch.random(self.dict.size)
            end
        else
            _,prediction_index=torch.max(pred,1);
            prediction_index=prediction_index[1]
        end
    end
    prediction=instance.AnswerCandidate[prediction_index]
    local correct
    if prediction == nil then -- rarely it may happen RF picks a pad token
        prediction_index=torch.random(#instance.AnswerCandidate)
        prediction=instance.AnswerCandidate[prediction_index]
    end
    if instance.answers[prediction]~=nil then
        correct=true;
    else correct=false;
    end
    return prediction, correct
end

function Data:constructLink()
    print('computing links to all entities')
    -- build hash of which entity is connected to which other entity
    local links = tds.hash()
    local open_train=assert(io.open(self.params.trainData,"r"), "could not open training data at " .. self.params.trainData)
    while true do
        local line=open_train:read("*line");
        if line==nil then break; end
        line = self:process_string(line)
        local t = line:find(" ")
        line = line:sub(t+1, -1)
        local split = stringx.split(line, "\t")
        if split[1] ~= "" and line:find("knowledgebase") ~= nil then
            local v = self:String2Vector(split[1])
            for ii = 3, v:size(1) do
                if links[v[2]] == nil then
                    links[v[2]] = torch.Tensor{v[ii]}
                else
                    -- avoid adding duplicates
                    if links[v[2]]:eq(v[ii]):sum() == 0 then
                        links[v[2]] =
                            torch.cat(links[v[2]], torch.Tensor{v[ii]})
                    end
                end
                if links[v[ii]] == nil then
                    links[v[ii]] = torch.Tensor{v[2]}
                else
                    -- avoid adding duplicates
                    if links[v[ii]]:eq(v[2]):sum() == 0 then
                        links[v[ii]] =
                            torch.cat(links[v[ii]], torch.Tensor{v[2]})
                    end
                end
            end
        end
    end
    self.links = links
end

function Data:getAnswerCandidate()
    local answers={"hallway","office","bedroom","garden","bathroom","kitchen"}
    self.AnswerCandidate={}
    self.AnswerCandidateReverse={}
    for i,word in pairs(answers)do
        local index=#self.AnswerCandidate+1
        self.AnswerCandidate[i]=self.dict[word];
        self.AnswerCandidateReverse[self.dict[word]]=i;
    end
end

function Data:process_string(string)
    string=stringx.strip(string:lower())
    string=stringx.replace(string,"?","");
    string=stringx.replace(string,".","");
    string=stringx.replace(string,",","");
    string=stringx.replace(string,"!","");
    string=stringx.replace(string,"'","");
    string=stringx.replace(string,"\"","");
    return string
end

function Data:Vect2Vect(vect)
    local vector=torch.Tensor(1,self.max_sen_length):fill(1);
    if vect:size(1)<self.max_sen_length then
        vector[{{1},{1,vect:size(1)}}]:copy(vect);
    else vector:copy(vect[{{1,vect:size(1)}}])
    end
    return vector
end

function Data:String2Vector(string)
    local words=stringx.split(stringx.strip(string)," ");
    local vector=torch.Tensor(#words):fill(1);
    for i=1,#words do
        if self.dict[words[i]]==nil then
            vector[i]=1
        else
            vector[i]=self.dict[words[i]];
        end
    end
    return vector;
end

function Data:ConstructNegativePool(ex)
    if self.responsePool==nil or #self.responsePool<5000 then
        if self.responsePool==nil then
            self.responsePool={}
        end
        self.responsePool[#self.responsePool+1]=ex.response;
    elseif #self.responsePool>=5000-1 then
        self.responsePool[torch.random(#self.responsePool)]=ex.response
    end
    if self.MaxResponseLength==nil then
        self.MaxResponseLength=-100;
    end
    if ex.response:size(1)>self.MaxResponseLength then
        self.MaxResponseLength=ex.response:size(1)
    end
end

function Data:Lines2ExsMovieQA(lines)
    local dataset={};
    local current_lines={};
    for i=1,#lines do
        local line=lines[i]
        local s=self:process_string(lines[i]);
        local t=s:find(" ")
        local line_index=tonumber(s:sub(1,t-1));
        if line_index==1 then
            if #current_lines~=0 then
                local instance=self:Lines2instanceMovieQA(current_lines)
                if self.params.policyGrad and not self.params.FP then
                    if instance.r[1]==1 then
                        dataset[#dataset+1]=instance
                    end
                else
                    dataset[#dataset+1]=instance
                end
            end
            current_lines={}
        end
        current_lines[#current_lines+1]=line
    end
    return dataset
end

function Data:Lines2instanceMovieQA(lines)
    local instance={};
    local kb_x={}
    local text_x={}
    for i=1,#lines do
        local s=self:process_string(lines[i]);
        local t=s:find(" ")
        s=s:sub(t+1,-1);
        local split=stringx.split(s,"\t");
        if i==#lines-1 then
            instance.question=self:String2Vector(split[1]);
            instance.hist_x={};
            instance.kb_x={};
            for i,v in pairs(kb_x)do
                instance.kb_x[i]=v;
            end
            text_x=self:AddTimeFeature(text_x)
            for i,v in pairs(text_x)do
                instance.hist_x[i]=v;
            end
            for i,v in pairs(kb_x)do
                instance.hist_x[#instance.hist_x+1]=v;
            end
            if #instance.hist_x==0 then
                instance.hist_x[1]=torch.Tensor(1):fill(1);
            end
            if #instance.kb_x==0 then
                instance.kb_x[1]=torch.Tensor(1):fill(1);
            end
            instance=self:HandleFinalAnswer(instance,split[2],i,lines)
            instance.AnswerCandidate,instance.AnswerCandidateReverse=
                self:GetAnswerCandidates(instance)
        elseif i==#lines then
            if self.params.FP then
                if split[1]~=nil and split[1]~="" then
                    instance.response=self:String2Vector(split[1])
                    if self.mode=="train" then
                        self:ConstructNegativePool(instance,"response","responsePool")
                    end
                else
                    instance.response=torch.Tensor(1):fill(1);
                end
            end
        end
        if split[1]~="" then
            if s:find("knowledgebase")~=nil then
                kb_x[#kb_x+1]=self:String2Vector(split[1]);
            else
                text_x[#text_x+1]=self:String2Vector(split[1]);
            end
        end
        if split[2]~="" then
            local v=self:String2Vector(split[2]);
            text_x[#text_x+1]=v
        end
    end
    return instance
end

function Data:HandleFinalAnswer(instance,answer_string,current_line_index,lines)
    local split_answer=stringx.split(answer_string," ");
    local answer;
    if self.dict[split_answer[1]]~=nil then
        answer=torch.Tensor(1):fill(self.dict[split_answer[1]])
    else answer=torch.Tensor(1):fill(1)
    end
    instance.answer=answer;
    local answers=tds.hash();
    local answers_reverse=tds.hash();
    for i,v in pairs(split_answer) do
        if self.dict[v]~=nil then
            if v:find("_")~=nil then
            answers[self.dict[v]]=1;
            answers_reverse[#answers_reverse+1]=self.dict[v]
            local answer_index=self.dict[v]
            if self.PotentialAns==nil then
                self.PotentialAns={};
                self.PotentialAnsIndex={};
            end
            if self.PotentialAnsIndex[answer_index]==nil then
                self.PotentialAns[#self.PotentialAns+1]
                    =answer_index;
                self.PotentialAnsIndex[answer_index]
                    =#self.PotentialAns;
            end
            end
        end
    end
    if #answers==0 then
        answers[1]=1;
        answers_reverse[1]=1
    end
    instance.answers=answers;
    instance.answers_reverse=answers_reverse;
    if current_line_index+1<=#lines then
        local next_line=self:process_string(lines[current_line_index+1]);
        local t=next_line:find(" ")
        next_line=next_line:sub(t+1,-1);
        local split_next_line=stringx.split(next_line,"\t");
        local str=split_next_line[1];
        instance.response=self:String2Vector(str);
        instance.r=torch.Tensor({tonumber(split_next_line[3])})
    end
    return instance
end

function Data:AddTimeFeature(text_x)
    for i=1,#text_x do
        text_x[i]=torch.cat(
            text_x[i],
            torch.Tensor(1):fill(#text_x-i+1+self.dict.size),1)
    end
    return text_x
end

function Data:Lines2ExsBabi(lines)
    local dataset={};
    local hist_x;
    local hist_y;
    local current_lines={}
    for i=1,#lines do
        current_lines[#current_lines+1]=
            self:process_string(lines[i]);
        if i~=#lines then
            local next_line=lines[i+1];
            local t=next_line:find(" ")
            if next_line:sub(1,t-1)=="1" then
                local Instances=self:Lines2instanceBabi(current_lines)
                for i,v in pairs(Instances)do
                    dataset[#dataset+1]=v
                end
                current_lines={}
            end
        else
            local Instances=self:Lines2instanceBabi(current_lines)
            for i,v in pairs(Instances)do
                dataset[#dataset+1]=v
            end
        end
    end
    return dataset
end

function Data:Lines2instanceBabi(lines)
    local instances={}
    local hist_x={}
    for i,s in pairs(lines)do
        t=s:find(" ")
        local line_index=tonumber(s:sub(1,t-1));
        s=s:sub(t+1,-1);
        local split=stringx.split(s,"\t");
        if split[2]~="" then
            local instance={};
            instance.question=self:String2Vector(split[1])
            instance.hist_x={};
            instance.text_x={};
            for j,v in pairs(hist_x)do
                instance.text_x[j]=v;
            end
            instance.hist_x=self:AddTimeFeature(instance.text_x)
            for j,v in pairs(instance.text_x)do
                instance.hist_x[j]=v;
            end
            instance.answer=torch.Tensor(1):fill(self.dict[split[2]]);
            instance.answers={}
            instance.answers[self.dict[split[2]]]=1
            instance.AnswerCandidate=self.AnswerCandidate
            instance.AnswerCandidateReverse=self.AnswerCandidateReverse
            if self.params.task==1 then
                --for task 1, reward is immediately given, and we wom't wait for the teacher at next turn
                instance.r=torch.Tensor(1):fill(tonumber(split[3])):cuda()
            end
            if i~=#lines then
                local next_line=lines[i+1];
                local t=next_line:find(" ")
                next_line=next_line:sub(t+1,-1)
                local split_next=stringx.split(next_line,"\t")
                instance.response=self:String2Vector(split_next[1])
                if self.params.task~=1 then
                    instance.r=torch.Tensor(1):fill(tonumber(split_next[3])):cuda()
                end
            else instance.response=torch.Tensor(1):fill(1);
            end
            self:ConstructNegativePool(instance)
            instances[#instances+1]=instance
        else
            hist_x[#hist_x+1]=self:String2Vector(split[1])
        end
    end
    return instances
end

function Data:load_data(filename)
    local f=io.open(filename);
    if f == nil then
        error("cannot load file: " .. filename)
    end
    local lines={};
    while true do
        local s=f:read("*line")       ;
        if s == nil then break; end
        lines[#lines+1]=self:process_string(s);
    end
    f:close();
    local dataset
    if self.params.dataset=="babi" then
        dataset=self:Lines2ExsBabi(lines)
    elseif self.params.dataset=="movieQA" then
        dataset=self:Lines2ExsMovieQA(lines)
    end
    return dataset
end

function Data:GetAnswerCandidates(ex)
    local AnswerCandidate={}
    local AnswerCandidateReverse={}
    local memory=ex.hist_x
    for j=1,#memory do
        for k=1,memory[j]:size(1)do
            local candidate=memory[j][k];
            if candidate~=1 and
                candidate<=self.dict.size and
                self.frequency[candidate]<10000 and
            AnswerCandidateReverse[candidate]==nil then
                AnswerCandidate[#AnswerCandidate+1] = candidate
                AnswerCandidateReverse[candidate]= #AnswerCandidate
            end
        end
        if #AnswerCandidate==0 then
            AnswerCandidate[#AnswerCandidate+1]=1
            AnswerCandidateReverse[1]=1;
        end
    end
    return AnswerCandidate, AnswerCandidateReverse
end

function Data:sortData(Dataset,fieldname)
    local max_length=0;
    local Data={};
    for i=1,#Dataset do
        local length=#Dataset[i][fieldname];
        if length>max_length then
            max_length=length;
        end
        if Data[length]==nil then
            Data[length]={};
        end
        Data[length][#Data[length]+1]=Dataset[i];
    end
    Dataset={};
    for i=0,max_length do
        if Data[i]~=nil then
            for j=1,#Data[i] do
                local ex=Data[i][j];
                Dataset[#Dataset+1]=ex;
            end
        end
    end
    return Dataset;
end

function Data:Initial(params)
    self.params=params;
    self:load_dict()
end

function Data:process_data(params)
    self.mode="train"
    self.trainData=self:load_data(self.params.trainData,"train")
    self.mode="test"
    self.testData=self:load_data(self.params.testData,"test")
    self.mode="dev"
    self.devData=self:load_data(self.params.devData,"dev")
end

return Data
