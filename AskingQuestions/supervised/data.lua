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
    self.dict.size=cnt;
    f:close()
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

function Data:Dic2Vector(dic)
    local vector=torch.Tensor(#dic):fill(1);
    for i=1,#(dic) do
        vector[i]=dic[i]
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
                if instance.r[1]==1 then
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

function Data:GetAnswerContext(ex)
    local memory=ex.hist_x
    local answer_context={};
    for i,v in pairs(ex.AnswerCandidate) do
        answer_context[i]={}
    end
    for j=1,#memory do
        for k=1,memory[j]:size(1)do
            local word=memory[j][k];
            if ex.AnswerCandidateReverse[word]~=nil then
                local word_index=ex.AnswerCandidateReverse[word];
                if #answer_context[word_index]==0 then
                    answer_context[word_index][#answer_context[word_index]+1]=memory[j][k]
                end
                local left_c_num=0;
                local k_=k-1;
                local j_=j;
                while true do
                    if k_==0 then
                        j_=j_-1;
                        if j_==0 then break end
                        k_=memory[j_]:size(1);
                    end
                    local w=memory[j_][k_];
                    if w~=self.dict["Student"] and w~=self.dict["Teacher"]
                        and self.dict[w]~=nil and #answer_context[word_index]<150  then
                        if left_c_num>=self.params.context_num then
                            break
                        else
                            answer_context[word_index][#answer_context[word_index]+1]=memory[j_][k_]
                            left_c_num=left_c_num+1
                        end
                    end
                    k_=k_-1
                end
                k_=k+1;
                j_=j;
                local right_c_num=0
                while true do
                    if k_==memory[j_]:size(1)+1 then
                        k_=1;
                        j_=j_+1;
                        if j_>#memory then break end
                    end
                    local w=memory[j_][k_];
                    if w~=self.dict["Student"] and w~=self.dict["Teacher"]
                        and self.dict[w]~=nil and #answer_context[word_index]<150 then
                        if right_c_num>=self.params.context_num then
                            break
                        else
                            answer_context[word_index][#answer_context[word_index]+1]=memory[j_][k_]
                            right_c_num=right_c_num+1;
                        end
                    end
                    k_=k_+1
                end
            end
        end
    end
    local answer_context_={}
    for i,v in pairs(answer_context)do
        if #v==0 then
            v[1]=1;
        end
        answer_context_[i]=self:Dic2Vector(v)
    end
    return answer_context_
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
    local dataset=self:Lines2ExsMovieQA(lines)
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

function Data:process_data(params)
    self.params=params;
    self:load_dict()
    self.mode="train"
    self.trainData=self:load_data(self.params.trainData,"train")
    self.mode="test"
    self.testData=self:load_data(self.params.testData,"test")
    self.mode="dev"
    self.devData=self:load_data(self.params.devData,"dev")
end

return Data
