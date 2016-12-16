-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local stringx = require('pl.stringx')
local utils=require("../simulator/movie_utils")
local movieTeacher={}

function movieTeacher:Initial(opts)
    for i, v in pairs(opts) do
        self[i] = v
    end
    self.name='movieTeacher'
    self.no_list = utils:gen_no_list()
    self.yes_list = utils:gen_yes_list()
    self.answers = utils:load_movie_answers(self.all_entities)
    self.current_reward = 0
    self.quitting = false
    self.curr_supfacts = {}

    self.curr_example=0
    self:reload_dataset()
end

function movieTeacher:reload_dataset()
    self.typos,self.typos_reverse,self.typos_dev
        =utils:loadTypo(self.typo_file)
    self.relations, self.relation_hints, self.hints =
        utils:load_movie_hints(self.kb_file)
    self.kb, self.sup_facts =utils:load_movie_sfs(
    self.kb_file, self.relations, self.relation_hints, self.hints)
    self.question_entity_dataset,self.question_relation_dataset,
    self.question_templates,self.question_categories=
        utils:load_questionEntity(
            self.dataset_file,self.question_entity_file,
            self.question_relation_file,self.template_file,self.CopyAllAnswers)
    self.dataset = utils:load_dataset_teacher(self.dataset_file,self.CopyAllAnswers)
    self.synonyms=utils:load_synonyms(self.question_template_file)
    self.curr_example = 0
end

function movieTeacher:build_answer(prediction, ground_truth,curr_answer)
    local utt=""
    local reward=0
    if not self.CopyAllAnswers then
        --the teacher needs to decide response
        if ground_truth:find(prediction) ~= nil then
            utt = self.yes_list[math.random(#self.yes_list)]
            reward=1
        else
            utt = self.no_list[math.random(#self.no_list)]
            reward=0
        end
    else
        utt = self.yes_list[math.random(#self.yes_list)]
        reward=1
    end
    return utt,reward
end

function movieTeacher:get_paraphrase(utter)
    --generate questions with typos, for task 1 and 2
    local out=""
    local utterance=stringx.replace(utter,"?","");
    local splits=stringx.split(utterance," ");
    for i,v in pairs(splits)do
        if self.typos[v]==nil then
            out=out..v.." ";
        elseif self.mode=="dev" or self.mode=="test" then
            out=out..self.typos_dev[v].." ";
        elseif self.mode=="train" then
            out=out..self.typos[v].." ";
        end
    end
    out=stringx.strip(out).."?"
    return out
end

function movieTeacher:random_questionUtterance()
    return self.dataset[torch.random(#self.dataset)][1]
end

function movieTeacher:questioningUtterance()
    local utterance
    if self.task==1 or self.task==2 then
        --used the question that has typo
        utterance=self.question_paraphrase
    elseif self.task==5 then
        --task5 question entity missing.
        --To make sure that the question entity has not been encountered before, we introduce typos
        if self.mode=="dev" or self.mode=="test" then
            self.question_replace=stringx.replace(self.question_asked,self.question_entity,self.question_entity.."_1");
            utterance=self.question_replace
        else
            utterance=self.question_asked
        end
    elseif self.task==7 or self.task==9 then
        --task7 missing relation entity
        --task9 missing everything
        if self.mode=="dev" or self.mode=="test" then
            local utt=stringx.replace(self.question_asked,"?","")
            local splits=stringx.split(utt," ")
            local utt_return=""
            for i,v in pairs(splits)do
                if v~=self.question_entity then
                    utt_return=utt_return..v..v:sub(-1,-1).." "
                    --mess up the relation
                elseif self.task==7 then
                    utt_return=utt_return..v.." "
                    --if task 7, we use the correct question entity
                elseif self.task==9 then
                    utt_return=utt_return..v.."_1".." "
                    --if task 7 (missing everything), we also mess up the question entity
                end
            end
            utterance=utt_return;
        else
            utterance=self.question_asked
        end
    else utterance=self.question_asked
    end
    return utterance
end

function movieTeacher:speak(log)
    local last=#log;
    local utterance
    local reward=0
    self.quitting = false
    if self.state == 'questioning' then
        utterance = self:questioningUtterance();
    elseif self.state=="AnswerStudentQuestion" then
        utterance,reward=self:AnswerStudentQuestionUtterance(log)
    elseif self.state=="random_question" then
        utterance=self:random_questionUtterance()
    elseif self.state=="ReAskQuestion" then
        utterance = self:questioningUtterance();
    elseif self.state == 'feedbacking' then
        utterance,reward=self:feedbackingUtterance(log)
    end
    --print("teacher "..self.state.." "..utterance.." "..reward)
    return utterance,reward;
end

function movieTeacher:feedbackingUtterance(log)
    local last = #log
    assert(log[last].agentName ~= self.name)
    utterance,reward=self:build_answer(log[last].utterance, self.curr_answers,self.curr_answer)
    self.quitting=true;
    return utterance,reward
end

function movieTeacher:update(log)
    local last=#log;
    if self.state == 'questioning' and log[last].agentName == self.name then
        self.state="AnswerStudentQuestion"
    elseif self.state=="AnswerStudentQuestion" and log[last].agentName == self.name then
        if self.junk then
            self.state="random_question"
            self.random_question_num=1;
        else self.state="ReAskQuestion"
        end
    elseif self.state=="random_question" and log[last].agentName == self.name then
        self.random_question_num=self.random_question_num+1;
        if self.random_question_num>self.randomQuestionNumTotal then
            self.state="ReAskQuestion";
        else self.state="random_question";
        end
    elseif self.state=="ReAskQuestion" and log[last].agentName == self.name then
        self.state="feedbacking";
    end
end

function movieTeacher:AnswerStudentQuestionUtterance(log)
    local utterance,reward
    local last_utter=log[#log].utterance;
    if self.task==1 then
        --Task 1: Question Paraphrase
        --the student asks the teacher to paraphrase the question in a form without the typo
        --and the teacher needs to paraphrase the question
        utterance="I mean "..self.question_asked;
        if utterance:sub(-1,-1)=="?" then
            utterance=utterance:sub(1,-2).."."
        end
        reward=1;
    elseif self.task==2 then
        --Task 2: Question Verification
        --the student asks whether the question with typos means a question without typos
        --and the teacher needs to give "yes or no" feedback
        if last_utter:find(self.question_asked)~=nil then
            utterance = self.yes_list[math.random(#self.yes_list)]
            reward=1;
        else
            utterance = self.no_list[math.random(#self.yes_list)]
            reward=0;
        end
    elseif self.task==3 then
        --Ask For Relevant Knowledge
        --the student asks the relevant supporting fact to answer the question
        --and the teacher needs to point out the relevant supporting fact
        utterance="it is related to "..self:PullOutRelevantSupportingFact()
        reward=1
    elseif self.task==4 then
        --Task 4: Knowledge Verification
        --the student asks whether a  fact is relevant to the question
        --and the teacher needs to give "yes or no" feedback
        local studentQuestion=log[#log].utterance
        if studentQuestion:find(self.question_relation)~=nil and
            studentQuestion:find(self.curr_answer)~=nil then
            utterance = self.yes_list[math.random(#self.yes_list)]
            reward=1;
        else
            utterance= self.no_list[math.random(#self.no_list)]
            reward=0;
        end
    else --task 5 6 7 8 9
        --Knowledge Acquisition
        --the teacher needs to give the correct answer
        if self.task==6 or self.task==9 then
            --task 6, unknown answer_entity ; task 9 everything unknown
            --we add typos to the entity to make it unknown
            if self.mode=="dev" or self.mode=="test" then
                utterance=self.curr_answer.."_1"
            else
                utterance=self.curr_answer;
            end
        else    utterance=self.curr_answer
        end
        utterance="The answer is "..utterance
        reward=1;
    end
    return utterance,reward
end


function movieTeacher:clear()
    self.curr_example = self.curr_example + 1
    if self.curr_example==#self.dataset+1 then
        self.curr_example=1;
    end
    self.curr_answer = self.dataset[self.curr_example][2]
    self.curr_answers = self.dataset[self.curr_example][3]
    local utterance = self.dataset[self.curr_example][1]
    self.curr_sup_facts = utils:get_supporting_facts(utterance,
                                                    self.curr_answer,
                                                    self.sup_facts,
                                                    self.kb,
                                                    self.relations)
    self.question_asked=utterance;
    self.question_entity=self.question_entity_dataset[self.curr_example];
    self.question_relation=self.question_relation_dataset[self.curr_example];
    self.question_template=self.question_templates[self.curr_example];
    self.question_category=self.question_categories[self.curr_example];
    self.question_paraphrase=self:get_paraphrase(self.question_asked)
    if self.setting=="AQ" or (self.setting=="mix" and self.curr_example%2==1) then
        self.state="questioning";
    elseif self.setting=="QA" or(self.setting=="mix" and self.curr_example%2==0) then
        if self.junk then
            self.state="random_question"
            self.random_question_num=1;
        else self.state="ReAskQuestion"
        end
    end
    if self.task==3 or self.task==4 then
        --Task 3: Ask For Relevant Knowledge
        --Task 4: Knowledge Verification
        self.true_question=self:generateStudentQuestion(self.question_entity,
            self.question_relation,self.question_asked);
    end
    self.quitting=false;
end

function movieTeacher:PullOutRelevantSupportingFact()
    if #self.true_question==0 then
        return ""
    end
    local right_question_index=torch.random(#self.true_question);
    fullutterance=self.true_question[right_question_index];
    local t1=fullutterance:find(" ");
    local movie_name=fullutterance:sub(0,t1-1);
    local t2=fullutterance:find(" ",t1+1);
    local relation=fullutterance:sub(t1+1,t2-1);
    local entity_string=fullutterance:sub(t2+1,-1)
    local entity_name
    if entity_string:find(self.curr_answer)==nil then
        entity_name=self.question_entity
    else entity_name=self.curr_answer
    end
    --return fullutterance
    return movie_name.." "..relation.." "..entity_name
end

function movieTeacher:generateStudentQuestion(question_entity,
    question_relation,question_asked)
    local question_candidates
    question_candidates=self.sup_facts[question_entity];
    local true_question={};
    for i=1,question_candidates:size(1)do
        local question_index=question_candidates[i];
        local question_utterance=self.kb[question_index];
        local question_relation;
        for i,v in pairs(self.relations) do
            if stringx.lfind(question_utterance,v)~=nil then
                question_relation=v;
                break;
            end
        end
        if question_relation==self.question_relation
            and stringx.lfind(question_utterance,self.curr_answer)~=nil then
            true_question[#true_question+1]=question_utterance;
        end
    end
    return true_question
end


function movieTeacher:quit()
    return self.quitting;
end

return movieTeacher
