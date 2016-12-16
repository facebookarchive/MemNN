-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local pl = require('pl.import_into')()
require('../simulator/Set')
local utils=require("../simulator/movie_utils")
local stringx = require('pl.stringx')

local movieStudent={}

function movieStudent:Initial(opts)
    for i, v in pairs(opts) do
        self[i] = v
    end
    self.name = opts.name or 'movieStudent'
    self.curr_example=0
    self:reload_dataset()
end

function movieStudent:reload_dataset()
    self.no_list = utils:gen_no_list()
    self.yes_list = utils:gen_yes_list()
    self.entities = utils:load_movie_answers(self.all_entities)
    self.relations, self.relation_hints, self.hints =utils:load_movie_hints(
        self.kb_file)
    self.kb, self.sup_facts =utils:load_movie_sfs(
        self.kb_file, self.relations, self.relation_hints, self.hints)
    self.dataset = utils:load_dataset_student(self.dataset_file)
    self.dataset_teacher = utils:load_dataset_teacher(self.dataset_file,self.CopyAllAnswers)
    self.question_entity_dataset,self.question_relation_dataset,
    self.question_templates,self.question_categories=utils:load_questionEntity(self.dataset_file,
        self.question_entity_file,self.question_relation_file
        ,self.template_file,self.CopyAllAnswers)
    self.synonyms=utils:load_synonyms(self.question_template_file)
    self.typos,self.typo_reverse,self.typos_dev=
        utils:loadTypo(self.typo_file)
end

function movieStudent:generate_answerUtterance(log)
    local utterance
    local last=#log
    if not self.CopyAllAnswers then
        assert(log[last].agentName ~= self.name)
        local candidates = self.dataset[self.question_asked]
        local pred
        if candidates == nil then
            utterance = 'Sorry. The answer is not in my dataset.'
            self.state = 'quitting'
            self.quitting='true'
        else
            if math.random()<self.prob_correct_final_answer then
                utterance=self.curr_answer;
            else
                utterance=self.entities[math.random(#self.entities)]
            end
        end
    else utterance=self.answer_string;
    end
    return utterance
end

function movieStudent:clear()
    self.quitting=false
    self.curr_example=self.curr_example+1;
    if self.curr_example==#self.dataset_teacher+1 then
        self.curr_example=1;
    end
    self.question_asked=
        self.dataset_teacher[self.curr_example][1];
    self.question_entity=self.question_entity_dataset[self.curr_example];
    self.curr_answer = self.dataset_teacher[self.curr_example][2]
    self.curr_answers = self.dataset_teacher[self.curr_example][3]
    self.question_relation=
        self.question_relation_dataset[self.curr_example];
    self.question_entity=self.question_entity_dataset[self.curr_example];
    self.question_template=self.question_templates[self.curr_example];
    self.question_category=self.question_categories[self.curr_example];
    self.answer_string=self.dataset_teacher[self.curr_example][4];
    local sup_facts=self.sup_facts[self.question_entity]
    self.answer_lists={};
    for i=1,sup_facts:size(1)do
        local utter=self.kb[sup_facts[i]]
        local split=stringx.split(stringx.strip(utter)," ")
        for i,v in pairs(split)do
            self.answer_lists[#self.answer_lists+1]=v;
        end
    end
    if self.setting=="AQ" or (self.setting=="mix" and self.curr_example%2==1) then
        self.state="AskQuestion";
    elseif self.setting=="QA" or(self.setting=="mix" and self.curr_example%2==0) then
        if self.junk then
            self.state="answer_random_question"
            self.random_question_num=1;
        else self.state="answering";
        end
    end
    if self.task==3 or self.task==4 then
        --Task 3: Ask For Relevant Knowledge
        --Task 4: Knowledge Verification
        self.true_question,self.false_question=self:generateStudentQuestion(self.question_entity,
            self.question_relation,self.question_asked);
    end
    if self.task==1 or self.task==2 then
        self:get_paraphrase()
    end
end


function movieStudent:update(log)
    local last=#log
    if self.state=="AskQuestion" and log[last].agentName == self.name then
        self.state="wait"
    elseif self.state=="wait" and log[last].agentName == self.name then
        if self.junk then
            self.random_question_num=1
            self.state="answer_random_question"
        else self.state="answering";
        end
    elseif self.state=="answer_random_question"and log[last].agentName == self.name then
        self.random_question_num=self.random_question_num+1;
        if self.random_question_num>self.randomQuestionNumTotal then
            self.state="answering"
        else self.state="answer_random_question"
        end
    end
end

function movieStudent:AskQuestionUtterance()
    local utterance
    if self.task==1 then
        --Task 1: Question Paraphrase
        --the student needs to ask the teacher to paraphrase the question in a form without the typo
        utterance="what do you mean?"
    elseif self.task==2 then
        --Task 2: Question Verification
        --the student needs to ask whether the question with typos means a question without typos
        if math.random()<self.prob_correct_intermediate_answer then
            utterance="do you mean "..self.question_asked .."?"
        else
            utterance="do you mean "..self.question_random .."?"
        end
    elseif self.task==3 then
        --Ask For Relevant Knowledge
        --the student needs to ask the relevant supporting fact to answer the question
        utterance="Can you give me a hint ?"
    elseif self.task==4 then
        --Task 4: Knowledge Verification
        --the student needs to ask whether a  fact is relevant to the question
        local fullutterance
        local random=math.random()
        if (self.false_question.nitems==0 or
            self.false_question.sampled_flags:sum()>=
                self.false_question.nitems or
            random<self.prob_correct_intermediate_answer)
            and #self.true_question>0 then
            local right_question_index=torch.random(#self.true_question);
            fullutterance=self.true_question[right_question_index];
            self.task_11_correct=true;
        else
            fullutterance=self.false_question:sample_without_replacement();
        end
        local t1=fullutterance:find(" ");
        local movie_name=fullutterance:sub(0,t1-1);
        local t2=fullutterance:find(" ",t1+1);
        local relation=fullutterance:sub(t1+1,t2-1);
        local entity_string=fullutterance:sub(t2+1,-1)
        local entity_name;
        if self.task_11_correct then
            if entity_string:find(self.curr_answer)==nil then
                entity_name=self.question_entity
            else entity_name=self.curr_answer
            end
        else
            if entity_string:find(self.curr_answer)==nil then
                entity_name=self.question_entity
            else entities=stringx.split(entity_string,", ");
                entity_name=entities[torch.random(#entities)]
            end
        end
        utterance="does it have something to do with \""..
            movie_name.." "..relation.." "..entity_name.."\"?"
    else --task 5 6 7 8 9
         --Knowledge Acquisition
         utterance="i dont know. what's the answer ?"
    end
    return utterance
end

function movieStudent:speak(log)
    local utterance
    local reward=0
    local last = #log
    if self.state =="AskQuestion" then
        utterance=self:AskQuestionUtterance()
    elseif self.state == 'answering' then
        utterance=self:generate_answerUtterance(log)
    elseif self.state==specialState then
        utterance=self:specialStateUtterance(log)
    elseif self.state=="wait" then
        utterance=""
    elseif self.state=="answer_random_question" then
        utterance=self:answer_random_questionUtterance()
    end
    --print("student "..self.state.." "..utterance)
    return utterance,reward
end


function movieStudent:get_paraphrase()
    self.question_paraphrase="";
    local utterance=stringx.replace(self.question_asked,"?","");
    local splits=stringx.split(utterance," ");
    for i,v in pairs(splits)do
        if self.typos[v]==nil then
            self.question_paraphrase=self.question_paraphrase..v.." ";
        elseif self.mode=="dev" or self.mode=="test" then
            self.question_paraphrase=
                self.question_paraphrase..self.typos_dev[v].." ";

        elseif self.mode=="train" then
            self.question_paraphrase=
                self.question_paraphrase..self.typos[v].." ";
        end
    end
    self.question_paraphrase=
        stringx.strip(self.question_paraphrase).."?"
    local candidates_index=torch.random(#self.synonyms.category2strs);
    local candidates_category=self.synonyms.categoryindex2str[candidates_index]
    while self.question_category==candidates_category do
        candidates_index=torch.random(#self.synonyms.category2strs);
        candidates_category=self.synonyms.categoryindex2str[candidates_index]
    end
    local candidates=self.synonyms.category2strs[candidates_category];
    local rand_index=torch.random(#candidates)
    local template=candidates[rand_index];
    local t1=template:find("X")
    self.question_random=template:sub(1,t1-1)..self.question_entity..template:sub(t1+1,-1);
end


function movieStudent:generateStudentQuestion(question_entity,
        question_relation,question_asked)
    --Task 4: Knowledge Verification
    --Generate questions that the student can ask
    --it can be either correct questions or incorrect ones
    local question_candidates;
    question_candidates=self.sup_facts[question_entity];
    local true_question={};
    local false_question={};
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
        elseif question_relation~=self.question_relation then
            false_question[#false_question+1]=question_utterance;
        end
    end
    if #true_question==0 and #false_question==0 then
        print(question_entity)
        print(self.question_asked)
        print(question_entity)
        print(question_candidates:size())
        for i=1,question_candidates:size(1) do
            print(self.kb[question_candidates[i]],self.curr_answer)
        end
        print("true_question")
        for i,v in pairs(true_question)do
            print(i,v)
        end
        print("false_question")
        for i,v in pairs(false_question)do
            print(i,v)
        end
        error("qeustion list zero")
    end
    false_question=Set(false_question)
    return true_question,false_question
end

function movieStudent:quit()
    return self.quitting;
end

function movieStudent:answer_random_questionUtterance()
    return self.dataset_teacher[torch.random(#self.dataset_teacher)][2]
end
return movieStudent
