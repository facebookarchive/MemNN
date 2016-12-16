-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local stringx = require('pl.stringx')
local movieDialogueSimulator={}

function movieDialogueSimulator:Initial(opt,agents)
    self.log = {}
    self.agents = {}
    for id, ag in pairs(agents) do
        self.agents[id] = ag
    end
    for i,v in pairs(opt)do
        self[i]=v;
    end
    self.output_file = io.open(self.output_fname, 'w')
end

function movieDialogueSimulator:nextUtterance(agent)
    local comment = {}
    local utterance = agent:speak(self.log)
    comment.agentName = agent.name
    comment.utterance = utterance
    table.insert(self.log, comment)
    if self.verbosity > 1 then
        print(agent.name .. ": " .. utterance)
    end
end

-- After an utterance has been released, update agents internal
-- states.
function movieDialogueSimulator:update()
    -- first get the rewards from every agent
    local rewards = {}
    -- agents exchange rewards
    for _, curr_agent in pairs(self.agents) do
        -- each agent appends rewards
        -- rewards are in the format rewards[to] = {(from, val)...}
        curr_agent:giveReward(self.log, rewards)
    end
    self.log[#self.log].rewards = rewards
    for _, current_agent in pairs(self.agents) do
        current_agent:update(self.log)
    end
end

-- Clear all message history and agents' internal state.
function movieDialogueSimulator:clearState()
    self.log = {}
    for _, current_agent in pairs(self.agents) do
        current_agent:clear()
    end
end

-- Simple communication protocol, agents talk
-- one after the other till one of them decides
-- to quit the conversation.
function movieDialogueSimulator:roundtable()
    self:clearState()
    local num_turns = 1
    local agent_id = 1
    local keep_going = true
    while keep_going do
        self:nextUtterance(self.agents[agent_id])
        self:update()
        keep_going = not self.agents[agent_id]:quit()
        num_turns = num_turns + 1
        agent_id = (num_turns - 1) % #self.agents + 1
    end
end



function movieDialogueSimulator:nextUtterance(agent)
    local comment = {}
    local utterance,reward = agent:speak(self.log)
    comment.agentName = agent.name
    comment.utterance = utterance
    comment.rewards=reward;
    table.insert(self.log, comment)
end


function movieDialogueSimulator:update()
    for _, current_agent in pairs(self.agents) do
        current_agent:update(self.log)
    end
end


function movieDialogueSimulator:print_log()
    assert(self.output_file)
    local nlines = #self.log
    local lctr = 0
    local str=""
    local utterance
    local reward
    local teacher=self.agents[1];
    local student=self.agents[2];
    local sup_facts=teacher.sup_facts[teacher.question_entity]
    for i=1,sup_facts:size(1)do
        local formal_utter=teacher.kb[sup_facts[i]]
        local should_write=true
        --whether we should keep or remove current kb fact
        --for task 1 2 3 4, all kb facts should be kept
        --for task 5 6 7 8 9, we deliberately hide facts to make the kb incomplete so that the
        --student can ask questions about the missing entities
        --whethr current fact should be removed is task-dependent
        if self.task==9 then
            --task 9, everything is missing, all facts need to be removed
            should_write=false
        elseif self.task==5 then
            --task5 question entity is missing
            if formal_utter:find(teacher.question_entity)~=nil then
                --remove that facts that contain the question entity
                should_write=false;
            end
        elseif self.task==6 then
            --task6 answer entity is missing
            --remove that facts that contain the answer entity
            if student.answer_string~=nil then
                local answers=stringx.split(student.answer_string,", ")
                for _,v in pairs(answers) do
                    if formal_utter:find(v)~=nil then
                        should_write=false;
                        break
                    end
                    --remove those containing answer entity
                end
            end
        elseif self.task==7 then
            --task7 relation entity is missing
            --remove the facts that contain the same relation entity
            if formal_utter:find(teacher.question_relation)~=nil then
                should_write=false;
            end
        elseif self.task==8 then
            --task8 the tripple is missing
             if formal_utter:find(teacher.question_entity)~=nil and
             formal_utter:find(teacher.curr_answer)~=nil and
             formal_utter:find(teacher.question_relation)~=nil then
                should_write=false;
            end
        end
        if should_write then
            lctr=lctr+1;
            self.output_file:write(lctr.." knowledgebase: "..formal_utter.."\t\t0\n")
        end
    end
    for i = 1, nlines do
        utterance = self.log[i]['utterance']
        if self.log[i].agentName=="movieTeacher" then
            lctr=lctr+1;
            str=lctr.." "..utterance;
            reward = self.log[i]["rewards"]
        else
            str=str.."\t"..utterance.."\t"..reward.."\n";
            self.output_file:write(str)
            str=""
        end
    end
    if str~=nil then
        str=str.."\t\t"..reward.."\n";
        self.output_file:write(str)
    end
end

function movieDialogueSimulator:close_files()
    self.output_file:close()
end

return movieDialogueSimulator
