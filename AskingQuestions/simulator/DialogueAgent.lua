-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--[[
--
-- Class that implements generic interface of an agent taking part
-- in a dialogue. Sub-classes have to implement the basic methods
-- and can potentially add class members like a model implementing
-- the policy used to interact.
--
--]]
local DialogueAgent =  torch.class('DialogueAgent')

function DialogueAgent:__init(opts, model)
    self.name = opts.name or 'Agent ' .. math.ceil(math.random() * 10000)
    self.agent_total_reward = 0
    self.quitting = false -- quit dialogue
end

-- takes current log of the conversation and outputs a string.
function DialogueAgent:speak(log)
    error('DialogueAgent:speak you have to implement this')
    return ""
end

-- update agent internal state
function DialogueAgent:update(log)
end

-- (optional) assigns a reward to other agents,
-- usually done only by the teacher.
function DialogueAgent:giveReward(log, rewards)
end

-- returns the current state of the agent
function DialogueAgent:getState()
end

function DialogueAgent:getReward(log)
    self.agent_total_reward = 0
    for idx = 1, #log do
        self.agent_total_reward = self.agent_total_reward +
            self:getRewardAt(log, idx)
    end
    return self.agent_total_reward
end

-- computes the reward at time "idx" of the agent,
-- and adds that to the agent total reward.
function DialogueAgent:getRewardAt(log, idx)
    idx = (idx == nil) and #log or idx
    local cr = log[idx].rewards[self.name] -- {(from, val)...} or nil
    local reward = 0
    if cr ~= nil then
        -- accumulate rewards from all agents
        for _, rew in pairs(cr) do
            reward = reward + rew[2]
        end
    end
    return reward
end

function DialogueAgent:clear()
    self.agent_total_reward = 0
end

function DialogueAgent:quit()
    return self.quitting
end
