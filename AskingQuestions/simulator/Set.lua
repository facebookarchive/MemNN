--[[
    --
    -- A simple data structure storing a set of items which can be indexed
    -- both using the forward index and reverse index
    --
--]]

local tds = require('tds')
local Set = torch.class('Set')


-- the constructor takes as input a table of items
function Set:__init(tbl)
    self.index2item = tds.hash()
    self.item2index = tds.hash()
    self.nitems = 0
    for i, v in pairs(tbl) do
        self.nitems = self.nitems + 1
        self.index2item[self.nitems] = v
        self.item2index[v] = self.nitems
    end
    self.sampled_flags = torch.IntTensor(self.nitems):fill(0)
end

function Set:find(item)
    return self.item2index[item]
end

function Set:get(index)
    return self.index2item[index]
end

function Set:sample_with_replacement()
    local index = math.random(self.nitems)
    return self.index2item[index]
end

function Set:sample_without_replacement()
    assert(self.sampled_flags:sum() < self.nitems, 'exhausted the full set. nothing to sample')
    local index = math.random(self.nitems)
    while self.sampled_flags[index] == 1 do
        index = math.random(self.nitems)
    end
    self.sampled_flags[index] = 1
    return self.index2item[index]
end


function Set:size()
    return self.nitems
end
return Set
