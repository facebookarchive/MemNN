-- Copyright 2004-present Facebook. All Rights Reserved.
-- Base dictionary class for MemNNs.

require('torch')
local pl = require('pl.import_into')()
local tds = require('tds')

local dict = {}
dict.__index = dict

function dict:create(opt, shared)
    tds = require('tds')
    local new_dict
    if shared == nil then
        new_dict = dict:init_dict(opt)
        setmetatable(new_dict, { __index = self })
        print("[num symbols: " .. new_dict.num_symbols .. "]")
    else
        if shared.dictFullLoading == false then
            new_dict = shared
            setmetatable(new_dict, { __index = self })
        else
            new_dict = dict:init_dict(opt)
            setmetatable(new_dict, { __index = self })
        end
    end
    return new_dict
end

function dict:init_dict(opt)
    local new_dict
    new_dict = {}
    new_dict.opt = opt
    new_dict.num_symbols = 0
    new_dict._index_to_symbol = tds.hash()
    new_dict._symbol_to_index = tds.hash()
    new_dict._unkindex_to_symbol = tds.hash()

    print("[loading dict:" .. opt.dictFile .. "]")
    local f = io.open(opt.dictFile)
    if f == nil then
        error(opt.dictFile .. ' not found')
    end
    local s = f:read("*all")
    f:close()
    local i1 = 1
    local i2
    -- Count the dictionary size first:
    while true do
        i2 = s:find('\n', i1)
        if i2 == nil then
            break
        end
        new_dict.num_symbols = new_dict.num_symbols + 1
        i1 = i2 + 1
    end
    new_dict.index_to_freq = torch.DoubleTensor(new_dict.num_symbols)
    new_dict.num_inputs = new_dict.num_symbols
    new_dict.num_labels = new_dict.num_symbols
    i1 = 1
    local cnt = 0
    while true do
        i2 = s:find('\n', i1)
        local line
        if i2 == nil then
            break
        else
            line = s:sub(i1, i2 - 1)
        end
        local t = pl.utils.split(line, '\t')
        cnt = cnt + 1
        new_dict._symbol_to_index[t[1]] = cnt
        new_dict._index_to_symbol[cnt] = t[1]
        new_dict.index_to_freq[cnt] = t[2] * 1.0
        i1 = i2 + 1
    end
    if opt.dictFullLoading and opt.dictUNKFile ~= nil then
        local f = io.open(opt.dictUNKFile)
        if f == nil then
            error(opt.dictUNKFile .. ' not found')
        end
        local s = f:read("*all")
        f:close()
        local i1 = 1
        local i2
        local cnt = 0
        while true do
            i2 = s:find('\n', i1)
            local line
            if i2 == nil then
                break
            else
                line = s:sub(i1, i2 - 1)
            end
            local t = pl.utils.split(line, '\t')
            cnt = cnt + 1
            new_dict._unkindex_to_symbol[tonumber(t[1])] = t[2]
            i1 = i2 + 1
        end
        print('[UNKhashed words loaded:' .. cnt  .. ']')
    end
    if opt.dictFullLoading ~= true then
        print "[clearing unnecessary dict fields to save memory.]"
        new_dict.dictFullLoading = false
        new_dict._index_to_symbol = nil
        new_dict._symbol_to_index = nil
        new_dict._unkindex_to_symbol = nil
        if opt.dictStoreIndexToFreq ~= true and opt.dictMinOcc == nil then
            new_dict.index_to_freq = nil
        end
        collectgarbage()
    end
    return new_dict
end

function dict:save(fname)
    if self.opt.allowSaving then
        local fw = io.open(fname .. ".tmp", "w")
        if fw == nil then
            error("saving dict failed: ", fname)
        end
        if true then
            -- Save in text format.
            for i = 1, self.num_symbols do
                fw:write(self:index_to_symbol(i) .. "\t"
                             .. self.index_to_freq[i] .. "\n")
            end
            fw:close()
        else
            -- Save in Torch format (not used yet).
            fw:close()
            local data = self:get_shared()
            torch.save(fname .. ".tmp", data)
        end
        if not os.rename(fname .. '.tmp', fname) then
            print('WARNING: renaming failed')
        end
    end
end

function dict:get_shared()
    local shared = {}
    shared.num_inputs = self.num_inputs
    shared.num_labels = self.num_labels
    shared.num_symbols = self.num_symbols
    if self.opt.dictStoreIndexToFreq == true then
        shared.index_to_freq = self.index_to_freq
    end
    shared._index_to_symbol = self._index_to_symbol
    shared._symbol_to_index = self._symbol_to_index
    shared.opt = self.opt
    return shared
end


function dict:index_to_symbol(index)
    if self.unks ~=nil and self.unks[index] ~= nil then
        -- Deal with unknown words.
        local word
        local ind = self.unks[index]
        if ind ~= nil then
            word = self._unkindex_to_symbol[ind]
        end
        if word == nil then
            -- Fall back to printing "unknown" if we don't know the word.
            word = self._index_to_symbol[index]
        end
        return word
    else
        -- Usual conversion with standard dictionary.
        return self._index_to_symbol[index]
    end
end

function dict:symbol_to_index(symbol)
    return self._symbol_to_index[symbol]
end

-- Return the dictionary elements that are used for ranking
-- as candidate labels. By default this is just the entire set of symbols.
function dict:get_labels()
    if self.labels ~= nil then
        return self.labels
    end
    -- Build labels for the first time.
    self.labels = require('tds').hash()
    for i = 1, self.num_symbols do
        self.labels[i] = torch.Tensor({{i, 1}})
    end
    return self.labels
end

function dict:text_to_vector(s)
    if not self.parser then
        local parserlib =
            require('library.parse')
        self.parser = parserlib:new(self.opt)
    end
    assert(s and s:len() > 0, 'dict.t2v: make sure you provided a nonempty ' ..
        'string, and used dict:t2v not dict.t2v')
    return self.parser:parse_test_time(s, self)
end

function dict:text_to_vector_legacy(s)
    local x = {}
    local words = pl.utils.split(s, ' ')
    for _, w in pairs(words) do
        local idx = self:symbol_to_index(w)
        if idx ~= nil then
            x[#x + 1] = {idx, 1}
        end
    end
    return x
end

function dict:vector_to_text(s)
    assert(s and string.find(torch.type(s), 'Tensor'), 'dict.v2t: make sure ' ..
        'you provided a valid tensor, and used dict:v2t not dict.v2t')
    local t = ""
    for i=1, s:size(1) do
        local ind
        if s:dim() == 1 then
            ind = s[i]
        else
            ind = s[i][1]
        end
        local w = self:index_to_symbol(ind)
        if w == nil then
            -- This is likely a time feature or something
            -- else not in the dictionary.
            w = '<ind=' .. ind .. '>'
        end
        t = t .. w
        t = t .. " "
    end
    if t:len() > 1 then
        -- Remove last space added.
        t = t:sub(1, -2)
    end
    return t
end

dict.v2t = dict.vector_to_text
dict.t2v = dict.text_to_vector
dict.t2v_l = dict.text_to_vector_legacy

return dict
