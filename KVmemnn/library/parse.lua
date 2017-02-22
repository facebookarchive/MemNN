-- Copyright 2004-present Facebook. All Rights Reserved.
-- To parse a line of the given input.

local pl = require('pl.import_into')()
local hash = require('library.hash')
local tds = require('tds')

local parser = {}

function parser:new(opt)
    local new_parser = {}
    setmetatable(new_parser, {__index = parser} )
    new_parser.opt = opt
    return new_parser
end

-- Simple preprocessing
function parser:preprocess(s)
    if self.opt.dictRemoveChars ~= nil then
        for i = 1, self.opt.dictRemoveChars:len() do
            local c = self.opt.dictRemoveChars:sub(i, i)
            if c == '.' then c = '%.'; end
            if c == '?' then c = '%?'; end
            s = s:gsub(c, '')
        end
    end
    if self.opt.preprocessText then
        -- collapse asterisks
        s = s:gsub('[%*%s]*[%*]+[%*%s]*', ' * ')
        -- collapse question marks
        s = s:gsub('[%?%s]*[%?]+[%?%s]*', ' ? ')
        -- collapse exclamation points
        s = s:gsub('[!%s]*[!]+[!%s]*', ' ! ')
        -- collapse commas
        s = s:gsub('[,%s]*[,]+[,%s]*', ' , ')
        -- collapse equality
        s = s:gsub('[=%s]*[=]+[=%s]*', ' = ')
        -- collapse dashes (but leave hyphens and negative signs)
        s = s:gsub('[%-%s]*%-+%s*%-+[%-%s]*', ' -- ')
        -- collapse and replace pipes
        s = s:gsub('[|%s]*[|]+[|%s]*', ' __pipe__ ')
        -- collapse periods / ellipses (keeps as ... if 3+ .'s in a row)
        -- s = s:gsub('[%.]+[%s]*[%.][%s]*[%.]+', ' <ELLIPSES> ')
        -- s = s:gsub('[%.%s]*[%.]+[%s%.]*', ' . ')  -- watch out for "Dr." etc
        -- s = s:gsub('<ELLIPSES>', '...')
        -- s = s:gsub('%s*%.%.%.%s*[%.%s]*', ' ... ')
    end
    if self.opt.customGsubs then
        local delimit = self.opt.customGsubs:sub(1,1)
        local gsubs = pl.utils.split(self.opt.customGsubs:sub(2, -1), delimit)
        assert(#gsubs % 2 == 0, 'Need an even number of pattern:replace pairs '
            .. 'in self.opt.customGsubs, with a delimiter as the first char')
        for i = 1, #gsubs, 2 do
            local pattern = gsubs[i]
            local replace = gsubs[i + 1]
            s = s:gsub(pattern, replace)
        end
    end
    -- collapse whitespace
    s = s:gsub('%s%s+', ' ')
    if self.opt.dictLowercaseData then
        return s:lower()
    end
    return s
end

-- Remove hyphens unless associated with digit
function parser:preprocess_word(w)
    -- remove leading '-'s if the second character is not a digit
    while w:sub(1, 1) == '-' and w:sub(2, 2):find('[0-9]') == nil do
        w = w:sub(2, -1)
    end
    -- remove trailing '-'s
    while w:sub(-1) == '-' do
        w = w:sub(1, -2)
    end
    return w
end

-- optionally include chars_to_ignore to remove some items from results,
-- ie if chars_to_ignore = ',.' then will not return words '.' or ','
function parser:split(s, c, chars_to_ignore)
   local t={}
   while true do
      local f = s:find(c)
      if f == nil then
         if s:len() > 0 then
            table.insert(t, s)
         end
         break
      end
      local word = s:sub(1, f - 1)
      local delim = s:sub(f, f)
      if self.opt.preprocessWords then
         word = self:preprocess_word(word)
         delim = self:preprocess_word(delim)
      end
      if chars_to_ignore then
         local word_esc = pl.utils.escape(word)
         if word_esc and not chars_to_ignore:find(word_esc) then
             table.insert(t, word)
         end
         local del_esc = pl.utils.escape(delim)
         if del_esc and not chars_to_ignore:find(del_esc) then
             table.insert(t, delim)
         end
      else
         table.insert(t, word)
         table.insert(t, delim)
      end
      s=s:sub(f + 1, -1)
   end
   return t
end

function parser:unknown_word(word, t, dict)
    if self.opt.dictUseUNK then
        if self.opt.dictUseUNKHash then
            if dict._unkf == nil then
                -- Convert hashes to torch floats so we don't lose precision.
                dict._unkf = torch.FloatTensor(1)
            end
            dict._unkf[1] = hash.hash(word) % 1e9 + dict.num_symbols
            local h = dict._unkf[1]
            if dict._unk == nil then
                dict._unk = tds.hash()
            end
            dict._unk[h] = word
            t[#t + 1] = h
        else
            t[#t + 1] = dict['<UNK>']
            if dict.symbol_to_index ~= nil then
                t[#t + 1] = dict:symbol_to_index('<UNK>')
            end
        end
    else
        -- Do nothing.
    end
end

-- s: input string
-- out_vec: output vector version of string
-- dict: dictionary containing words
function parser:parse_internal(s, out_vec, dict, addword_fun, addword_arg)
    local v = {}
    if s == nil then
        return out_vec
    end
    local s = self:preprocess(s)
    local st = self:split(s, self.opt.dictWhiteSpace)
    local s2i = dict.symbol_to_index ~= nil
    local word_pos = 1
    local max_ngram = #st
    -- Try the existing dictionary on different n-gram sizes...
    while word_pos <= #st do
        -- Find if there is an n-gram match in this position.
        local word = ''
        local found = ''
        local found_pos = nil
        if self.opt.dictMaxNGramSz ~= nil then
            max_ngram = math.min(#st, word_pos + self.opt.dictMaxNGramSz - 1)
        end
        for j = word_pos, max_ngram do
            word = word .. st[j]
            if dict[word] ~= nil or
            (s2i and dict:symbol_to_index(word) ~= nil) then
                found = word
                found_pos = j
            end
        end
        if found:len() ~= 0 then
            if addword_fun ~= nil then
                addword_fun(addword_arg, found, 1)
            else
                if s2i then
                    v[#v + 1] = dict:symbol_to_index(found)
                else
                    v[#v + 1] = dict[found]
                end
            end
            word_pos = found_pos
        elseif word:len() > 0 then
            local word = st[word_pos]
            if addword_fun ~= nil then
                addword_fun(addword_arg, word, 1)
            else
                self:unknown_word(st[word_pos], v, dict)
            end
        end
        word_pos = word_pos + 1
    end
    if self.opt.dictRemoveWhiteSpace ~= nil then
        -- Remove space and , from the data (dumb preprocessing).
        for i = 1, #v do
            local s
            if dict[v[i]] ~= nil then
                s = pl.utils.escape(dict[v[i]])
            end
            if s ~= nil and self.opt.dictRemoveWhiteSpace:find(s) ~= nil then
                -- Skip.
            else
                out_vec[#out_vec + 1] = v[i]
            end
        end
    else
        -- Keep all the spaces and such.
        out_vec = v
    end
    return out_vec
end

function parser:parse_supporting_facts(s, dict)
    if s == nil or s:len() == 0 then return; end
    local d = pl.utils.split(s, ' ')
    local sfs = torch.FloatTensor(#d)
    local cnt = 1
    for _, sf in pairs(d) do
        sfs[cnt] = tonumber(sf)
        cnt = cnt + 1
    end
    return sfs
end

function parser:parse_candidates(s, dict)
    local c = tds.hash()
    local c_strings = tds.hash()
    if s == nil then return; end
    local d = pl.utils.split(s, '|')
    for _, ci in pairs(d) do
        local ci0 = self:preprocess(ci)
        -- do not add duplicates (after preprocessing)
        if c_strings[ci0] == nil then
            local cip = self:parse_internal(ci, {}, dict)
            if #cip > 0 then
                c[#c + 1] = torch.FloatTensor(cip)
                c_strings[ci0] = 1
            end
        end
    end
    return c
end

function parser:parse(s, dict)
    local in_vec = {}
    local out_vec = {}
    -- First, get rid of the index at the beginning,
    -- this doesn't go in the dictionary.
    local i1 = s:find(' ')
    if i1 ~= nil then
        local index = s:sub(1, i1) * 1.0
        in_vec[#in_vec + 1] = index
        s = s:sub(i1 + 1, -1)
    end
    -- Now split potentially into QA pairs.
    local all = pl.utils.split(s, '\t')
    in_vec  = self:parse_internal(all[1], in_vec, dict)
    out_vec = self:parse_internal(all[2], out_vec, dict)

    local res = {in_vec, out_vec}

    -- (potential) supporting facts in slot 3
    res[3] = self:parse_supporting_facts(all[3], dict)

    -- (potential) candidates in slot 4
    res[4] = self:parse_candidates(all[4], dict)

    return res
end

-- This is very similar to parser:parse_internal.
function parser:parse_internal_test_time(s, out_vec, dict)
    local v = {}
    if s == nil then return; end
    s = self:preprocess(s)
    local st = self:split(s, self.opt.dictWhiteSpace)
    local word_pos = 1
    local found_pos = nil
    -- Try the existing dictionary on different n-gram sizes...
    while word_pos <= #st do
        -- Find if there is an n-gram match in this position.
        local word = ''
        local found = ''
        for j = word_pos, #st do
            word = word .. st[j]
            if dict:symbol_to_index(word) ~= nil then
                found = word
                found_pos = j
            end
        end
        if found:len() == 0 then
            self:unknown_word(st[word_pos], v, dict)
        else
            v[#v + 1] = dict:symbol_to_index(found)
            word_pos = found_pos
        end
        word_pos = word_pos + 1
    end
    if self.opt.dictRemoveWhiteSpace ~= nil then
        -- Remove space and , from the data (dumb preprocessing).
        for i = 1, #v do
            local s
            if v[i] <= dict.num_symbols then
                s = dict:index_to_symbol(v[i])
            end
            if s ~= nil then s = pl.utils.escape(s) end
            if s ~= nil and self.opt.dictRemoveWhiteSpace:find(s) ~= nil then
                -- Skip.
            else
                out_vec[#out_vec + 1] = v[i]
            end
        end
    else
        -- Keep all the spaces and such.
        out_vec = v
    end
    return out_vec
end

function parser:parse_test_time(s, dict)
    local vec = {}
    self:parse_internal_test_time(s, vec, dict)
    local r = torch.FloatTensor(#vec, 2)
    for i = 1, #vec do
        r[i][1] = vec[i]
        local freq = 1
        if dict.index_to_freq:size(1) > vec[i] then
            freq = dict.index_to_freq[vec[i]]
        end
        r[i][2] = 1 / math.pow(freq + 10, self.opt.dictTFIDFPow)
    end
    if r:dim() == 0 then
        return
    end
    local xwt = r:t()
    local nwv = xwt[2]:norm()
    if nwv > 0.00001 then xwt[2]:div(nwv); end
    return r
end

return parser
