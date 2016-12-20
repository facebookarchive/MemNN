-- Copyright 2004-present Facebook. All Rights Reserved.
-- Hashing class.

require('torch')
local tds = require 'tds'
local varray =
    require('library.vector_array')
local util = require('library.util')

local hash = {}
hash.__index = hash

function hash:new(opt, dict)
    local new_mem = {}
    setmetatable(new_mem, hash)
    new_mem.opt = opt

    -- vector of tensors to store facts
    -- stored as a pair (facts1, facts2) for input and output, if applicable.
    new_mem.facts1 = tds.hash()
    new_mem.facts2 = tds.hash()

    -- vector of maps from hash_index to indices of facts
    new_mem.facts_hash = tds.hash()

    new_mem.dict = dict

    return new_mem
end

function hash:load_dictionary_clusters(fname)
    self.dict_clusters = torch.load(fname)
    self.num_clusters = self.dict_clusters:max()
    print("[loaded " .. self.num_clusters .. " clusters]")
end

local function count(cnt, index)
    if cnt[index] == nil then
        return 0
    else
        return cnt[index]
    end
end

local function add_to_count(cnt, index)
    if cnt[index] == nil then
        cnt[index] = 0
    end
    cnt[index] = cnt[index] + 1
end

function hash:add_facts(data)
    print "[counting facts sizes..]"
    local total_lines = data:num_loaded_examples()
    local start_time = os.time()
    local cnts = tds.hash()
    data:reset_examples()
    local facts_sz = 0
    while true do
        local ex, finished = data:get_next_example()
        if finished == true then break; end
        facts_sz = facts_sz + 1
        util.log_progress(facts_sz, total_lines, start_time, opt.logEveryNSecs,
                          "count facts:")
        local x = ex[1]
        if x:dim() > 0 and x:size(1) >= 1 then
            local h = self:get_hash_indices(x, true)
            if h:dim() > 0 then
                for j = 1, h:size(1) do
                    add_to_count(cnts, h[j])
                end
            end
        end
    end
    print "[creating facts hash..]"
    for j, cntsj in pairs(cnts) do
        if cntsj < self.opt.memHashMaxHashes then
            self.facts_hash[j] = torch.DoubleTensor(cntsj)
        else
            if not self.opt.memHashThrowAwayIfOverMaxHashes then
                self.facts_hash[j] = torch.DoubleTensor(
                    self.opt.memHashMaxHashes)
            end
        end
    end
    -- Create an empty fact hash as well.
    self.facts_hash[0] = torch.DoubleTensor(0)

    print "[building facts hash..]"
    -- Stores the absolute fact position, document index and
    -- sentence position within document as 3 values per fact.
    self.facts_ind = torch.DoubleTensor(facts_sz, 3)
    local doc_index = 0
    -- Reset the counts.
    cnts = tds.hash()
    data:reset_examples()
    local start_time = os.time()
    local i = 0
    while true do
        i = i + 1
        util.log_progress(i, total_lines, start_time, opt.logEveryNSecs,
                          "build facts hash")
        local ex, finished = data:get_next_example()
        if finished == true then break; end
        local x = ex[1]
        if x:dim() > 0 and x:size(1) >= 1 then
            local y = ex[2]
            if self.opt.maxHashSizeSortedByTFIDF <= 0 then
                -- If no rescoring with TFIDF, we cut off the weights,
                -- and only keep the words.
                x = x:t()[1]
                y = y:t()[1]
            end
            self.facts1[i] = x
            self.facts2[i] = y
            self.facts_ind[i][1] = i
            -- Keep track of doc and sentence indices.
            local sentence_index = ex.index
            if sentence_index == 1 then
                doc_index = doc_index + 1
            end
            self.facts_ind[i][2] = doc_index
            self.facts_ind[i][3] = sentence_index
            local ind = i
            local h = self:get_hash_indices(x, true)
            if h:dim() > 0 then
                for j = 1, h:size(1) do
                    local hi = self.facts_hash[h[j]]
                    if hi ~= nil and hi:dim() > 0
                    and hi:size(1) > count(cnts, h[j]) then
                        add_to_count(cnts, h[j])
                        hi[cnts[h[j]]] = ind
                    end
                end
            end
        else
            if self.opt.maxHashSizeSortedByTFIDF <= 0 then
                local NULL = torch.Tensor({1})
                self.facts1[i] = NULL
                self.facts2[i] = ex[2]:t()[1]
            else
                local NULL = torch.Tensor({{1,0}})
                self.facts1[i] = NULL
                self.facts2[i] = NULL
            end
            self.facts_ind[i]:fill(0)
        end
    end
end

function hash:get_hash_indices(x)
    local h = {}
    local function add_words(x)
        for i = 1, x:size(1) do
            local wi = x[i]
            -- We only hash words below a certain frequency, or
            -- if the word is an UNKnown (> dictionary size).
            if wi > self.dict.index_to_freq:size(1)
            or self.dict.index_to_freq[wi] < self.opt.memHashFreqCutoff then
                if self.opt.memClusterHashing then
                    local ind = self.dict_clusters[wi]
                    if self.opt.memDictHashing then
                        ind = ind + self.dict.num_symbols
                    end
                    h[#h + 1] = ind
                end
                if self.opt.memDictHashing then
                    h[#h + 1] = wi
                end
            end
        end
    end
    if x:dim() == 2 then
        x = x:t()[1]
    end
    add_words(x)
    h = torch.Tensor(h)
    return h
end

function hash:size()
    if self.facts1_va ~= nil then
        return self.facts1_va:size()
    else
        return #self.facts1
    end
end

function hash:get_fact(index)
    if index < 1 then return nil; end
    local fact1, fact2
    if self.facts1_va ~= nil then
        if self.facts1_va:size() < index then return nil; end
        fact1 = self.facts1_va:get(index)
        fact2 = self.facts2_va:get(index)
    else
        fact1 = self.facts1[index]
        fact2 = self.facts2[index]
    end
    local find = self.facts_ind[index]
    return fact1, fact2, find
end

function hash:get_candidate_set(qx)
    -- Given an input this returns a vector of tensors representing
    -- the candidate set.
    local candidates1 = tds.hash()
    local candidates2 = tds.hash()
    -- The document and sentence indices of the resulting facts.
    local cinds = tds.hash()
    -- Tracks which fact index each candidate came from.
    local hashes = {}
    local hinds = self:get_hash_indices(qx)
    if hinds:dim() == 0 then
        -- No candidates found.
        return candidates1, candidates2
    end
    for i = 1, hinds:size(1) do
        local wi = hinds[i]
        local h
        if self.facts_hash_va ~= nil then
            if self.facts_hash_va:size() >= wi then
                h = self.facts_hash_va:get(wi)
            end
        else
            h = self.facts_hash[wi]
        end
        if h ~= nil and h:dim() ~= 0 and h[1] ~= 0 then
            for j = 1, h:size(1) do
                local fact1, fact2, find = self:get_fact(h[j])
                -- Prevent the same fact being included twice in the set.
                local function add_fact(fact1, fact2, find)
                    local hashid = find[1]
                    if hashes[hashid] == nil then
                        hashes[hashid] = true
                        candidates1[#candidates1 + 1] = fact1
                        candidates2[#candidates2 + 1] = fact2
                        cinds[#cinds + 1] = find
                    end
                end
                if self.opt.hashDocWindow == 0 then
                        add_fact(fact1, fact2, find)
                else
                    -- Add a window around the fact (possibly a whole doc)
                    local wsz = self.opt.hashDocWindow
                    local doc = find[2]
                    for i = -wsz, wsz do
                        if h[j] + i < (#self.facts_ind)[1] and h[j] + i > 0 then
                            local f1, f2, fi = self:get_fact(h[j] + i)
                            if fi ~= nil and fi[2] == doc then
                                add_fact(f1, f2, fi)
                            end
                        end
                    end
                end
            end
        end
    end
    return candidates1, candidates2, cinds
end

function hash:sortByTFIDF(qx, candidates1, candidates2)

    if not self.qx_vec then
        self.qx_vec = torch.FloatTensor(self.dict.num_symbols)
    end
    self.qx_vec:zero()
    for i = 1, qx:size(1) do
        local index = qx[i][1]
        -- Ignore <NULL>, <EOS> and <UNK> words.
        if index > 3 then
            self.qx_vec[index] = qx[i][2]
        end
    end

    if not self.sparse then self.sparse = require('sparse') end
    local qx_norm = self.sparse.SdotD(qx, self.qx_vec)
    local cscores = torch.Tensor(#candidates1)
    for i = 1, #candidates1 do
        local c1i = candidates1[i]
        cscores[i] = self.sparse.SdotD(c1i, self.qx_vec)
    end

    local val,idx = torch.sort(cscores, true)
    local filt_cands1 = {}
    local filt_cands2 = {}
    local i = 0
    while #filt_cands1 < self.opt.maxHashSizeSortedByTFIDF do
        i = i + 1
        if i > #candidates1 then break end
        if not self.opt.removeQueryDuplicatesFromHashes or
        math.abs(val[i] - qx_norm) > 1e-5 then
            -- we check the norm difference to potentially
            -- remove qx from the memory (when hashing the train)
            filt_cands1[#filt_cands1 + 1] = candidates1[idx[i]]
            filt_cands2[#filt_cands2 + 1] = candidates2[idx[i]]
        end
    end
    return filt_cands1, filt_cands2
end

function hash:convert_to_vector_array()
    print "[converting to vector array..]"
    local va = varray:create_from_tds_hash(self.facts_hash)
    self.facts_hash_va = va
    va = varray:create_from_tds_hash(self.facts1)
    self.facts1_va = va
    va = varray:create_from_tds_hash(self.facts2)
    self.facts2_va = va
end

function hash:save(filename)
    print "[saving hash..]"
    torch.save(filename .. '.facts_ind', self.facts_ind)
    torch.save(filename .. '.facts1', self.facts1)
    torch.save(filename .. '.facts2', self.facts2)
    torch.save(filename .. '.facts_hash', self.facts_hash)
end

function hash:load(filename)
    print("[load hash:" .. filename .. "]")
    self.facts_ind = torch.load(filename .. '.facts_ind')
    self.facts1 = torch.load(filename .. '.facts1')
    self.facts2 = torch.load(filename .. '.facts2')
    self.facts_hash = torch.load(filename .. '.facts_hash')
end

function hash:save_va(filename)
    print "[saving hash..]"
    torch.save(filename .. '.facts_ind', self.facts_ind)
    self.facts1_va:save(filename .. '.facts1_va')
    self.facts2_va:save(filename .. '.facts2_va')
    self.facts_hash_va:save(filename .. '.facts_hash_va')
end

function hash:load_va(filename)
    print("[load hash:" .. filename .. "]")
    self.facts_ind = torch.load(filename .. '.facts_ind')
    self.facts1_va = varray:load(filename .. '.facts1_va')
    self.facts2_va = varray:load(filename .. '.facts2_va')
    self.facts_hash_va = varray:load(filename .. '.facts_hash_va')
end

return hash
