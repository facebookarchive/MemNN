-- Copyright 2004-present Facebook. All Rights Reserved.
-- Class for handling training and eval data.

require('torch')
local VectorArray =require('library.vector_array')
local thread_utils = require('library.thread_utils')
local hashlib = require('library.hash')
local pl = require('pl.import_into')()
local tds = require('tds')

local ffi = require('ffi')
local C = ffi.load('libmemnn.so')
ffi.cdef [[
void addTFIDF(float* inp, float* out, int size, double TFIDFPow, double* freqs);
void resolveMinOcc(
    int dim, int dictMinOcc, long* sizes, float* x, double* index_to_freq
);
]]

local data = {}
data.__index = data

function data:add_cmdline_options(opt)
    opt:option('trainData',
               nil,
               "filename of training data", 'data')
    opt:option('rankLabelDocuments', false,
               'rank outputs as document vectors, rather than single labels',
               'data', true)
    opt:option('sampleCands', 1, 'ratio of times sample real cands', 'model')
    opt:option('negSampleFromSameSrc', false, 'sample negative documents '
                   .. 'from the same source as the positive one',
               'data', true)
    opt:option('negSampleFromMemY', 0, 'sample negative labels from memy, '
                   .. 'parameter controls the ratio of negative sampled from '
                   .. 'memy vs random (default 0 = no memy)')
    opt:option('threadsShareData', false, 'threads share the same data, '
                   .. 'otherwise stream different data into each thread',
               'data', true)
    opt:option('memHashFile', nil,
               'hashing into memory fixed file of knowledge', 'data')
    opt:option('starSpaceMode', false, 'concatenate memories into LHS x',
               'data')
    opt:option('datasetDistbn', false,
               'probability of drawing from each dataset', 'data')

    -- Parsing options.
    opt:option('preprocessText', false, 'Apply advanced preprocessing to text '
        .. 'when building dictionary and data.', 'parse')
    opt:option('customGsubs', nil, 'Custom gsub arguments. First char is a '
        .. 'delimiter, followed by pattern-replace pairs. (e.g. ";%s%s+; "'
        .. ' means replace all continuous whitespace with a single space',
        'parse')

    -- Dictionary options.
    opt:option('dictStoreIndexToFreq', true, 'keep this field', 'data')
    opt:option('dictTFIDF', true, 'apply TF normalization', 'data')
    opt:option('dictTFIDFPow', 0, 'TF normalization hyperparam', 'data')
    opt:option('dictTFIDFLabel', false,
               'apply TF normalization to label', 'data')
    opt:option('dictMaxNGramSz', 1, 'max Ngram size when parsing text', 'data')
    opt:option('dictWhiteSpace', '[,.?;: ]',
               'Set of characters considered as whitespace', 'data')
    opt:option('dictRemoveWhiteSpace', ' ,.',
               'Set of characters considered as whitespace'
               .. ' to remove from input', 'data')
    opt:option('dictRemoveChars', '',
               'Set of characters removed from input', 'data')
    opt:option('dictLowercaseData', true, 'lower case the text', 'data')
    opt:option('dictUseUNK', false, 'replace unknowns with UNK', 'data')
    opt:option('dictUseUNKHash', false,
               'replace unknowns with UNK hash', 'data')
    opt:option('dictUseUNKHashLookup', false,
               'lookup unknown words in dictionary using UNK hash'
               .. '- this requires storing the hash as non-contiguous', 'data')
    opt:option('dictNumUNKs', 0, 'replace unknowns with UNK hash', 'data')
    opt:option('dictMinOcc', nil,
               'do not load items into dict with less than this frequency',
               'dict', true)

    -- Memory and Hash options.
    opt:option('dataMaxMemSize', 10000,
               'largest memory stored in an example', 'data')
    opt:option('dictHashBelowFreq', 0, 'replace unknowns with UNK hash', 'hash')
    opt:option('hashLastN', 1, 'add hashes for up to N previous sentences '
                   .. 'in the dialog', 'hash')
    opt:option('memHashY', true, 'hash things in Y position as well', 'hash')
    opt:option('memDictHashing', true, 'hash based on words', 'hash')
    opt:option('memHashFreqCutoff', 1000, "don't hash too freq words", 'hash')
    opt:option('memHashMaxHashes', 1000,
               "don't hash more than this number of memories per hash bucket",
               'hash')
    opt:option('memHashThrowAwayIfOverMaxHashes', true,
               "don't keep a hash at all if memHashMaxHashes reached "
               .. 'for a given bucket',
               'hash')
    opt:option('hashDocWindow', 0,
               "retrieve memories in same document within a given window size "
               .. "around each retrieved memory from the hash", 'hash')
    opt:option('maxHashSizeSortedByTFIDF', -1,
               "sort the hashes using tfidf score with the current example "
               .. "and keep the top ones (default -1 = keep all unsorted)",
               'hash')
    opt:option('removeQueryDuplicatesFromHashes', false,
               "check if the query and its answer are included in the hashes"
                   .. "and remove it (useful when hashing the train)", 'hash')
    opt:option('notUseHashInMemory', false, 'set to true if one wants to get '
                   .. 'hashes but not add them to the memory')
end

function data:num_loaded_examples()
    if not self.num_loaded_ex then
        self.num_loaded_ex = 0
        for i = 1, #self.examples_x do
            self.num_loaded_ex = self.num_loaded_ex + self.examples_x[i]:size()
        end
    end
    return self.num_loaded_ex
end

function data:get_input(ex)
    return ex[1]
end

function data:is_null(x)
    if x == nil then return true; end
    local dim = x:dim()
    local fst = x[1]
    if dim == 1 then
        if fst == self.NULL_X or fst == 0 then return true end
    elseif dim == 2 then
        fst = fst[1]
        if fst == self.NULL_X or fst == 0 then return true end
    end
    return false
end

function data:rank_label_documents()
    return self.rankLabelDocuments[self.dataset]
end

function data:get_positive_label(ex)
    local a = ex[2]
    if self:is_null(a) then return nil end

    if self:rank_label_documents() then
        -- We are ranking documents, so return the whole label
        -- vector from the example.
        return a
    else
        -- Otherwise, we are ranking labels, so
        -- return one of the labels from the label vector.
        local index
        if a:dim() == 1 then
            index = a[math.random(a:size(1))]
        else
            index = a[math.random(a:size(1))][1]
        end
        self.y[1][1] = index
        self.pos = index
        return self.y
    end
end

-- Check if two example vectors have the same indices or not.
-- This is used to find a negative document vector that differs
-- from the positive one.
local function same_label(y1, y2)
    if y1:size(1) ~= y2:size(1) then return false; end
    for i = 1, y1:size(1) do
        if y1[i][1] ~= y2[i][1] then return false; end
    end
    return true
end

function data:get_random_xy_pair()
    local src = math.random(#self.examples_x)
    local index = math.random(self.examples_x[src]:size())
    local x = self.examples_x[src]:get(index)
    if x:size(1) > 1 then
        x = x:sub(2, x:size(1))
    else
        x = torch.Tensor({1})
    end
    local y = self.examples_y[src]:get(index)
    return x, y
end

function data:get_random_label_document()
    local src
    if self.opt.negSampleFromSameSrc then
        src = self.dataset
    else
        src = math.random(#self.examples_y)
    end
    local index = math.random(self.examples_y[src]:size())
    local nex = {}
    nex[2] = self.examples_y[src]:get(index)
    return self:get_positive_label(nex)
end

function data:get_negative_label(ex)
    if self.opt.useCandidateLabels and ex.cands ~= nil and #ex.cands > 0
    and math.random() <= self.opt.sampleCands then
        -- Get one of the candidates.
        -- We loop trying a few times to find a label that is different
        -- to the current one.
        local ny
        for i = 1, 10 do
            local index = math.random(#ex.cands)
            ny = ex.cands[index]
            if not same_label(ex[2], ny) then break; end
        end
        return ny
    else
        local ny
        if self:rank_label_documents() then
            -- We are ranking documents, so return the whole label
            -- vector of a random example.
            while true do
                local ny
                if self.opt.negSampleFromMemY
                and self.memy ~= nil and #self.memy > 0
                and math.random() < self.opt.negSampleFromMemY then
                    -- we can use memories as negative (to penalize
                    -- repeating the same kind of thing)
                    ny = self.memy[math.random(#self.memy)]
                end
                if not ny or self:is_null(ny) then
                    -- default strategy (random)
                    ny = self:get_random_label_document()
                end
                if not self:is_null(ny) then
                    ny = self:resolveUNK(ny, false)
                    ny = self:addTFIDF(ny, true)
                    return ny
                end
            end
        else
            while true do
                ny = math.random(self.dict.num_symbols)
                if ny ~= self.pos then break; end
            end
            self.yneg[1][1] = ny
            return self.yneg
        end
    end
end

function data:resolveUNK(x, create_new)
    if create_new == nil then create_new = true; end
    -- cache options
    local dictMinOcc = self.opt.dictMinOcc
    local dictUseUNKHash = self.opt.dictUseUNKHash

    if dictMinOcc or dictUseUNKHash then
        x = x:clone()
    else
        -- nothing to do
        return x
    end

    local index_to_freq = self.dict.index_to_freq
    if dictUseUNKHash then
        -- Search for UNKnowns in the vector...
        for i = 1, x:size(1) do
            local id = x[i]
            local hashRare = false
            if self.opt.dictHashBelowFreq ~= nil then
                if id < self.dict.num_symbols and
                index_to_freq[id] < self.opt.dictHashBelowFreq then
                    hashRare = true
                end
            end
            if hashRare or id > self.dict.num_symbols then
                local unk_id = self.dict.id_to_unkid[id]
                if unk_id == nil then
                    if create_new then
                        if self.dict.unks_cnt < self.opt.dictNumUNKs then
                            self.dict.unks_cnt = self.dict.unks_cnt + 1
                        else
                            -- Should we print a warning that there are too
                            -- many unknowns?
                        end
                        local unk_id = self.dict.unks_cnt + 3
                        -- Could use this to randomly permute:
                        --local id = self.unks_take[self.unks_cnt] + 3
                        self.dict.unkid_to_id[unk_id] = id
                        self.dict.id_to_unkid[id] = unk_id
                        x[i] = unk_id
                    else
                        if id > self.dict.num_symbols then
                            x[i] = self.NULL_X
                        end
                    end
                else
                    x[i] = unk_id
                end
            end
        end
    end
    -- Only compute min occ after doing the relevant hashing.
    if dictMinOcc then
        local x_size = x:size()
        C.resolveMinOcc(
            x_size:size(),
            dictMinOcc,
            torch.data(x_size),
            torch.data(x),
            torch.data(index_to_freq)
        )
    end
    return x
end


function data:reset()
    -- Members used for storing current example.
    -- This is so we do not have to keep creating them.
    self.bigx = self:Tensor(10000, 2):fill(1)
    self.x = self:Tensor(10000, 2):fill(1)
    self.a = self:Tensor(10000, 2):fill(1)
    self.y = self:Tensor(1, 2):fill(1)
    self.yneg = self:Tensor(1, 2):fill(1)
    -- Special vector for the NULL memory.
    self.NULL = self:Tensor(1, 2):fill(1)
    self.NULL_X = self.NULL[1][1]
    self.num_loaded_ex = nil

    self:init_dataset_distbns()

    self:reset_examples()

    self.rankLabelDocuments = {}
    if type(self.opt.rankLabelDocuments) == 'string' then
        local rlds = pl.utils.split(self.opt.rankLabelDocuments, ',')
        if #rlds ~= #self.sentence_pos then
            print("[WARNING: rankLabelDocuments:"
                      .. self.opt.rankLabelDocuments
                      .. " does not match data class with "
                      .. #self.sentence_pos .. " datasets.]")
        end
        for i = 1, #self.sentence_pos do
            local rld = false
            if rlds[i] == 'true' then rld = true; end
            self.rankLabelDocuments[i] = rld
        end
    else
        for i = 1, #self.sentence_pos do
            self.rankLabelDocuments[i] = self.opt.rankLabelDocuments
        end
    end
end

function data:reset_examples()
    self.sentence_pos = {}
    for i = 1, #self.examples_x do
        self.sentence_pos[i] = 0
    end
    self.dataset = 1
    self:reset_conversation()
end

function data:reset_conversation()
    if self.opt.dictUseUNKHash then
        self.dict.unkid_to_id = nil
        self.dict.id_to_unkid = nil
        self.dict.unkid_to_id = tds.hash()
        self.dict.id_to_unkid = tds.hash()
        self.dict.unks_cnt = 0
        -- Could use this to randomly permute:
        -- self.unks_take = torch.randperm(self.opt.dictNumUNKs)
    end
end

function data:get_next_example(loop_through_data)
    if self.sentence_pos[self.dataset] >=
    self.examples_x[self.dataset]:size() then
        if self.dataset >= #self.sentence_pos then
            -- We hit then end of the data.
            if loop_through_data then
                self.dataset = 1
                self.sentence_pos[self.dataset] = 1
            else
                -- Indicate the end of the dataset.
                return nil, true
            end
        else
            -- Move on to the next source.
            self.dataset =  self.dataset + 1
            self.sentence_pos[self.dataset] = 1
        end
    else
        self.sentence_pos[self.dataset] = self.sentence_pos[self.dataset] + 1
    end
    local index =
        self.examples_x[self.dataset]:get(self.sentence_pos[self.dataset])[1]
    if index == 1 then self:reset_conversation(); end
    return self:get_current_example()
end

function data:get_next_example_partitioned(jobid, numjobs, maxexamples)
    assert(self.examples_st, 'Dialog start indices not available')
    assert(jobid, 'Please provide valid jobid')
    assert(numjobs, 'Please provide valid numjobs')
    assert(jobid <= numjobs, 'Invalid jobid / numjobs arguments')

    -- first time this method is called, select first dialog for this job
    if not self.curr_dialog then
        self:reset_examples()
        self.curr_dialog = jobid
        while self.curr_dialog > self.examples_st[self.dataset]:size() do
            -- no data for this thread
            if self.dataset == #self.examples_st then return nil, true end
            -- else, move to next dataset
            self.curr_dialog =
                self.curr_dialog - self.examples_st[self.dataset]:size()
            self.dataset = self.dataset + 1
        end
        self.sentence_pos[self.dataset] =
            self.examples_st[self.dataset]:get(self.curr_dialog)[1]
    else
        -- otherwise increment sentence_pos
        if self.sentence_pos[self.dataset] >=
        self.examples_x[self.dataset]:size() then
            if self.dataset >= #self.sentence_pos then
                -- Indicate the end of the dataset.
                return nil, true
            else
                -- Move on to the next source.
                self.dataset =  self.dataset + 1
                self.sentence_pos[self.dataset] = 1
            end
        else
            self.sentence_pos[self.dataset] =
                self.sentence_pos[self.dataset] + 1
        end
        local index = self.examples_x[self.dataset]
            :get(self.sentence_pos[self.dataset])[1]
        -- check index: if this is a dialog start then should increment dialog
        if index == 1 then
            self:reset_conversation()
            self.curr_dialog = self.curr_dialog + numjobs
            while self.curr_dialog > self.examples_st[self.dataset]:size() do
                -- no data for this thread
                if self.dataset == #self.examples_st then return nil, true end
                -- else, move to next dataset
                self.curr_dialog =
                    self.curr_dialog - self.examples_st[self.dataset]:size()
                self.dataset = self.dataset + 1
            end
            self.sentence_pos[self.dataset] =
                self.examples_st[self.dataset]:get(self.curr_dialog)[1]
        end
    end
    -- make sure that you still want more examples
    if maxexamples then
        local num_done = 0
        for i = 1, self.dataset - 1 do
            num_done = num_done + self.examples_x[self.dataset]:size()
        end
        num_done = num_done + self.sentence_pos[self.dataset]
        if num_done > maxexamples then return nil, true end
    end
    return self:get_current_example()
end

function data:init_dataset_distbns()
    if type(self.opt.datasetDistbn) == "string" then
        local weights = pl.utils.split(self.opt.datasetDistbn, ',')
        self.randData = torch.zeros(#weights)
        for i = 1, #weights do
            self.randData[i] = tonumber(weights[i])
        end
        self.randData:div(self.randData:sum())
        for i = 2, #weights do
            self.randData[i] = self.randData[i] + self.randData[i - 1]
        end
    else
        -- Draw from datasets randomly with equal weight.
        self.randData = nil
    end
end

function data:get_random_dataset()
    local index
    if self.randData == nil then
        index = math.random(#self.sentence_pos)
    else
        local r = math.random()
        index = self.randData:size(1)
        for i = 1, self.randData:size(1) do
            if r < self.randData[i] then
                index = i
                break
            end
        end
    end
    return index
end

function data:get_random_example()
    if self.sentence_pos[self.dataset] >=
    self.examples_x[self.dataset]:size() then
        self.sentence_pos[self.dataset] = 1
    else
        self.sentence_pos[self.dataset] = self.sentence_pos[self.dataset] + 1
    end
    local index =
        self.examples_x[self.dataset]:get(self.sentence_pos[self.dataset])[1]
    if index == 1 then
        self:reset_conversation()
        -- Possibly switch datasets..
        self.dataset = self:get_random_dataset()
        -- Possibly switch to a random conversation within the dataset.
        if self.examples_st[self.dataset] ~= nil then
            local ind = math.random(self.examples_st[self.dataset]:size())
            local rand_pos = self.examples_st[self.dataset]:get(ind)[1]
            self.sentence_pos[self.dataset] = rand_pos
        end
    end
    return self:get_current_example()
end

-- This function is internal only and called by either get_random_example
-- or get_next_example.
function data:get_current_example()
    local res = {}
    -- Get next sentence.
    local x1 =
        self.examples_x[self.dataset]:get(self.sentence_pos[self.dataset])
    res.index = x1[1]
    if x1:size(1) > 1 then
        x1 = x1:sub(2, x1:size(1))
    else
        x1 = torch.Tensor({1})
    end
    local x1_preunk = x1
    x1 = self:resolveUNK(x1)
    x1 = self:addTFIDF(x1)
    self.story_index = res.index
    if self.story_index == 1 then
        self.memx = {}
        self.memy = {}
        self.memhx = {}
        self.memhx.n = 0
        self.memhy = {}
        self.memhy.n = 0
    else
        self.memx[#self.memx + 1] = self.lastx
        self.memy[#self.memy + 1] = self.lasty
        if #self.memx > self.opt.dataMaxMemSize then
            table.remove(self.memx, 1)
            table.remove(self.memy, 1)
        end
    end
    self.lastx = x1
    res.memx = self.memx
    res.memy = self.memy
    if self.hash ~= nil then    -- hashLastN
        local c1, c2  = self.hash:get_candidate_set(x1_preunk)

        if self.opt.dictUseUNKHash then
            for i = 1, #c1 do
                c1[i] = self:resolveUNK(c1[i])
                c2[i] = self:resolveUNK(c2[i])
            end
        end
        for i = 1, #c1 do
            c1[i] = self:addTFIDF(c1[i])
            c2[i] = self:addTFIDF(c2[i], true)
        end

        -- This adds c to lasthash, removes expired hashes, and adds the rest
        -- to memh (which points to either memhx or memhy)
        local function add_cur_hashes(c, lasthash, memh)
            lasthash.n = lasthash.n + 1
            lasthash[lasthash.n] = c
            for i = 1, lasthash.n - self.opt.hashLastN do
                -- these are not needed anymore
                lasthash[i] = nil
            end
            for i = lasthash.n - self.opt.hashLastN + 1, lasthash.n do
                local alist = lasthash[i]
                if alist then
                    for j = 1, #alist do
                        memh[#memh + 1] = alist[j]
                    end
                end
            end
        end
        res.memhx = {}
        add_cur_hashes(c1, self.memhx, res.memhx)
        res.memhy = {}
        add_cur_hashes(c2, self.memhy, res.memhy)

        -- Possibly do the last reply (memy) as well.
        if self.opt.memHashY and #res.memy > 0 then
            local c1, c2  =
                self.hash:get_candidate_set(res.memy[#res.memy])
            for i = 1, #c1 do
                c1[i] = self:addTFIDF(c1[i])
                c2[i] = self:addTFIDF(c2[i], true)
            end
            add_cur_hashes(c1, self.memhx, res.memhx)
            add_cur_hashes(c2, self.memhy, res.memhy)
        end

        if #res.memhx > 0 and self.opt.maxHashSizeSortedByTFIDF > 0 then
            -- Possibly filter hashes to keep the most similar (via tfidf)
            res.memhx, res.memhy = self.hash:sortByTFIDF(x1, res.memhx,
                                                         res.memhy)
        end
    else
        res.memhx = {}
        res.memhy = {}
    end

    res.cands = self:get_candidates(res)

    -- Add known supporting facts, if any.
    local ex_index = self.sentence_pos[self.dataset]
    if self.examples_sf[self.dataset] ~= nil and
        self.examples_sf[self.dataset]:size() >= ex_index
    then
        res.sfs =
            self.examples_sf[self.dataset]:get(ex_index)
    end

    local a1 =
        self.examples_y[self.dataset]:get(ex_index)
    a1 = self:resolveUNK(a1, false)
    a1 = self:addTFIDF(a1, true)
    self.lasty = a1
    res[1] = x1
    res[2] = a1

    if self.opt.starSpaceMode then
        res = self:starspace_mode(res)
    end

    res.dataset = self.dataset

    return res
end

function data:get_candidates(ex)
    local cands = {}
    if (self.opt.useCandidateLabels or self.opt.testWithCandidateLabels)
    and self.examples_ci[self.dataset]
    and self.examples_c[self.dataset] then
        local ci =
            self.examples_ci[self.dataset][self.sentence_pos[self.dataset]]
        if ci[1] > 0 and ci[2] > 0 then
            for ii = ci[1], ci[2]-1 do
                local cand = self.examples_c[self.dataset]:get(ii)
                cand = self:resolveUNK(cand)
                cand = self:addTFIDF(cand)
                table.insert(cands, cand)
            end
        end
    end
    return cands
end

function data:starspace_mode(res)
    -- Concatenate everything into x.
    local sz = res[1]:size(1)
    for i = 1, #res.memx do
        local nsz = sz + res.memx[i]:size(1)
        if nsz <= self.bigx:size(1) then
            sz = nsz
        else
            break
        end
    end
    for i = 1, #res.memy do
        local nsz = sz + res.memy[i]:size(1)
        if nsz <= self.bigx:size(1) then
            sz = nsz
        else
            break
        end
    end
    local bigx = self.bigx:sub(1, sz)

    bigx:sub(1, res[1]:size(1)):copy(res[1])
    sz = res[1]:size(1)
    for i = 1, #res.memx do
        local start = sz + 1
        local nsz = sz + res.memx[i]:size(1)
        if nsz <= self.bigx:size(1) then
            sz = nsz
            bigx:sub(start, sz):copy(res.memx[i])
        else
            break
        end
    end
    for i = 1, #res.memy do
        local start = sz + 1
        local nsz = sz + res.memy[i]:size(1)
        if nsz <= self.bigx:size(1) then
            sz = nsz
            bigx:sub(start, sz):copy(res.memy[i])
        else
            break
        end
    end
    res.old1 = res[1]
    res[1] = bigx
    -- Normalize.
    local bxt = bigx:t()
    local nwv = bxt[2]:norm()
    if nwv > 0.00001 then bxt[2]:div(nwv); end
    return res
end

function data:addTFIDF(x, isLabel, overwrite)
    -- If there's already a second dimension of weights, we don't run this.
    if x:dim() == 2 and not overwrite then return x end
    -- Check for the empty vector, e.g. for non-answers.
    if x[1] == 0 then return self.NULL; end
    local sz = x:size(1)
    local xnew
    if x:dim() == 2 then
        xnew = x -- don't reallocate tensor if x is already two-dimensional
        x = x:t()[1]
    else
        xnew = torch.FloatTensor(sz, 2)
    end
    local xwt = xnew:t()
    if (self.opt.dictTFIDF and not isLabel)
    or (isLabel and self.opt.dictTFIDFLabel) then
        -- Add word frequency normalization.
        C.addTFIDF(torch.data(x), torch.data(xwt), sz, self.opt.dictTFIDFPow,
            torch.data(self.dict.index_to_freq))
    else
        xwt[1]:copy(x)
        xwt[2]:fill(1 / math.sqrt(sz))
    end
    return xnew
end

function data:get_shared()
    local shared = {}
    shared.x = {}
    shared.y = {}
    shared.c = {}
    shared.ci = {}
    shared.sf = {}
    shared.st = {}
    shared.mem_hash = {}
    for i = 1, #self.examples_x do
        shared.x[i] = self.examples_x[i]:get_shared()
        shared.y[i] = self.examples_y[i]:get_shared()
        if self.examples_st[i] ~= nil then
            shared.st[i] = self.examples_st[i]:get_shared()
        end
        if self.examples_sf[i] ~= nil then
            shared.sf[i] = self.examples_sf[i]:get_shared()
        end
    end
    for i = 1, #self.examples_c do
        shared.c[i] = self.examples_c[i]:get_shared()
        shared.ci[i] = thread_utils.get_shared_ptr(self.examples_ci[i],
                                                   'FloatStorage*')
    end
    if #self.mem_hash > 0 then
        for i = 1, 3 do
            shared.mem_hash[i] = self.mem_hash[i]:get_shared()
        end
        if #self.mem_hash > 3 then
            shared.mem_hash[4] =
                thread_utils.get_shared_ptr(self.mem_hash[4], 'DoubleStorage*')
        end
    end
    return shared
end

function data:Tensor(...)
    return torch.FloatTensor(...)
end

function data:create_data(filename, shared_data, options, dict)
    local new_data = {}
    setmetatable(new_data, { __index = self })
    if options == nil then
        error('missing options')
    end
    new_data.opt = options

    new_data.filename = filename
    new_data.load_from_file = false
    if shared_data == nil or new_data.opt.threadsShareData == false then
        local files = pl.utils.split(filename, ',')
        -- Load data via preprocessed torch files.
        new_data.examples_x = {}
        new_data.examples_y = {}
        new_data.examples_st = {}
        new_data.examples_sf = {}
        new_data.examples_c = {}
        new_data.examples_ci = {}
        for i = 1, #files do
            new_data.examples_x[i] = VectorArray:load(files[i] .. '.x')
            new_data.examples_y[i] = VectorArray:load(files[i] .. '.y')
            local fst = io.open(files[i] .. '.st')
            if fst then
                fst:close()
                new_data.examples_st[i] = VectorArray:load(files[i] .. '.st')
            end
            local fsf = io.open(files[i] .. '.sf')
            if fsf then
                fsf:close()
                new_data.examples_sf[i] =
                    VectorArray:load(files[i] .. '.sf')
            end
            if options.useCandidateLabels
            or options.testWithCandidateLabels then
                local fc = io.open(files[i] .. '.c')
                local fci = io.open(files[i] .. '.ci')
                if fc and fci then
                    fc:close()
                    fci:close()
                    new_data.examples_c[i] = VectorArray:load(files[i] .. '.c')
                    new_data.examples_ci[i] = torch.load(files[i] .. '.ci')
               end
            end
            print('[loaded examples[' .. i .. ']:'
                      .. new_data.examples_x[i]:size() .. ']')
        end

        local hashfile = options.memHashFile
        new_data.mem_hash = {}
        if hashfile ~= nil then
            print('[loading mem hash:' .. hashfile .. ']')
            if options.dictUseUNKHashLookup then
                new_data.mem_hash[1] = torch.load(hashfile .. '.facts1')
                new_data.mem_hash[2] = torch.load(hashfile .. '.facts2')
                new_data.mem_hash[3] = torch.load(hashfile .. '.facts_hash')
                new_data.mem_hash[4] = torch.load(hashfile .. '.facts_ind')
            else
                new_data.mem_hash[1] =
                    VectorArray:load(hashfile .. '.facts1_va')
                new_data.mem_hash[2] =
                    VectorArray:load(hashfile .. '.facts2_va')
                new_data.mem_hash[3] =
                    VectorArray:load(hashfile .. '.facts_hash_va')
                new_data.mem_hash[4] = torch.load(hashfile .. '.facts_ind')
            end
        end
    else
        new_data.examples_x = {}
        new_data.examples_y = {}
        new_data.examples_st = {}
        new_data.examples_c = {}
        new_data.examples_ci = {}
        new_data.examples_sf = {}
        for i = 1, #shared_data.x do
            new_data.examples_x[i] = VectorArray:new_shared(shared_data.x[i])
            new_data.examples_y[i] = VectorArray:new_shared(shared_data.y[i])
            if shared_data.st[i] ~= nil then
                new_data.examples_st[i] =
                    VectorArray:new_shared(shared_data.st[i])
            end
            if shared_data.sf[i] ~= nil then
                new_data.examples_sf[i] =
                    VectorArray:new_shared(shared_data.sf[i])
            end
            if (options.useCandidateLabels or options.testWithCandidateLabels)
            and shared_data.c[i] ~= nil  then
                new_data.examples_c[i] = VectorArray:new_shared(
                    shared_data.c[i])
                new_data.examples_ci[i] = thread_utils.create_from_shared_ptr(
                    shared_data.ci[i], 'FloatStorage*')
            end
        end
        new_data.mem_hash = {}
        if options.memHashFile ~= nil then
            if options.dictUseUNKHashLookup then
                error('dictUseUNKHash cannot be shared')
            end
            new_data.mem_hash[1] =
                VectorArray:new_shared(shared_data.mem_hash[1])
            new_data.mem_hash[2] =
                VectorArray:new_shared(shared_data.mem_hash[2])
            new_data.mem_hash[3] =
                VectorArray:new_shared(shared_data.mem_hash[3])
            new_data.mem_hash[4] = thread_utils.create_from_shared_ptr(
                shared_data.mem_hash[4], 'DoubleStorage*')
        end
    end

    if dict ~= nil then
        local dictClass = require(options.dictClass)
        if dict.shared_table then
            new_data.dict = dictClass:create(options, dict)
        else
            new_data.dict = dict
        end
    end

    new_data:reset()

    -- Hashing stuff for long-term memories.
    if options.memHashFile ~= nil then
        local hash = hashlib:new(options, dict)
        if options.dictUseUNKHashLookup then
            hash.facts1 = new_data.mem_hash[1]
            hash.facts2 = new_data.mem_hash[2]
            hash.facts_hash = new_data.mem_hash[3]
            hash.facts_ind = new_data.mem_hash[4]
        else
            hash.facts1_va = new_data.mem_hash[1]
            hash.facts2_va = new_data.mem_hash[2]
            hash.facts_hash_va = new_data.mem_hash[3]
            hash.facts_ind = new_data.mem_hash[4]
        end
        new_data.hash = hash
    end

    return new_data
end

return data
