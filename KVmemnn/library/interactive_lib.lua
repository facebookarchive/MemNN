-- Copyright 2004-present Facebook. All Rights Reserved.
--------------------------------------------------------------------------------
local tds = require('tds')
local util = require('library.util')
local stringlib = require('string')

local inter = {}

function inter:create(mlp, data)
    local new_inter = {}
    setmetatable(new_inter, { __index = self })
    new_inter.mlp = mlp
    new_inter.data = data
    local parserlib =
       require('library.parse')
    new_inter.parser = parserlib:new(mlp.opt)
    new_inter.opt = mlp.opt
    local evallib = require(mlp.opt.evalClass)
    new_inter.eval = evallib:create(mlp, mlp.opt)
    return new_inter
end

function inter:predict(ex, do_print)
    self.eval:eval_one_example(ex, do_print)
end

function inter:print_example(ex)
    print "---- "
    if not ex.dataFast then
        for i = 1, #ex.memhx do
            print("MEMHX" .. i .. ":"
                      .. self.mlp.dict:vector_to_text(ex.memhx[i])
                      .. "\t\t | MEMHY" .. i .. ":"
                      .. self.mlp.dict:vector_to_text(ex.memhy[i]))
        end
        for i = 1, #ex.memx do
            print("MEMX" .. i .. ":"
                      .. self.mlp.dict:vector_to_text(ex.memx[i])
                      .. "\t\t | MEMY" .. i .. ":"
                      .. self.mlp.dict:vector_to_text(ex.memy[i]))
        end
        print "----"
        print("INP:" .. self.mlp.dict:vector_to_text(self.data:get_input(ex)))
        if ex[2] ~= nil then
            print("OUT:" .. self.mlp.dict:vector_to_text(ex[2]))
        end
    else
        local mem = ex[1][2]
        local cnt = 1
        local pos = 1
        for i = 1, mem:size(1), self.mlp.opt.sentenceSize do
            local s = mem:sub(pos, pos + self.mlp.opt.sentenceSize - 1)
            local str = self.mlp.dict:vector_to_text(s)
            str = str:gsub("<NULL> ", "")
            print("MEM" .. cnt .. ":" .. str)
            cnt = cnt + 1
            pos = pos + self.mlp.opt.sentenceSize
        end
        if self.mlp.lastMemClipped then
            print("[WARNING: memory was clipped]")
        end
        print "----"
        print("INP:" .. self.mlp.dict:vector_to_text(ex[1][1]))
        if ex[2] ~= nil then
            print("OUT:" .. self.mlp.dict:vector_to_text(ex[2]))
        end
    end
    if #ex.cands > 0 then
        local c = ''
        for i = 1, #ex.cands do
            c = c .. self.mlp.dict:v2t(ex.cands[i]) ..  "|"
        end
        c = c:sub(1, -2)
        print("CANDS:" .. c)
    end
end

function inter:next_example(data, do_print)
    if do_print == nil then do_print = true; end
    if data == nil then
        print "No data provided."
        return
    end
    local ex = data:get_next_example()
    if do_print ~= false then self:print_example(ex); end
    local metrics = self:predict(ex, do_print)
    if do_print ~= false then print(""); end
    return ex, metrics
end

function inter:next_example_with_candidates(data, do_print)
    local ex, metrics
    while true do
        ex, metrics = self:next_example(data, do_print)
        if #ex.cands > 0 then
            break
        end
    end
    return ex, metrics
end

function inter:one_random_example(data, do_print)
    if data == nil then
        print "No data provided."
        return
    end
    if do_print == nil then do_print = true; end
    local ex = data:get_random_example()
    if do_print ~= false then self:print_example(ex); end
    local metrics = self:predict(ex, do_print)
    if do_print ~= false then print(""); end
    return ex, metrics
end

function inter:get_negative(ex, data)
    if data == nil then data = g_train_data; end
    local y = data:get_negative_label(ex)
    print("NEG:" .. self.mlp.dict:v2t(y))
    return y
end

function inter:eval(data, num)
    local evallib =
        require(self.opt.evalClass)
    local eval = evallib:create(self.mlp, self.mlp.opt)
    eval:eval(data, num)
end

function inter:run(s, s2)
    print ""
    local x = self.parser:parse_test_time(s, self.mlp.dict)
    local ex = {}
    ex[1] = x
    ex[2] = x
    ex.memhx = {}
    ex.memyx = {}
    ex.cands = {}
    if self.data.hash then
        local c1, c2 = self.data.hash:get_candidate_set(ex[1])
        for i = 1, #c1 do
            c1[i] = self.data:addTFIDF(c1[i])
            c2[i] = self.data:addTFIDF(c2[i], true)
        end
        if #c2 > 0 and self.opt.maxHashSizeSortedByTFIDF > 0 then
            c1, c2 = self.data.hash:sortByTFIDF(x, c1, c2)
        end
        ex.memhx = c1
        ex.memhy = c2
    end
    ex.memx = {}
    if s2 ~= nil then
        ex.memx[1] = self.parser:parse_test_time(s2, self.mlp.dict)
    end
    self:print_example(ex)
    print "----"
    self:predict(ex, true)
    print ""
    return ex
end

-- sequentially perform predictions and add previous prediction to the
-- memory, the way a dialog system would do.
-- Setting clearHistory = true erases the history and starts over.
function inter:run_dialog(s, clearHistory, debugMode)

    print ""
    local x = self.parser:parse_test_time(s, self.mlp.dict)
    local ex = {}
    ex[1] = x
    ex[2] = x
    ex.memhx = {}
    ex.memyx = {}
    ex.cands = {}

    local useHashedCandidates = false
    if self.data.hash then
        local c1, c2 = self.data.hash:get_candidate_set(ex[1])
        for i = 1, #c1 do
            c1[i] = self.data:addTFIDF(c1[i])
            c2[i] = self.data:addTFIDF(c2[i], true)
        end
        ex.cands = c2
        if #ex.cands > 0 then
            -- we use hashed reponses as candidates
            useHashedCandidates = true
            self.updateLabelsEmbeddings = true
        end
        if not self.notUseHashInMemory then
            if #c2 > 0 and self.opt.maxHashSizeSortedByTFIDF > 0 then
                c1, c2 = self.data.hash:sortByTFIDF(x, c1, c2)
            end
            ex.memhx = c1
            ex.memhy = c2
        end
    end

    if not useHashedCandidates then
        -- we use a sample of 10k random responses from the train set
        -- as candidates
        if not self.train_cands then
            self.train_cands = tds.hash()
            while #self.train_cands < 10000 do
                local candi = self.data:get_random_label_document()
                if candi then
                    candi = self.data:resolveUNK(candi)
                    candi = self.data:addTFIDF(candi)
                    self.train_cands[#self.train_cands + 1] = candi
                end
            end
            self.updateLabelsEmbeddings = true
        end
        ex.cands = self.train_cands
    end

    if not self.history or clearHistory then
        print("[Cleared dialog history]")
        self.history = {}
        self.history.memx = {}
        self.history.memy = {}
    end
    ex.memx = self.history.memx
    ex.memy = self.history.memy

    -- fast predict with cached candidate embeddings, those are
    -- recomputed if the set of candidates changed
    local y, cands, cscores =
        self.mlp:fast_predict_with_candidates(ex,
                                              self.updateLabelsEmbeddings)

    for i = 1, #ex.memx do
        print("User[" .. i .. "]:\t" ..
                  self.mlp.dict:vector_to_text(ex.memx[i]))
        print("Syst[" .. i .. "]:\t" ..
                  self.mlp.dict:vector_to_text(ex.memy[i]))
    end
    print("User[?]:\t" .. self.mlp.dict:vector_to_text(self.data:get_input(ex)))

    -- we set up the flag for updating candidate embeddings at the next round
    if useHashedCandidates then
        self.updateLabelsEmbeddings = true
        print("[" .. #ex.cands .. " candidates hashed based on query]")
    else
        self.updateLabelsEmbeddings = false
        print("[" .. #ex.cands .. " candidates sampled at random]")
    end
    print("Syst[!]:\t" .. self.mlp.dict:v2t(y))

    if debugMode then
        for i = 1, math.min(10, cscores:size(1)) do
            local name = self.mlp.dict:v2t(cands[i])
            local score = util.shortFloat(cscores[i])
            print("PRED" .. i .. ": "
                      .. name .. '\t[' .. score  .. ']')
        end
        print "----------"
    end
    print ""
    table.insert(self.history.memx, x)
    table.insert(self.history.memy, y)
    return y
end

-- Prints the details of the softmax activations during the last call
-- to predict: useful to visualize hops.
function inter:print_detailed_prediction(maxNbMemToPrint)

    maxNbMemToPrint = maxNbMemToPrint or 20
    local mlpi = self.mlp

    -- to uncover the memLabelFeatures in the memory
    local memLabelFeat = {}
    if mlpi.opt.useMemLabelFeatures then
        for i, k in pairs(mlpi.memLabel) do
            memLabelFeat[k] = '<memFeat ' .. i:upper() .. '>'
        end
    end

    -- rewriting of dict:v2t that do not print <NULL> and explicts
    -- time and memLabel features
    local function pretty_print(s)
        local t = ""
        local lt = 0
        local maxlt = 20
        for i = 1, s:size(1) do
            local ind
            if s:dim() == 1 then
                ind = s[i]
            else
                ind = s[i][1]
            end
            if ind == mlpi.NULL.x[1] then ind = nil end
            if ind then
                local w = mlpi.dict:index_to_symbol(ind)
                if w == nil then
                    if memLabelFeat[ind] then
                        w = memLabelFeat[ind]
                    else
                        w = '<timeFeat #' .. mlpi.dictSz - ind .. '>'
                    end
                    t = t .. w .. " "
                else
                    if lt <= maxlt then
                        t = t .. w .. " "
                    elseif lt == maxlt + 1 then
                        t = t .. "[...] "
                    end
                    lt = lt + 1
                end
            end
        end
        if t:len() > 1 then t = t:sub(1, -2) end
        return t
    end

    -- we read everything from the last forward of the memNN
    local modules = mlpi.mlp.modules
    local input = modules[1].output
    if input:dim() == 0 then
        print('[an example needs be forwarded before printing]')
        return;
    end

    local memory
    if mlpi.opt.variableMemories then
        local memory0 = modules[3].output[1]
        local lens = modules[3].output[2]
        memory = torch.FloatTensor(lens:size(1),
                                   lens:max()):fill(mlpi.NULL.x[1])
        local offset = 1
        for j = 1, lens:size(1) do
            memory[j]:narrow(1, 1, lens[j]):copy(memory0:narrow(
                                                     1, offset, lens[j]):t()[1])
            offset = offset + lens[j]
        end
    else
        local memory0 = modules[3].output:clone()
        -- reformatting of the memory
        memory = nn.View(memory0:size(1) / mlpi.opt.sentenceSize,
                         mlpi.opt.sentenceSize,
                             -1):setNumInputDims(2):forward(memory0)
    end
    -- convert memories into text
    local memtext = {}
    local maxmemlen = 0
    for j = 1, memory:size(1) do
        local mjt = pretty_print(memory[j])
        if #mjt > maxmemlen then maxmemlen = #mjt end
        table.insert(memtext, 'M #'.. j .. ': ' .. mjt)
    end
    -- pad memories for nicer printing
    for j, k in pairs(memtext) do
        memtext[j] = k .. stringlib.rep(' ', maxmemlen - #k)
    end
    local header = 'MEMORY:' .. stringlib.rep(' ', maxmemlen - 7)

    -- we record the softmax activations for each hop
    local hop1sm = torch.FloatTensor(#memtext)
    local nhops = 0
    for i = 1, #modules do
        if modules[i].modules and #modules[i].modules >= 3 and
        modules[i].modules[3].__typename == 'nn.SoftMax' then
            nhops = nhops + 1
            header = header .. '\tHOP' .. nhops
            if #memtext > 0 then
                local sm_prob = modules[i].modules[3].output
                for p = 1, sm_prob:size(1) do
                    hop1sm[p] = sm_prob[p]
                    memtext[p] = memtext[p] .. '\t' ..
                        string.format('%.03f', sm_prob[p]):sub(2)
                end
            end
        end
    end
    local val = torch.sort(hop1sm, true)
    local probThreshold = -1
    if #memtext > maxNbMemToPrint then
        probThreshold = val[maxNbMemToPrint]
    end

    -- save print the whole thing
    local toprint = 'QUERY: ' .. pretty_print(input) .. '\n'
    toprint = toprint .. header .. '\n'
    for p = #memtext, 1, -1 do
        if hop1sm[p] >= probThreshold then
            toprint = toprint .. memtext[p] .. '\n'
        end
    end
    print(toprint)
    return toprint
end

return inter
