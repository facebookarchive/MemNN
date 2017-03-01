-- Copyright 2004-present Facebook. All Rights Reserved.
-- MemNN model that expects memories structured as {key, value}.
require('torch')
require('nn')
require('sys')
require('math')
require('os')
require('nngraph')
local memNN = require('library.memnn_model')
local VectorArray = require('library.vector_array')
require('library.WeightedLookupTableSkinny')
require('library.PositionalEncoder')
local kvmemNN = {}
setmetatable(kvmemNN, { __index = memNN })

kvmemNN.modelClass = 'library.kvmemnn_model'

function kvmemNN:create_nngraph()
    if self.opt.trainWithSupportingFacts then
        error('not supported')
    end

    local mlp
    local qemb = self:create_query_embedding()
    local key_mems_emb1in = self:create_memory_embedding()
    local val_mems_emb1in = self:create_memory_embedding()

    if self.opt.doNotUpdateEmbeddings then
        -- we erase the updateParameters function of the lookuptable
        qemb:get(1).updateParameters = function (...) return; end
        key_mems_emb1in:get(1).updateParameters = function (...) return; end
        val_mems_emb1in:get(1).updateParameters = function (...) return; end
    end

    -- NNgraph implementation
    local qin_n = nn.Identity()()
    local key_memin_n = nn.Identity()()
    local val_memin_n = nn.Identity()()
    local qemb_n = qemb(qin_n)
    local key_mems_emb1in_n = key_mems_emb1in(key_memin_n)
    local val_mems_emb1in_n = val_mems_emb1in(val_memin_n)

    -- Softmax over key memories using key memories
    local sm_mem = nn.Sequential()
    sm_mem:add(nn.MM(false, true))
    sm_mem:add(nn.View(-1):setNumInputDims(2))
    sm_mem:add(nn.SoftMax())
    sm_mem:add(nn.View(1, -1):setNumInputDims(1))
    self.sm_mem = sm_mem
    local sm_mem_n = sm_mem({qemb_n, key_mems_emb1in_n})
    -- make q2
    local q2 = nn.Sequential()
    q2:add(nn.MM(false, false))
    local q2_n

    q2_n = q2({sm_mem_n, val_mems_emb1in_n})

    -- Add q to q2
    local addqq2 = nn.Sequential()
    addqq2:add(nn.JoinTable(1))
    addqq2:add(nn.Sum(1))
    local addqq2_n = addqq2({qemb_n, q2_n})

    -- response module
    local resp = nn.Sequential()
    if self.opt.useCandidateLabels or self.opt.numNegSamples ~= nil then
        if self.opt.LTsharedWithResponse then
            self.rlin.weight = self.wt[1].weight:sub(1, self.dict.num_symbols)
        end
        self.labelLT = nn.Sequential()
        self.labelLT:add(nn.WeightedLookupTableSkinny(
                             self.dict.num_symbols,
                             self.opt.embeddingDim,
                             self.opt.LTMaxNorm,
                             self.opt.adagrad,
                             self.opt.embeddingLRScale,
                             self.mask))
        self.labelLT:get(1).weight = self.rlin.weight
        self.labelLT:add(nn.Sum(1))
        if self.opt.rotateAfterLookupTable then
            if self.opt.LTsharedWithResponse then
                self.labelLT:add(self.lin_rotate[#self.lin_rotate])
            else
                self.labelLT:add(self.lin_rotate[#self.lin_rotate - 1])
            end
        end
        resp:add(nn.Identity())
    else
        if self.opt.LTsharedWithResponse then
            -- Same weights for output response as for word embeddings.
            self.rlin.weight = self.wt[1].weight:sub(1, self.dict.num_symbols)
        end
        resp:add(self.rlin)
        if self.opt.rankLoss == false then
            resp:add(nn.LogSoftMax())
        end
    end

    -- Final bits.
    local resp_n
    local lastq_n = addqq2_n
    local sm_mems_n = { resp_n, sm_mem_n }
    for i = 2, self.opt.maxHops do
        local q_rotate = nn.Sequential()
        q_rotate:add(self.lin_rotate[i - 1])
        q_rotate:add(nn.View(1, self.opt.embeddingDim):setNumInputDims(1))
        local qs_rotate_n = q_rotate(lastq_n)
        local sm_mem_i = sm_mem:clone()
        sm_mems_n[i + 1] = sm_mem_i({qs_rotate_n, key_mems_emb1in_n})
        local qs_n = q2:clone()({sm_mems_n[i + 1], val_mems_emb1in_n})
        lastq_n = addqq2:clone()({qs_rotate_n, qs_n})
    end
    if not self.opt.rotateBeforeResponse then
        resp_n = resp(lastq_n)
    else
        local final_rotate = self.lin_rotate[self.opt.maxHops]
        local final_rotate_n = final_rotate(lastq_n)
        resp_n = resp(final_rotate_n)
    end

    mlp = nn.gModule({qin_n, key_memin_n, val_memin_n}, { resp_n })
    return mlp
end

function kvmemNN:build_input(qx, memx)
    local ret = {qx, memx[1], memx[2]}
    if self.opt.usePE then
        local q_len = torch.DoubleTensor(1):fill(qx:size(1))
        ret[1] = {qx, q_len}
    end
    return ret
end

function kvmemNN:build_memory_vectors(ex)
    local maxLen = self.opt.sentenceSize
    local empty = true
    local clipped = false
    local function add_memory(m, mem_type, mem, fall_back_m)
        empty = false

        if mem.mems == self.opt.memSize then
            -- Cannot add more memories! Have to clip .. :/
            clipped = true
            return
        end

        -- replace null memories (mostly for empty values)
        if m:size(1) == 1 and m[1][1] == self.NULL.x[1] and fall_back_m then
            m = fall_back_m
        end

        local sz = m:size(1)
        if mem.memBuffer:size(1) < sz + 2 then
            mem.memBuffer:resize(sz * 2 + 2, 2)
        end
        local mx = mem.memBuffer:sub(1, sz)
        self:fill_memory_slot(mx, m)

        local timLabel
        if mem_type == 'hx' or mem_type == 'hy' then
            if self.opt.useMemHTimeFeatures then
                mem.htim = mem.htim + math.random(self.opt.timeVariance)
                timLabel = mem.htim
            end
        elseif self.opt.useTimeFeatures then
            mem.tim = mem.tim + math.random(self.opt.timeVariance)
            timLabel = mem.tim
        end

        if timLabel ~= nil or self.opt.useMemLabelFeatures then
            local feats = 0
            if timLabel ~= nil then feats = feats + 1 end
            if self.opt.useMemLabelFeatures then feats = feats + 1 end
            mx = mem.memBuffer:sub(1, sz + feats)

            local idx = 0
            if timLabel ~= nil then
                idx = idx + 1
                mx[sz + idx][1] = self.dictSz - timLabel
                mx[sz + idx][2] = 1
            end
            if self.opt.useMemLabelFeatures then
                idx = idx + 1
                mx[sz + idx][1] = self.memLabel[mem_type]
                mx[sz + idx][2] = 1
            end
        end
        mem.memx:add(mx)
        mem.mems = mem.mems + 1
    end

    if self.memBufferKey == nil then
        -- maxLen + feats is nonbinding--code below grows buffer when needed
        self.memBufferKey = torch.Tensor(maxLen * 2 + 3, 2)
        self.memBufferVal = torch.Tensor(maxLen * 2 + 3, 2)
        self.memBufferQuery = torch.Tensor(maxLen * 2 + 3, 2)
    end
    if self.key_memx == nil then
        -- memSize * maxLen is nonbinding--vecarrays auto-grow when needed
        local sz = self.opt.memSize * maxLen
        self.key_memx = VectorArray:new(sz, self.opt.memSize, 2)
        self.val_memx = VectorArray:new(sz, self.opt.memSize, 2)
        self.query_memx = VectorArray:new(sz, self.opt.memSize, 2)
    end
    self.key_memx:clear()
    self.val_memx:clear()
    self.query_memx:clear()

    -- key and values have their own time reference which are synced
    local function init_mem(vec, buffer)
        local t = {}
        t.memx = vec
        t.memBuffer = buffer
        t.mems = 0
        t.tim = -1
        t.htim = -1
        return t
    end
    local key_mem = init_mem(self.key_memx, self.memBufferKey)
    local val_mem = init_mem(self.val_memx, self.memBufferVal)
    local query_mem = init_mem(self.query_memx, self.memBufferQuery)

    for i = #ex.memx, 1, -1 do
        add_memory(ex.memx[i], 'x', key_mem)
        if ex.memy and ex.memy[i] then
            add_memory(ex.memy[i], 'y', val_mem, ex.memx[i])
        else
            add_memory(ex.memx[i], 'x', val_mem, ex.memx[i])
        end
    end
    -- Add hashed deep memories that might be relevant.
    for i = 1, #ex.memhx do
        add_memory(ex.memhx[i], 'hx', key_mem)
        if ex.memhy and ex.memhy[i] then
            add_memory(ex.memhy[i], 'hy', val_mem, ex.memhx[i])
        else
            add_memory(ex.memhx[i], 'hx', val_mem)
        end
    end
    if empty then
        -- No memories found: Make a dummy memory so we have at least one!
        local null = torch.Tensor(1, 2):fill(0)
        null[1][1] = self.NULL.x[1]
        key_mem.memx:add(null)
        val_mem.memx:add(null)
    end

    -- Useful to store this to extract it e.g. at evaluation time.
    self.lastMemClipped = clipped
    key_mem.memx:clip()
    val_mem.memx:clip()
    local memx_return = {{key_mem.memx.data, key_mem.memx.len},
                         {val_mem.memx.data, val_mem.memx.len},
                         {query_mem.memx.data, query_mem.memx.len}}
    return memx_return, clipped, empty
end

return kvmemNN
