-- Copyright 2004-present Facebook. All Rights Reserved.

require('torch')
require('nn')
require('sys')
require('math')
require('os')
require('nngraph')
local tablex = require('pl.tablex')
local thread_utils = require('library.thread_utils')
local util = require('library.util')
local pl = require('pl.import_into')()
local baseNN = require('library.base_model')
local VectorArray = require('library.vector_array')
require('library.PositionalEncoder')
require('library.SumVecarr')
require('library.WeightedLookupTableSkinny')

local memNN = {}
setmetatable(memNN, { __index = baseNN })

function memNN:add_cmdline_options(cmd)
    baseNN:add_cmdline_options(cmd)

    -- Architecture options.
    cmd:option('embeddingDim', 20, 'embedding dimension', 'model')
    cmd:option('maxHops', 1, 'max hops', 'model')
    cmd:option('rotateHasBias', true, '', 'model')
    cmd:option('rotateBeforeResponse', false,
               'more rotations can help', 'model')
    cmd:option('rotateAfterLookupTable', false,
               'can help to scale embeddings initialized from another source',
               'model')
    cmd:option('LTsharedWithResponse', true,
               'single lookuptable or not', 'model')
    cmd:option('metric', 'dot', 'dot or cosine, currently only supported'
               .. 'for numNegSamples>0 and rankLoss=true', 'model')
    cmd:option('usePE', false, 'use positional encoding layer')

    -- Loss and Training options
    cmd:option('learningRate', 0.01, 'learning rate', 'model')
    cmd:option('initWeights', 0.01, 'mlp weights are scaled by this', 'model')
    cmd:option('LTMaxNorm', false,
               'max norm of an LT embedding vector', 'model', true)
    cmd:option('margin', 0.1, 'margin, only used for ranking loss', 'model')
    cmd:reset_default('numThreads', 30)
    cmd:reset_default('maxTrainTime', 1000)
    cmd:reset_default('saveEveryNSecs', nil)
    cmd:reset_default('logEveryNSecs', 1)
    cmd:reset_default('logTrainingFile', false)
    cmd:option('rankLoss', false, 'use rank loss', 'model')
    cmd:option('predictTopK', 1, 'return K predictions', 'model')
    cmd:option('useCandidateLabels', false, 'use a subset of words as ' ..
                   'candidate set', 'model')
    cmd:option('onlyTrainWithExWithCandidates', false,
               'Do not train on examples where #ex.cands==0', 'model')
    cmd:option('useMemHYasCandidates', false,
               'rank ys found in hash as candidates', 'data')
    cmd:option('useMemYasCandidates', false,
               'rank ys found in mem as candidates', 'data')
    cmd:option('cacheViolators', false, 'used to find negatives fast', 'model')
    cmd:option('cacheViolatorsSz', 10,
               'size used to find negatives fast', 'model')
    cmd:option('numNegSamples', nil, 'neg sampling candidates', 'model')
    cmd:option('sampleCands', 1, 'ratio of times sample real cands', 'model')
    cmd:option('labelSize', 1, 'maximum label size (in ranking case)', 'model')
    cmd:option('computeResponseLoss', false,
               'compute response loss while sampling, this is slow and '
               .. 'mostly used for debugging purposes', 'model')
    cmd:option('stopTrainZeroIts', -1,
               'stop training if zero error is reported during training '
               .. 'for more than this number of iterations in a row (-1 '
                   .. 'cancels this option, this is the default)', 'model')
    cmd:option('lrDecayDivisor', 1, 'divide lr by this every N secs', 'model')
    cmd:option('lrDecaySecs', 250, 'lr decay every N secs', 'model')
    cmd:option('lrDecayFinal', 0.0001, 'minimum possible lr', 'model')
    cmd:option('trainWithSupportingFacts', false,
               'strong supervision', 'model', true)
    cmd:option('gradClipMaxNorm', nil,
               'Clip the gradients if set', 'model', true)
    cmd:option('initFromNoMemNNModel', nil,
               'if given, loads the nonmemnn_model file and initializes the '
                   .. 'embedding weights with those weights', 'model', true)

    -- Data and Memory options
    cmd:option('dictUNKFile', nil, 'Set of unknown (hashed) dictionary words',
               'model')
    cmd:option('memSize', 200, 'max mem size', 'model')
    cmd:option('sentenceSize', 20, 'sentence size', 'model')
    cmd:option('useTimeFeatures', false, 'use time features', 'model')
    cmd:option('useMemHTimeFeatures', false, 'use memhx/y time features',
               'model')
    cmd:option('timeVariance', 1, 'add variance to time step', 'model')
    cmd:option('useMemy', false, 'if true, then also add memhy to the memory '
                    .. 'vectors. Needs to be used with time features on s.t. '
                    .. 'it knows which answer is for which question', 'model')
    cmd:option('useMemHy', false, 'use memhy', 'model')
    cmd:option('useMemLabelFeatures', false, 'use mem label features', 'model')
    cmd:option('dropLargeEx', false, 'drop examples with large memory '
                    .. '(otherwise keep but truncate memory)', 'model')
    cmd:option('dropNoMem', false, 'drop examples with no memory', 'model')
    cmd:option('rankDictionary', false,
               'if true, final output ranks dictionary of symbols', 'model')
end

memNN.modelClass = 'library.memnn_model'

function memNN:num_features()
    local tfeats = 0
    if self.opt.useMemLabelFeatures then
        tfeats = tfeats + 4
    end
    if self.opt.useTimeFeatures or self.opt.useMemHTimeFeatures then
        tfeats = tfeats + self.opt.timeVariance * self.opt.memSize
    end
    return self.dict.num_symbols + tfeats
end

function memNN:create_modules(num_rotations)
    num_rotations = num_rotations or self.opt.maxHops
    -- dictionary lookup table.
    local wt = {}
    for i = 1, 1 do
        wt[i] = nn.WeightedLookupTableSkinny(self.dictSz, self.opt.embeddingDim)
        wt[i]:reset(self.opt.initWeights)
    end
    self.wt = wt
    self.lin_rotate = {}
    local bias = self.opt.rotateHasBias or self.opt.rotateHasBias == nil
    for i = 1, num_rotations do
        self.lin_rotate[i] = nn.Linear(self.opt.embeddingDim,
                                       self.opt.embeddingDim,
                                       bias)
        self.lin_rotate[i]:reset(self.opt.initWeights)
    end
    if self.opt.rotateAfterLookupTable then
        local nbExtraRotations = 1
        if (self.opt.useCandidateLabels or self.opt.numNegSamples ~= nil)
        and not self.opt.LTsharedWithResponse then
            nbExtraRotations = 2
        end
        for nidx = num_rotations + 1, num_rotations + nbExtraRotations do
            self.lin_rotate[nidx] =  nn.Linear(self.opt.embeddingDim,
                                               self.opt.embeddingDim,
                                               bias)
            self.lin_rotate[nidx].weight = torch.eye(self.opt.embeddingDim)
        end
    end
    local rlin = nn.Linear(self.opt.embeddingDim, self.dict.num_symbols, false)
    rlin:reset(self.opt.initWeights)
    self.rlin = rlin
end

function memNN:clone_mlp(shared_mlp, options)
    local new_model = {}
    setmetatable(new_model, { __index = self })
    new_model.opt = options
    local mlp = nn.Sequential()
    new_model.mlp = mlp
    torch.setdefaulttensortype("torch.FloatTensor")
    -- Create an MLP from shared weights for multithreading.
    new_model.dict = shared_mlp.dict
    local dictClass = require(options.dictClass)
    setmetatable(new_model.dict, {__index = dictClass})
    new_model.dictSz = new_model:num_features()
    if options.cacheViolators then
        new_model.labelViolators = torch.Tensor(new_model.dict.num_symbols,
                                                options.cacheViolatorsSz)
        thread_utils.reset_to_shared_ptr(
            new_model.labelViolators,
            shared_mlp.labelViolators, shared_mlp.storageType
        )
    end
    local mask
    if options.LTUpdateMaskFile then
        mask = torch.load(options.LTUpdateMaskFile)
        self.mask = mask
    end
    local wt = {}
    for i = 1, 1 do
        wt[i] = nn.WeightedLookupTableSkinny(new_model.dictSz,
                                             options.embeddingDim)
        thread_utils.reset_to_shared_ptr(
            wt[i].weight, shared_mlp.wt[i], shared_mlp.storageType
        )
    end
    new_model.wt = wt
    new_model.lin_rotate = {}
    local lin_rotate = {}
    local bias =
        new_model.opt.rotateHasBias == nil or new_model.opt.rotateHasBias
    for i = 1, #shared_mlp.lin_rotate_weight do
        lin_rotate[i] = nn.Linear(options.embeddingDim,
                                  options.embeddingDim,
                                  bias)
        if bias then
            thread_utils.reset_to_shared_ptr(
                lin_rotate[i].bias,
                shared_mlp.lin_rotate_bias[i], shared_mlp.storageType
            )
        end
        thread_utils.reset_to_shared_ptr(
            lin_rotate[i].weight,
            shared_mlp.lin_rotate_weight[i], shared_mlp.storageType
        )
        new_model.lin_rotate[i] = lin_rotate[i]
    end
    local rlin = nn.Linear(options.embeddingDim,
                           new_model.dict.num_symbols,
                           false)
    thread_utils.reset_to_shared_ptr(
        rlin.weight, shared_mlp.rlin_weight, shared_mlp.storageType
    )
    new_model.rlin = rlin
    new_model:reset()
    collectgarbage()
    return new_model
end

function memNN:create_mlp(options)
    local new_model = {}
    setmetatable(new_model, { __index = self })
    new_model.opt = options
    local mlp = nn.Sequential()
    new_model.mlp = mlp
    torch.setdefaulttensortype("torch.FloatTensor")
    print '[creating dict.]'
    local dictClass = require(options.dictClass)
    new_model.dict = dictClass:create(options)
    print '[creating mlp.]'
    new_model.dictSz = new_model:num_features()
    -- Cache of violators.
    if options.cacheViolators then
        new_model.labelViolators = torch.zeros(new_model.dict.num_symbols,
                                               options.cacheViolatorsSz)
    end
    if options.LTUpdateMaskFile then
        self.mask = torch.load(options.LTUpdateMaskFile)
    end
    new_model:create_modules()
    print '[created.]'
    if new_model.opt.initFromNoMemNNModel then
        new_model:initFromNoMemNNModel()
    end
    new_model:reset()
    collectgarbage()
    return new_model
end

function memNN:initFromNoMemNNModel()
    print("[initializing from nomemnn_model: "
              .. self.opt.initFromNoMemNNModel .. "]")
    local nomem_w = torch.load(self.opt.initFromNoMemNNModel)
    local nomem_opt = torch.load(self.opt.initFromNoMemNNModel .. '.opt')
    if nomem_opt.embeddingDim ~= self.opt.embeddingDim then
        error("different embeddingDim in nomemnn_model")
    end
    if nomem_opt.RHSDict == self.opt.LTsharedWithResponse then
        error("LTsharedWithResponse="
                  .. tostring(self.opt.LTsharedWithResponse)
                  .. ' vs. RHSDict=' .. tostring(nomem_opt.RHSDict)
                  .. " in nomemnn_model")
    end
    -- Copy input weights.
    local w1 = nomem_w:sub(1, self.dict.num_symbols)
    self.wt[1].weight:sub(1, self.dict.num_symbols):copy(w1)
    if nomem_opt.RHSDict then
        local w2 = nomem_w:sub(self.dict.num_symbols + 1,
                               self.dict.num_symbols * 2)
        self.rlin.weight:sub(1, self.dict.num_symbols):copy(w2)
    end
end

function memNN:freeze_weights()
    print('[freezing weights]')
    self.wt_tmp = {}
    for i = 1, #self.wt do
        self.wt_tmp[i]  = self.wt[i]
        self.wt[i] = self.wt_tmp[i]:clone()
    end
    self.lin_rotate_tmp = {}
    for j = 1, #self.lin_rotate do
        self.lin_rotate_tmp[j] = self.lin_rotate[j]
        self.lin_rotate[j] = self.lin_rotate_tmp[j]:clone()
    end
    self.rlin_tmp = self.rlin
    self.rlin = self.rlin_tmp:clone()
    -- we reset to ensure that the shared weights remain sync'ed
    self.mlp = self:create_nngraph()
end

function memNN:unfreeze_weights()
    print('[unfreezing weights]')
    if self.wt_tmp then
        for i = 1, #self.wt do
            self.wt[i] = self.wt_tmp[i]
            self.wt_tmp[i] = nil
        end
        self.wt_tmp = nil
    end
    if self.lin_rotate_tmp then
        for j = 1, #self.lin_rotate_tmp do
            self.lin_rotate[j] = self.lin_rotate_tmp[j]
            self.lin_rotate_tmp[j] = nil
        end
    end
    if self.rlin_tmp then
        self.rlin = self.rlin_tmp
        self.rlin_tmp = nil
    end
    -- we reset to ensure that the shared weights remain sync'ed
    self.mlp = self:create_nngraph()
    collectgarbage()
    collectgarbage()
end


-- Returns a dictionary with pointers to the parameters that should be shared
-- between threads
function memNN:get_shared()
    local shared = {}
    shared.storageType = 'THFloatStorage*'
    shared.wt = {}
    for i = 1, #self.wt do
        shared.wt[i] = thread_utils.get_shared_ptr(
            self.wt[i].weight, shared.storageType
        )
    end
    shared.lin_rotate_weight = {}
    shared.lin_rotate_bias = {}
    for i = 1, #self.lin_rotate do
        shared.lin_rotate_weight[i] = thread_utils.get_shared_ptr(
            self.lin_rotate[i].weight, shared.storageType
        )
        if self.lin_rotate[i].bias ~= nil then
            shared.lin_rotate_bias[i] = thread_utils.get_shared_ptr(
                self.lin_rotate[i].bias, shared.storageType
            )
        end
    end
    shared.rlin_weight = thread_utils.get_shared_ptr(
        self.rlin.weight, shared.storageType
    )
    shared.dict = self.dict:get_shared()
    if self.labelViolators ~= nil then
        shared.labelViolators = thread_utils.get_shared_ptr(
            self.labelViolators, shared.storageType
        )
    end
    return shared
end

function memNN:create_query_embedding()
    -- query embedding
    local qemb = nn.Sequential()
    if self.opt.usePE then
        local mlpP = nn.ParallelTable()
        mlpP:add(self.wt[1])
        mlpP:add(nn.Identity())
        qemb:add(mlpP)
        qemb:add(nn.PositionalEncoder(false))
    else
        qemb:add(self.wt[1])
    end
    qemb:add(nn.Sum(1))

    if self.opt.rotateAfterLookupTable then
        qemb:add(self.lin_rotate[#self.lin_rotate])
    end
    qemb:add(nn.View(1, self.opt.embeddingDim):setNumInputDims(1))
    self.qemb = qemb
    return qemb
end

function memNN:create_memory_embedding()
    -- Memory embedding.
    local mem_emb = {}

    for i = 1, #self.wt do
        mem_emb[i] = nn.Sequential()
        local parallel = nn.ParallelTable()
        parallel:add(self.wt[i])
        parallel:add(nn.Identity())
        mem_emb[i]:add(parallel)
        if self.opt.usePE then
            mem_emb[i]:add(nn.PositionalEncoder(true))
        end
        mem_emb[i]:add(nn.SumVecarr())
        if self.opt.rotateAfterLookupTable then
            mem_emb[i]:add(self.lin_rotate[#self.lin_rotate])
        end
    end

    self.mem_embz = mem_emb[1]
    -- 2nd version, don't share weights gradients.
    local mems_emb1in = mem_emb[1]:clone('weight','bias')
    return mems_emb1in
end

function memNN:create_nngraph()
    local mlp
    local qemb = self:create_query_embedding()
    local mems_emb1in = self:create_memory_embedding()

    -- NNgraph implementation
    local qin_n = nn.Identity()()
    local memin_n = nn.Identity()()
    local qemb_n = qemb(qin_n)
    local mems_emb1in_n = mems_emb1in(memin_n)

    -- Softmax
    local sm_mem = nn.Sequential()
    sm_mem:add(nn.MM(false, true))
    sm_mem:add(nn.View(-1):setNumInputDims(2))
    sm_mem:add(nn.SoftMax())
    sm_mem:add(nn.View(1, -1):setNumInputDims(1))
    self.sm_mem = sm_mem
    local sm_mem_n = sm_mem({qemb_n, mems_emb1in_n})

    -- make q2
    local q2 = nn.Sequential()
    q2:add(nn.MM(false, false))
    local q2_n = q2({sm_mem_n, mems_emb1in_n})

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
        self.labelLT:add(nn.WeightedLookupTableSkinny(self.dict.num_symbols,
                                                      self.opt.embeddingDim))
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
        if self.opt.doNotUpdateEmbeddings then
            -- we erase the updateParameters function of the lookuptable
            self.rlin.updateParameters = function (...) return; end
        end
        resp:add(self.rlin)
        if self.opt.rankLoss == false then
            resp:add(nn.LogSoftMax())
        end
    end

    -- Final bits.
    local resp_n
    -- Note: The special cases for 1 and 2 are there for legibility; the > 2
    -- hops case is general enough for all cases
    if self.opt.maxHops == 1 then
        local final_rotate_n
        if not self.opt.rotateBeforeResponse then
            resp_n = resp(addqq2_n)
        else
            local final_rotate = self.lin_rotate[1]
            final_rotate_n = final_rotate(addqq2_n)
            resp_n = resp(final_rotate_n)
        end
        nngraph.annotateNodes()
        if self.opt.trainWithSupportingFacts then
            mlp = nn.gModule({qin_n, memin_n},{ resp_n, sm_mem_n })
        else
            mlp = nn.gModule({qin_n, memin_n},{ resp_n })
        end
    elseif self.opt.maxHops == 2 then
        local q2_rotate = nn.Sequential()
        q2_rotate:add(self.lin_rotate[1])
        q2_rotate:add(nn.View(1, self.opt.embeddingDim):setNumInputDims(1))
        local q2_rotate_n = q2_rotate(addqq2_n)
        local sm_mem2_n = sm_mem:clone()({q2_rotate_n, mems_emb1in_n})
        local q3_n = q2:clone()({sm_mem2_n, mems_emb1in_n})
        local addq2q3_n = addqq2:clone()({q2_rotate_n, q3_n})
        local final_rotate_n
        if not self.opt.rotateBeforeResponse then
            resp_n = resp(addq2q3_n)
        else
            local final_rotate = self.lin_rotate[2]
            final_rotate_n = final_rotate(addq2q3_n)
            resp_n = resp(final_rotate_n)
        end
        nngraph.annotateNodes()
        if self.opt.trainWithSupportingFacts then
            mlp = nn.gModule(
                {qin_n, memin_n}, { resp_n, sm_mem_n, sm_mem2_n })
        else
            mlp = nn.gModule({qin_n, memin_n},{ resp_n })
        end
    else
        -- For > 2 hops.
        local lastq_n = addqq2_n
        local sm_mems_n = { resp_n, sm_mem_n }
        for i = 2, self.opt.maxHops do
            local q_rotate = nn.Sequential()
            q_rotate:add(self.lin_rotate[i - 1])
            q_rotate:add(nn.View(1, self.opt.embeddingDim):setNumInputDims(1))
            local qs_rotate_n = q_rotate(lastq_n)
            sm_mems_n[i + 1] = sm_mem:clone()({qs_rotate_n, mems_emb1in_n})
            local qs_n = q2:clone()({sm_mems_n[i + 1], mems_emb1in_n})
            lastq_n = addqq2:clone()({qs_rotate_n, qs_n})
        end
        if not self.opt.rotateBeforeResponse then
            resp_n = resp(lastq_n)
        else
            local final_rotate = self.lin_rotate[self.opt.maxHops]
            local final_rotate_n = final_rotate(lastq_n)
            resp_n = resp(final_rotate_n)
        end
        nngraph.annotateNodes()
        if self.opt.trainWithSupportingFacts then
            mlp = nn.gModule({qin_n, memin_n}, { sm_mems_n })
        else
            mlp = nn.gModule({qin_n, memin_n}, { resp_n })
        end
    end
    return mlp
end

function memNN:reset()
    baseNN.reset(self)
    self.dictSz = self:num_features()
    if self.opt.metric == 'cosine' then self.cosine = true; end

    self.mlp = self:create_nngraph()

    self.criterion = nn.ClassNLLCriterion()
    if self.opt.trainWithSupportingFacts then
        self.sf_criterion = {}
        for i = 1, self.opt.maxHops do
            self.sf_criterion[i] = nn.ClassNLLCriterion()
        end
    end
    self.gradNorm = -1
    if self.opt.gradClipMaxNorm ~= nil then
        local paramx, paramdx = self.mlp:getParameters()
        self.paramx = paramx
        self.paramdx = paramdx
    end

    -- Zero out the NULL symbol.
    for i = 1, #self.wt do
        self.wt[i].weight[self.NULL.x[1]]:zero()
    end

    -- Temporary variables we might need.
    self.inputEmbedding = self:Tensor(self.opt.embeddingDim):zero()
    self.posLabelEmbedding = self:Tensor(self.opt.embeddingDim):zero()
    self.negLabelEmbedding = self:Tensor(self.opt.embeddingDim):zero()
    self.temp = self:Tensor(self.opt.embeddingDim):zero()
    self.temp2 = self:Tensor(self.opt.embeddingDim):zero()
    self.memLabel = {}
    self.memLabel['x'] = self.dict.num_symbols + 1
    self.memLabel['y'] = self.dict.num_symbols + 2
    self.memLabel['hx'] = self.dict.num_symbols + 3
    self.memLabel['hy'] = self.dict.num_symbols + 4
    self:reset_prediction_cache()
end

local function file_exists(fname)
    local f = io.open(fname)
    if f == nil then
        return false
    else
        f:close()
        return true
    end
end

function memNN:load(fname)
    if self.opt.allowLoading == false then
        return false
    end
    if file_exists(fname .. '.wt') then
        -- Backwards compatible to loading old models.
        -- Remove this code when no old models are left around!!
        if self.dict == nil then
            local dictClass = require(self.opt.dictClass)
            self.dict = dictClass:create(self.opt)
        end
        print("[loading mlp:" .. fname .. "]")
        self.mlp = torch.load(fname)
        self.wt = torch.load(fname .. '.wt')
        if file_exists(fname .. '.linrot') then
            -- Backwards compatible to old models.
            self.lin_rotate = {}
            self.lin_rotate[1] = torch.load(fname .. '.linrot')
        end
        if file_exists(fname .. '.linrots') then
            self.lin_rotate = torch.load(fname .. '.linrots')
        end
        if file_exists(fname .. '.lv') then
            self.labelViolators = torch.load(fname .. '.lv')
        end
        self.rlin = torch.load(fname .. '.rlin')
    else
        if file_exists(fname) then
            if self.dict == nil then
                local dictClass = require(self.opt.dictClass)
                self.dict = dictClass:create(self.opt)
            end
            local all_model = torch.load(fname)
            for i, k in pairs(all_model) do
                -- Not clear if you want to wipe the options, or not.
                if i ~= 'opt' then
                    self[i] = k
                else
                    -- Compare to existing options and issue a warning
                    -- for each difference.
                    for key, value in pairs(k) do
                        if self.opt[key] ~= value then
                            print("WARNING: loading model with original "
                                      .. " opt[" .. key .. "]="
                                      .. tostring(value) .. " vs. "
                                      .. tostring(self.opt[key]))
                        end
                    end
                end
            end
        else
            return false
        end
    end
    if self.rlin.bias ~= nil then
        print("[oops.. rlin was broken in this model...fixing..]")
        torch.setdefaulttensortype("torch.FloatTensor")
        local tmp = self.rlin
        self.rlin = nn.Linear(tmp.weight:size(2), tmp.weight:size(1), false)
        self.rlin.weight:copy(tmp.weight)
        if (self.opt.rotateHasBias == false)
        and self.lin_rotate[1].bias ~= nil then
            print "[fixing lin_rotate biases as well.. god dammmm...]"
            for i = 1, #self.lin_rotate do
                local tmp = self.lin_rotate[i]
                self.lin_rotate[i] = nn.Linear(tmp.weight:size(2),
                                               tmp.weight:size(1),
                                               false)
                self.lin_rotate[i].weight:copy(tmp.weight)
            end
        end
    end

    -- We might need to reinitialize the mlp variables.
    self:reset()
    return true
end

function memNN:save_all(fname)
    if self.opt.allowSaving then
         print("[saving mlp: " .. fname .. "]")
         local all_model = {}
         all_model.wt = self.wt
         all_model.rlin = self.rlin
         all_model.lin_rotate = self.lin_rotate
         all_model.opt = self.opt
         if self.labelViolators then
             all_model.labelViolators = self.labelViolators
         end
         torch.save(fname .. '.tmp', all_model)
         if not os.rename(fname .. '.tmp', fname) then
            print('WARNING: renaming failed')
         end
         torch.save(fname .. '.opt', self.opt)
    end
    return true
end

function memNN:save(fname)
    error('should not get here')
    return true
end

--[[
data has format...
{
    Query Tensor (length_of_query X 2),
    {
        Tensor with all memories concatenated (total_len_of_mems X 2),
        Tensor specifying length of each memory ('memSize' X 1)
    }
}
--]]
function memNN:build_input(qx, memx)
    if self.opt.usePE then
        local q_len = torch.DoubleTensor(1):fill(qx:size(1))
        return {{qx, q_len}, memx}
    end
    return {qx, memx}
end

function memNN:scores_to_pred(cscores, cands)
    local response
    if cscores:dim() == 0 then
        -- Dummy response, no memories.
        response = torch.Tensor({{1,1}})
    else
        local _val, ind = cscores:max(1)
        response = cands[ind[1]]
    end
    if self.opt.rankDictionary then
        if not self.new_cscores then
            self.new_cscores = torch.Tensor(self.dict.num_symbols)
        end
        self.new_cscores:fill(-1000)
        for i = 1, #cands do
            local c = cands[i]
            local score = cscores[i]
            for j = 1, c:size(1) do
                local index = c[j][1]
                self.new_cscores[index] =
                    math.max(self.new_cscores[index], score)
            end
        end
        cands = self.dict:get_labels()
        cscores = self.new_cscores
    end
    return response, cands, cscores
end

function memNN:reset_prediction_cache()
    self.labelEmbeddings = nil
end

-- Predict labels given an example.
function memNN:predict(ex)
    -- Query.
    local qx = ex[1]
    -- Build memories.
    local memx = self:build_memory_vectors(ex)
    local x = self:build_input(qx, memx)
    x = self:prepare_input(x)

    local scores
    if self.opt.useCandidateLabels
    or self.opt.testWithCandidateLabels then
        local labels = ex.cands
        if self.opt.useMemHYasCandidates then
            labels = ex.memhy
            -- ex.cands = ex.memhy
        elseif self.opt.useMemYasCandidates then
            labels = ex.memy
            -- ex.cands = ex.memy
        end
        -- Copy the table so that we don't edit the original example inplace
        if self.opt.predEOS then
            labels = tablex.copy(labels)
            labels[#labels + 1] = g_train_data.EOS
        end
        self.inputEmbedding = self.mlp:forward(x)
        if self.scores == nil or self.scores:dim() == 0
        or self.scores:size(1) ~= #labels or not ex.numFixedCands then
            self.scores = self:Tensor(#labels):zero()
            self.labelEmbeddings = nil
        end
        if ex.numFixedCands and self.labelEmbeddings then
            -- Only compute the non-fixed candidates, as those have changed.
            for i = ex.numFixedCands + 1, #labels do
                local ycand = labels[i]
                self:embed(ycand, self.labelEmbeddings[i])
            end
        else
            -- Create cached label embedding matrix for speed on
            -- subsequent runs.
            if self.labelEmbeddings == nil or
                self.labelEmbeddings:size(1) ~= #labels then
                    self.labelEmbeddings = self:Tensor(
                        #labels, self.opt.embeddingDim)
            end
            for i = 1, #labels do
                local ycand = labels[i]
                self:embed(ycand, self.labelEmbeddings[i])
            end
        end
        if self.labelEmbeddings:dim() > 0 then
            torch.mv(self.scores, self.labelEmbeddings, self.inputEmbedding)
        end
        return self:scores_to_pred(self.scores, labels)
    else
        local labels = self.dict:get_labels()
        if self.opt.numNegSamples ~= nil then
            -- The output of the network is an embedding, and has to be
            -- combined with the possible label embeddings.
            if self.scores == nil then
                self.scores = torch.Tensor(self.dict.num_symbols)
            end
            if self.labelEmbeddings == nil then
                self.labelEmbeddings =
                    self.labelLT:get(1).weight:sub(1, self.dict.num_symbols)
            end
            local inputEmbedding = self.mlp:forward(x)
            torch.mv(self.scores, self.labelEmbeddings, inputEmbedding)
            scores = self.scores
        else
            -- The output of the network is the prediction for each
            -- dictionary symbol, so we only need to forward through the net.
            scores = self.mlp:forward(x)
            if self.opt.trainWithSupportingFacts then scores = scores[1]; end
        end
        local _val, ypred = scores:max(1)
        local response = torch.Tensor({ypred[1]})
        return response, labels, scores
    end
end

function memNN:compute_label_embeddings(labels)
    self.labelEmbeddings = torch.Tensor(#labels, self.opt.embeddingDim)
    for i = 1, #labels do
        -- embedding of a candidate is the average of those of its members
        local cc = labels[i]
        self:embed(cc, self.labelEmbeddings[i])
    end
    self.cscores = torch.Tensor(#labels)
end

-- Predict labels given an example.
function memNN:fast_predict_with_candidates(ex, updateLabelsEmbeddings)

    if not self.opt.useCandidateLabels and
    not self.opt.testWithCandidateLabels then
        -- fall back to the standard prediction function
        return self:predict(ex)
    end

    -- Query.
    local qx = ex[1]
    -- Build memories.
    local memx = self:build_memory_vectors(ex)
    local x = self:build_input(qx, memx)
    x = self:prepare_input(x)

    local labs = ex.cands
    if self.opt.useMemHYasCandidates then
        labs = ex.memhy
    elseif self.opt.useMemYasCandidates then
        labs = ex.memy
    end
    -- caching the candidates embeddings
    if not self.labelEmbeddings or #labs ~= self.labelEmbeddings:size(1) or
        self.opt.useMemHYasCandidates or self.opt.useMemYasCandidates or
    updateLabelsEmbeddings then
        self:compute_label_embeddings(labs)
    end
    local inputEmbedding = self.mlp:forward(x)
    self.cscores:mv(self.labelEmbeddings, inputEmbedding)
    return self:scores_to_pred(self.cscores, labs)
end

function memNN:fill_memory_slot(mx, m)
    if m:dim() == 2 then
        mx:copy(m)
    else
        mx:t()[1]:copy(m)
        local sz = m:size(1)
        local n = 1 / math.sqrt(sz)
        mx:t()[2]:fill(n)
    end
end

function memNN:build_memory_vectors(ex)
    local maxLen = self.opt.sentenceSize

    if self.memBuffer == nil then
        -- maxLen + feats is nonbinding--code below grows buffer when needed
        -- add slots for time/label features
        self.memBuffer = torch.Tensor(maxLen * 2 + 3, 2)
    end
    if self.memx == nil then
        -- memSize * maxLen is nonbinding--vecarrays auto-grow when needed
        self.memx =
            VectorArray:new(self.opt.memSize * maxLen, self.opt.memSize, 2)
    end
    self.memx:clear()

    local memx = self.memx
    local mems = 0
    local empty = true
    local clipped = false
    local tim = -1
    local htim = -1

    local function add_memory(m, mem_type)
        empty = false
        if mems == self.opt.memSize then
            -- Cannot add more memories! Have to clip .. :/
            clipped = true
            return
        end
        local sz = m:size(1)
        if self.memBuffer:size(1) < sz + 2 then
            self.memBuffer:resize(sz * 2 + 2, 2)
        end
        local mx = self.memBuffer:sub(1, sz)
        self:fill_memory_slot(mx, m)

        local timLabel
        if mem_type == 'hx' or mem_type == 'hy' then
            if self.opt.useMemHTimeFeatures then
                htim = htim + math.random(self.opt.timeVariance)
                timLabel = htim
            end
        elseif self.opt.useTimeFeatures then
            tim = tim + math.random(self.opt.timeVariance)
            timLabel = tim
        end

        if timLabel ~= nil or self.opt.useMemLabelFeatures then
            local feats = 0
            if timLabel ~= nil then feats = feats + 1 end
            if self.opt.useMemLabelFeatures then feats = feats + 1 end
            mx = self.memBuffer:sub(1, sz + feats)

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
        memx:add(mx)
        mems = mems + 1
    end

    for i = #ex.memx, 1, -1 do
        add_memory(ex.memx[i], 'x')
        if self.opt.useMemy then
            -- Add memories from previous own answers that might be relevant
            add_memory(ex.memy[i], 'y')
        end
    end
    -- Add hashed deep memories that might be relevant.
    for i = 1, #ex.memhx do
        add_memory(ex.memhx[i], 'hx')
        if self.opt.useMemHy then
            -- Add "replies" in history as well.
            add_memory(ex.memhy[i], 'hy')
        end
    end
    if empty then
        -- No memories found: Make a dummy memory so we have at least one!
        -- add_memory(torch.Tensor(1):fill(0), 'x')
        local null = torch.Tensor(1, 2):fill(0)
        null[1][1] = self.NULL.x[1]
        memx:add(null)
    end

    -- Useful to store this to extract it e.g. at evaluation time.
    self.lastMemClipped = clipped
    memx:clip()
    local memx_return = {memx.data, memx.len}
    return memx_return, clipped, empty
end

function memNN:get_valid_example()
    local ex
    while true do
        ex = g_train_data:get_random_example()
        if ex ~= nil then
            local a = g_train_data:get_positive_label(ex)
            if a ~= nil then break; end
        end
    end
    return ex
end

function memNN:embed(x, feature)
    local f = self.labelLT:forward(x)
    feature:copy(f)
    if self.cosine then
        local norm = feature:norm()
        if norm ~= 0 then
            feature:div(norm)
        end
    end
    return feature
end

function memNN:get_negative_label(ex, apos, i)
    local aneg
    if self.opt.cacheViolators and self.labelViolators ~= nil
    and i < self.labelViolators:size(2) then
        local pos_index = apos[1][1]
        local idx =  self.labelViolators[pos_index][i]
        if ex.negsTried == nil then
            ex.negsTried = {}
        end
        if idx ~= 0 and ex.negsTried[idx] == nil then
            aneg = g_train_data.yneg
            aneg[1][1] = idx
            ex.negsTried[idx] = true
        end
    end
    if aneg == nil then
        aneg = g_train_data:get_negative_label(ex)
    end
    return aneg
end

function memNN:update_violators(ex, apos, aneg)
    if self.opt.cacheViolators then
        -- Record violators for each label.
        local pos_index = apos[1][1]
        local neg_index = aneg[1][1]
        local idx = math.random(self.labelViolators:size(2))
        self.labelViolators[pos_index][idx] = neg_index
    else
        -- Do nothing, no caching.
    end
end

function memNN:add_to_qx_vec(qx, weight)
    if weight == nil then weight = 1; end
    if not self.qx_vec then
        self.qx_vec = torch.Tensor(self.dict.num_symbols)
    end
    self.qx_vec:zero()
    for i = 1, qx:size(1) do
        local index = qx[i][1]
        -- Ignore <NULL>, <EOS> and <UNK> words.
        if index > 3 then
            self.qx_vec[index] = qx[i][2] * weight
        end
    end
end

function memNN:negsample_update(ex, x, a)
    if self.opt.trainWithSupportingFacts then
        error('strong supervision not supported yet')
    end
    -- We can embed a bag of words for the answer in this setting.
    self.inputEmbedding:copy(self.mlp:forward(x))
    self:embed(a, self.posLabelEmbedding)
    local pos_score = torch.dot(self.inputEmbedding, self.posLabelEmbedding)

    -- Measure losses.
    local loss = {}
    loss.rank_loss = 1
    loss.mean_rank = 0
    if self.opt.computeResponseLoss and not self.opt.rankLabelDocuments then
        -- Compute response loss, this takes time so we mostly do it
        -- when debugging.
        loss.r = 1
        if self.scores == nil then
            self.scores = torch.Tensor(self.dict.num_symbols)
            self.labelEmbeddings =
                self.labelLT:get(1).weight:sub(1, self.dict.num_symbols)
        end
        torch.mv(self.scores, self.labelEmbeddings, self.inputEmbedding)
        local pred = self.scores
        local _val, ypred = pred:max(1)
        if ypred[1] == a[1] then
            loss.r = 0
        end
    end

    local errs = 0
    local anegs = {}
    self.temp2:zero()
    for i = 1, self.opt.numNegSamples do
        local aneg = self:get_negative_label(ex, a, i)
        self:embed(aneg, self.negLabelEmbedding)
        local neg_score = torch.dot(self.inputEmbedding, self.negLabelEmbedding)
        if neg_score > pos_score - self.opt.margin then
            self.temp2:add(self.negLabelEmbedding)
            anegs[#anegs + 1] = aneg:clone()
            self:update_violators(ex, a, aneg)
            errs = errs + 1
        end
        if i == 1 then
            -- On the first loop we measure the rank error.
            if pos_score > neg_score then loss.rank_loss = 0; end
        end
    end

    -- Rank loss applied to the positive label and SOME negatives.
    if errs == 0 then return loss; end
    self.temp:copy(-self.posLabelEmbedding)
    self.temp:add(1 / errs, self.temp2)

    self.mlp:zeroGradParameters()
    self.mlp:backward(x, self.temp)
    self.mlp:updateParameters(self.opt.learningRate)

    -- Must update the labelLTs as well!
    self.labelLT:forward(a)
    self.labelLT:zeroGradParameters()
    self.labelLT:backward(a, -self.inputEmbedding)
    -- labelLT Neg update.
    self.inputEmbedding:mul(1 / errs)
    for i = 1, #anegs do
        self.labelLT:forward(anegs[i])
        self.labelLT:backward(anegs[i], self.inputEmbedding)
    end
    self.labelLT:updateParameters(self.opt.learningRate)
    loss.mean_rank = #anegs
    return loss
end

function memNN:NLLsample_update(ex, x, a)
    if self.opt.trainWithSupportingFacts then
        error('strong supervision not supported yet')
    end

    if not self.NLLLabels then
        -- Create a small MLP that does the matrix multiplication
        -- between the chosen labels and the memNN output embedding.
        local d = nn.Sequential()
        local d2 = nn.ParallelTable()
        d2:add(nn.View(1, -1):setNumInputDims(1))
        d2:add(nn.Identity())
        d:add(d2)
        d:add(nn.MM(false, true))
        d:add(nn.View(-1):setNumInputDims(2))
        d:add(nn.LogSoftMax())
        self.NLLLabelRanker = d

        local lnn = nn.Sequential()
        lnn:add(self.labelLT:get(1))
        lnn:add(nn.View(-1, self.opt.labelSize,
                        self.opt.embeddingDim):setNumInputDims(2))
        lnn:add(nn.Sum(2))
        self.NLLLabelLT = lnn

        -- The target will always be the first candidate
        self.NLLlabelY = torch.Tensor({1})
    end

    local sample_sz

    -- Copy so that we don't edit the candidates in-place
    local ex_cands = ex.cands
    if self.opt.predEOS then
        ex_cands = tablex.copy(ex_cands)
        ex_cands[#ex_cands + 1] = g_train_data.EOS
    end
    local sample_negs = false
    if self.opt.useCandidateLabels
    and math.random() <= self.opt.sampleCands then
        sample_sz = #ex_cands * self.opt.labelSize
    elseif self.opt.numNegSamples then
        sample_sz = (1 + self.opt.numNegSamples) *  self.opt.labelSize
        sample_negs = true
    else
        error("can only be used with negative sampling or a candidate set")
    end
    if not self.NLLLabels or
    self.NLLLabels:size(1) < sample_sz + self.opt.labelSize then
        self.NLLLabels = torch.Tensor(sample_sz + self.opt.labelSize, 2)
    end

    local labs = self.NLLLabels
    labs:fill(self.NULL.x[1])

    -- We need to build an input which is the set of positives and negatives
    -- or a set of candidates
    local sz = 0
    local function add_label(a)
        local pos = sz + 1
        self.log:verbose(3, 'adding candidate target %s', a)
        if a:size(1) > self.opt.labelSize then
            a = a:sub(1, self.opt.labelSize)
        end
        labs:sub(pos, pos + a:size(1) - 1):copy(a)
        sz = sz + self.opt.labelSize
    end
    add_label(a)
    if sample_negs then
        for i = 2, self.opt.numNegSamples + 1 do
            local aneg = self:get_negative_label(ex, a, i)
            add_label(aneg)
            if self.opt.debugMode and self.dict.v2t then
                self.log:verbose(5, "NEG" .. i ..  ":" .. self.dict:v2t(aneg),
                    aneg:t()[2]:norm())
            end
        end
        if self.opt.debugMode and self.dict.v2t then
            self.log:verbose(5, "POS:" .. self.dict:v2t(a), a:t()[2]:norm())
        end
    else
        for i = 1, #ex_cands do
            if not util.same_tensor(a, ex_cands[i]) then
                add_label(ex_cands[i])
            end
        end
    end
    local labst = labs:sub(1, sz)
    local labelLT = self.NLLLabelLT
    local xe = self.mlp:forward(x)
    local ye = labelLT:forward(labst)
    local pred = self.NLLLabelRanker:forward({xe, ye})

    -- NLL update.
    local y = self.NLLlabelY
    self.criterion:forward(pred, y)
    self.NLLLabelRanker:zeroGradParameters()
    self.mlp:zeroGradParameters()
    labelLT:zeroGradParameters()
    local gl = self.criterion:backward(pred, y)
    local t = self.NLLLabelRanker:backward({xe, ye}, gl)
    self.mlp:backward(x, t[1])
    self.mlp:updateParameters(self.opt.learningRate)
    labelLT:backward(labst, t[2])
    labelLT:updateParameters(self.opt.learningRate)

    -- Measure losses:
    -- Rank loss.
    local loss = {};
    loss.r = 1
    loss.rank_loss = 1
    if pred[1] > pred[2] then loss.rank_loss = 0; end
    -- Response loss (i.e., is the response correct?).
    local _val, ypred = pred:max(1)
    if ypred[1] == 1 then
        loss.r = 0
    end
    -- Mean rank of the true label.
    local true_rank = pred:ge(pred[1]):sum()
    loss.mean_rank = true_rank

    return loss
end

-- This method only works for 1-word outputs
-- and strong supervision
function memNN:strongsup_single_word_output_update(ex, x, a)
    local preds = self.mlp:forward(x)
    local pred = preds[1]
    local y = a[1]

    -- Measure losses:
    -- Rank loss.
    local loss = {}
    loss.rank_loss = 1
    loss.r = 1
    local ind = math.random(pred:size(1))
    if pred[y[1]] > pred[ind] or y[1] == ind then loss.rank_loss = 0; end
    -- Response loss (i.e., is the response correct?).
    local _val, ypred = pred:max(1)
    if ypred[1] == y[1] then
        loss.r = 0
    end
    -- Mean rank of the true label.
    local true_rank = pred:ge(pred[y[1]]):sum()
    loss.mean_rank = true_rank

    -- Do update.
    if self.opt.rankLoss then
        error('not supported yet')
    end
    self.criterion:forward(pred, y[1])
    self.mlp:zeroGradParameters()
    local t = self.criterion:backward(pred, y[1])
    local ts = { t }
    local sfs = ex.sfs
      for i = 1, #preds - 1 do
        -- For each hop, add supervision wrt supporting facts.
        local index = i
        if sfs:size(1) < index then index = sfs:size(1); end
        local sf_label = sfs[index]
        self.sf_criterion[i]:forward(preds[i + 1], sf_label)
        ts[#ts + 1] =
            self.sf_criterion[i]:backward(preds[i + 1], sf_label)
    end
    self.mlp:backward(x, ts)
    self.mlp:updateParameters(self.opt.learningRate)
    return loss
end


-- This method only works for 1-word outputs.
function memNN:single_word_output_update(ex, x, a)
    local pred = self.mlp:forward(x)
    local y = a[1]

    -- Measure losses:
    -- Rank loss.
    local loss = {}
    loss.r = 1
    loss.rank_loss = 1
    local ind = math.random(pred:size(1))
    if pred[y[1]] > pred[ind] or y[1] == ind then loss.rank_loss = 0; end
    -- Response loss (i.e., is the response correct?).
    local _val, ypred = pred:max(1)
    if ypred[1] == y[1] then
        loss.r = 0
    end
    -- Mean rank of the true label.
    local true_rank = pred:ge(pred[y[1]]):sum()
    loss.mean_rank = true_rank

    -- Do update: two flavors -- rankLoss or NLLCriterion.
    if self.opt.rankLoss then
        -- Rank loss applied to the positive label AND all negatives at once.
        self.mlp:zeroGradParameters()
        local t = pred:clone():zero()
        local pos = pred[y[1]]
        local errs = 0
        for i = 1, pred:size(1) do
            if i ~= y[1] then
                if pred[i] > pos - self.opt.margin then
                    t[i] = 1
                    errs = errs + 1
                end
            end
        end
        t:div(errs)
        t[y[1]] = -1
        if errs > 0 then
            self.mlp:backward(x, t)
            self.mlp:updateParameters(self.opt.learningRate)
        end
    else
        -- NLL update.
        self.criterion:forward(pred, y[1])
        self.mlp:zeroGradParameters()
        local t = self.criterion:backward(pred, y[1])
        self.mlp:backward(x, t)
        self.mlp:updateParameters(self.opt.learningRate)
    end
    return loss
end

function memNN:gradClip()
    self.gradNorm  = self.paramdx:norm()
    if self.gradNorm > self.opt.gradClipMaxNorm then
        local scalar = self.opt.gradClipMaxNorm / self.gradNorm
        self.paramdx:mul(scalar)
    end
end

function memNN:prepare_input(x)
    -- Do nothing in the base class.
    return x
end

function memNN:update(ex)
    if self.opt.onlyTrainWithExWithCandidates then
        if ex.cands == nil or #ex.cands == 0 then
            -- Bail out, there are no candidates.
            return
        end
    end

    -- Put all questions and statements in the memory.
    -- If there is an answer in the current implementation we pick one of them
    -- at random.
    local a = g_train_data:get_positive_label(ex)
    if a == nil or a[1][1] == self.NULL.x[1] then
        -- For now, we don't try to answer statements.
        return nil, 0, 0
    end

    -- Get the data.
    if self.ex_cnt == nil then self.ex_cnt = 0; end
    self.ex_cnt = self.ex_cnt + 1
    local qx = ex[1]
    -- Build memories.
    local memx, clipped, empty = self:build_memory_vectors(ex)
    if (self.opt.dropLargeEx and clipped)
        or (self.opt.dropNoMem and empty) then
        return nil
    end
    local x = self:build_input(qx, memx)
    x = self:prepare_input(x)

    local loss
    if self.opt.numNegSamples == nil and not self.opt.useCandidateLabels then
        if self.opt.trainWithSupportingFacts then
            loss = self:strongsup_single_word_output_update(ex, x, a)
        else
            loss = self:single_word_output_update(ex, x, a)
        end
    else
        if self.opt.rankLoss then
            loss = self:negsample_update(ex, x, a)
        else
            loss = self:NLLsample_update(ex, x, a)
        end
    end

    -- Zero out the NULL symbol.
    for i = 1, #self.wt do
        self.wt[i].weight[self.NULL.x[1]]:zero()
    end
    -- Update stats.
    loss.mems = x[2][2]:size(1)
    return loss
end

-- Train one worker, can be used for hogwild training
-- or single-threaded.
function memNN:do_train_worker()
    self.exs_processed = 0
    local its = 0
    local loss = {}
    loss.rank_loss = 0; loss.mean_rank = 0; loss.mems = 0
    local losscnt = {}
    losscnt.rank_loss = 0; losscnt.mean_rank = 0
    local batch = 0
    local start_time = sys.clock()
    local last_save_time = sys.clock()
    local last_log_time = sys.clock();
    local last_valid_time = sys.clock();
    local last_lr_time = sys.clock();
    local best_valid_metric
    local last_valid_metric
    local zeroIts = 0
    while true do
        -- Train on a single example.
        g_train_data.trainData = true
        local ex = g_train_data:get_random_example()
        self.exs_processed = self.exs_processed + 1
        if ex ~= nil then
            local update_loss = self:update(ex)
            if update_loss ~= nil then
                for k, v in pairs(update_loss) do
                    if not loss[k] then
                        loss[k] = 0
                    end
                    if v >= 0 then
                        if loss[k] ~= nil then
                            loss[k] = loss[k] + v
                        end
                        if losscnt[k] ~= nil then
                            losscnt[k] = losscnt[k] + 1
                        end
                    end
                end
                batch = batch + 1
            end
        end
        -- Possibly change the learning rate.
        local time = sys.clock()
        local lr_time = time - last_lr_time
        if lr_time > self.opt.lrDecaySecs and self.opt.lrDecayDivisor > 1
        and self.my_threadidx == 1 then
            local s = '[lr decrease: ' .. self.opt.learningRate .. ' -> '
            self.opt.learningRate = self.opt.learningRate
                / self.opt.lrDecayDivisor
            if self.opt.learningRate < self.opt.lrDecayFinal then
                self.opt.learningRate = self.opt.lrDecayFinal
            end
            s = s .. self.opt.learningRate  .. ']'
            print(s)
            last_lr_time = time
        end
        -- Logging of various types follows.
        local total_time = time - start_time
        if self.opt.maxTrainTime ~= nil then
            if total_time > self.opt.maxTrainTime then break end
        end
        local num_epochs =
            self.exs_processed / g_train_data:num_loaded_examples()
        if self.opt.maxTrainEpochs ~= nil then
            if num_epochs > self.opt.maxTrainEpochs then break end
        end
        -- Save the model every N seconds.
        local save_time = time - last_save_time
        if (self.my_threadidx == 1 and self.opt.saveEveryNSecs ~= nil and
            save_time >= self.opt.saveEveryNSecs) then
            if self.opt.modelFilename ~= nil then
                self:save_all(self.opt.modelFilename)
            end
            last_save_time = sys.clock();
        end

        last_valid_time, best_valid_metric, last_valid_metric =
            self:validate_while_training(
                g_valid_data, best_valid_metric, last_valid_metric,
                time, last_valid_time)

        -- Log training error every N seconds.
        local log_time = time - last_log_time
        if (log_time >= self.opt.logEveryNSecs) then
            its = its + batch
            local thread_string = ''
            local doPrint = true
            if self.opt.numThreads > 1 then
                thread_string = 'thread:' .. self.my_threadidx .. ' ';
                if not self.opt.debugMode and self.my_threadidx > 1 then
                    doPrint = false;
                end
            end
            if loss.r and loss.r == 0 then
                zeroIts = zeroIts + 1
            else
                zeroIts = 0
            end
            if self.opt.stopTrainZeroIts > 0 and
            zeroIts >= self.opt.stopTrainZeroIts then
                break
            end
            if doPrint then
                local lossrdisplay = -1
                if loss.r  then
                    lossrdisplay = util.shortFloat(loss.r / batch)
                end
                -- calculate LT stats
                local lt_weights = self.wt[1].weight
                local max_norm = 0
                for i = 1, lt_weights:size(1) do
                    local curr_norm = lt_weights[i]:norm()
                    if curr_norm > max_norm then max_norm = curr_norm end
                end
                local lt_norm = lt_weights:norm()
                if lt_norm ~= lt_norm then error('weights have norm of nan') end

                -- build log string
                local log_string = '[' .. thread_string
                    .. 'exs:' .. its
                    .. ' epoch:' .. util.shortFloat(num_epochs)
                    .. ' resp_loss:' .. lossrdisplay
                    .. ' rank_loss:'
                    .. util.shortFloat(loss.rank_loss / losscnt.rank_loss)
                    .. ' mean_rank:'
                    .. util.shortFloat(loss.mean_rank / losscnt.mean_rank)
                    .. ' time:' .. math.floor(total_time) .. 's'
                    .. ' data:' .. g_train_data:num_loaded_examples()
                    .. ' mems:' .. util.shortFloat(loss.mems / batch)
                    .. ' wtnorm:' .. util.shortFloat(lt_norm)
                    .. ' wtmaxnorm:' .. util.shortFloat(max_norm)
                    .. ' rnorm:' .. util.shortFloat(self.rlin.weight:norm())
                if best_valid_metric then
                    log_string = log_string
                        .. ' best_valid:' .. util.shortFloat(best_valid_metric)
                end

                if self.gradNorm ~= -1 then
                    if self.gradNorm > self.opt.gradClipMaxNorm then
                        log_string = log_string
                            .. ' gnorm:>' .. util.shortFloat(self.gradNorm)
                    else
                        log_string = log_string
                            .. ' gnorm:' .. util.shortFloat(self.gradNorm)
                    end
                end
                log_string = log_string .. ']'
                print(log_string)
                if self.opt.logTrainingFile and self.opt.modelFilename and
                self.opt.allowSaving then
                    local f, err = io.open(self.opt.modelFilename .. ".log",
                                           "a")
                    if f ~= nil then
                        f:write(os.date() .. ":" .. log_string .. "\n")
                        f:close()
                    else
                        print('output failed! make sure output folder exists'..
                            'and has allowable permissions')
                        error('writing log failed! ' .. tostring(err))
                    end
                end
            end
            collectgarbage()
            for k, _ in pairs(loss) do loss[k] = 0; end
            losscnt.rank_loss = 0; losscnt.mean_rank = 0
            batch = 0
            last_log_time = sys.clock();
        end
    end
    return loss, batch
end

function memNN:closest_symbols(rx, topk,
                               queryUsesLabelDict, targetUsesLabelDict)
    if queryUsesLabelDict == nil then queryUsesLabelDict = false; end
    if targetUsesLabelDict == nil then targetUsesLabelDict = false; end
    if topk == nil then topk = 10; end
    -- Print out the neighboring symbols of an example (string).
    local tx
    if type(rx) == 'userdata' then
        -- We can input a pre-built vector to this function, too.
        if rx:dim() == 1 then
            tx = self:Tensor(rx:size(1), 2):fill(1)
            tx:t()[1]:copy(rx)
        else
            tx = rx
        end
    else
        print("[Looking for neighbors of " .. rx .. "]")
        local x = {}
        local words = pl.utils.split(rx, ' ')
        for k,v in pairs(words) do
            if self.dict:symbol_to_index(v) ~= nil then
                x[#x+1] = {self.dict:symbol_to_index(v), 1.}
            end
        end
        tx = self:Tensor(x)
    end
    local nearest = function(topk)
        local WQ, WT, W1, W2
        W1 = self.wt[1].weight
        W2 = self.labelLT:get(1).weight
        if queryUsesLabelDict then
            WQ = W2
        else
            WQ = W1
        end
        if targetUsesLabelDict then
            WT = W2
        else
            WT = W1
        end
        -- We assume only one word.
        if tx:size(1) > 1 then
            error("only single word similarities supported")
        end
        local v = WQ[tx[1][1]]
        self.scores = torch.Tensor(WT:size(1))
        torch.mv(self.scores, WT, v)
        -- Possibly clip extra stuff like time features, etc.
        local scores = self.scores:sub(1, self.dict.num_symbols)
        local cands = self.dict:get_labels()
        local vals, inds = torch.topk(scores, topk, 1, true, true)
        for i= 1, topk do
            print(self.dict:v2t(cands[inds[i]]) .. ' '
                      .. vals[i] .. ' ' ..  inds[i])
        end
    end
    print("-- Nearest outputs ..")
    nearest(topk)
end

return memNN
