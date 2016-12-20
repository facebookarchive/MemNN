-- Copyright 2004-present Facebook. All Rights Reserved.

local cmdline = require('library.cmd')
cmd = cmdline:new()
cmd:reset_default('modelClass', 'library.kvmemnn_model')
cmd:reset_default('dataClass', 'library.data')

opt = cmd:parse(arg, true)

-- Define the data and model and add their command line options.
local mlp = require(opt.modelClass)
mlp:add_cmdline_options(cmd)
local data = require(opt.dataClass)
data:add_cmdline_options(cmd)
cmd:reset_default('initWeights', 0.1)
cmd:reset_default('learningRate', 0.005)

cmd:reset_default('negSampleFromSameSrc', true)

cmd:reset_default('numThreads', 4)
cmd:reset_default('logEveryNSecs', 10)
cmd:reset_default('maxTrainTime', 3 * 24 * 60 * 60)
cmd:reset_default('embeddingDim', 500)
cmd:reset_default('numNegSamples', 1000)
cmd:reset_default('threadsShareData', true)
cmd:reset_default('logTrainingFile', true)
cmd:reset_default('validEveryNSecs', 900)
cmd:reset_default('validMetric', 'h1')

cmd:reset_default('dictTFIDFPow', 0)
cmd:reset_default('dictTFIDF', true)
cmd:reset_default('dictTFIDFLabel', false)
cmd:reset_default('dictWhiteSpace', '[ ]')
cmd:reset_default('dictUseUNK', false)
cmd:reset_default('preprocessText', true)
cmd:reset_default('dictMaxNGramSz', 100)

-- Quite MemN2N specific
cmd:reset_default('metric', 'dot')
cmd:reset_default('maxHops', 2)
cmd:reset_default('memSize', 1000)
cmd:reset_default('useTimeFeatures', false)
cmd:reset_default('useMemy', true)
cmd:reset_default('useMemHy', true)
cmd:reset_default('LTsharedWithResponse', true)
cmd:reset_default('rotateBeforeResponse', false)
cmd:option('wordModel', 'bowTFIDF')

cmd:option('memoryType', 'wiki')
cmd:option('wikiFormat', 'w=0-d=3-i-m')
cmd:option('multidictSuffix', '_1') -- blank for nothing, _1 for one extra dict

cmd:option('id', '1', 'Name of model.')
local opt = cmd:parse(arg)

-- Set filenames based on other parameters.
local model_basename = "./output/"

local data_basename = './data/torch/'

cmd:reset_default('dictFile', data_basename .. 'dict.txt')
local trainData = data_basename .. 'train' .. opt.multidictSuffix .. '.txt.vecarray'
cmd:reset_default('trainData', trainData)
local validData = data_basename .. 'dev' .. opt.multidictSuffix .. '.txt.vecarray'
cmd:reset_default('validData', validData)
local testData = data_basename .. 'test' .. opt.multidictSuffix .. '.txt.vecarray'
cmd:reset_default('testData', validData)
local hash_file = data_basename .. 'wiki-' .. opt.wikiFormat .. '.txt.hash'
cmd:reset_default('memHashFile', hash_file)

opt = cmd:parse(arg)

-- include any parameters that you're varying in the name so the runs save to
-- different log files
local function name(v)
    local s = model_basename .. 'kvmemnn-' .. opt.wikiFormat
        .. '-id' .. opt.id
        .. "-lr=" .. opt.learningRate
        .. "-eDim=" .. opt.embeddingDim
        .. "-initWeights=" .. opt.initWeights
        .. "-hops=" .. tostring(opt.maxHops)
        .. '-neg=' .. tostring(opt.numNegSamples)
        .. "-TPow=" .. opt.dictTFIDFPow
        .. '-rots=' .. tostring(opt.rotateBeforeResponse)
        .. '-ltshare=' .. tostring(opt.LTsharedWithResponse)
        .. '-metric=' .. tostring(opt.metric)
    s = s .. ".model"
    return s
end

if opt.modelFilename == nil then
    opt.modelFilename = name(opt.version)
end

cmd:print(opt)

opt.debugMode = false

return opt
