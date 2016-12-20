-- Copyright 2004-present Facebook. All Rights Reserved.
-- Main function to evaluate
-- @lint-skip-luachecker
-- Arguments:
-- 1) Name of file which contains options.
local optFile = arg[1]

require('torch')
if optFile ~= nil then
    -- Kill the first arg
    local args = {}
    for i = 2, #arg do
        args[#args + 1] = arg[i]
    end
    arg = args
    if optFile == 'model' then
        -- Load from model options instead.
        local cmdline =
            require('library.cmd')
        cmd = cmdline:new()
        opt = cmd:parse_from_modelfile_and_override_with_args(arg)
        cmd:print(opt)
        print("[from loaded options]")
    else
        opt = require(optFile)
    end
end

local model = require(opt.modelClass)
if mlp == nil then
    opt.dictFullLoading = true
    local allow = opt.allowSaving
    opt.allowSaving = true
    if not opt.evalIgnoreMissingModelFile then
        local f = io.open(opt.modelFilename)
        if f == nil then
            error('cannot find model: ' .. opt.modelFilename)
        else
            f:close()
        end
    end
    mlp = model:init_mlp(opt)
    opt.allowSaving = allow
end

local evalData = opt.testData
if opt.evalFileSuffix == 'valid' and opt.validData ~= nil then
    evalData = opt.validData
end

datalib = require(opt.dataClass)
g_test_data = datalib:create_data(evalData, nil,
                                  opt, mlp.dict:get_shared())

evallib = require(opt.evalClass)
eval = evallib:create(mlp, opt)

local t1 = os.time()
test_metrics, test_stats = eval:eval(g_test_data, opt.evalMaxTestEx)
local t2 = os.time()
print(string.format('Finished evaluation in %d seconds.', t2 - t1))

if opt.allowSaving ~= false then
    local evalFileSuffix = opt.evalFileSuffix
    if evalFileSuffix ~= nil then
        evalFileSuffix = '_' .. evalFileSuffix
    end
    eval:save_metrics(
        opt.modelFilename .. '.eval' .. evalFileSuffix,
        test_metrics, 'test')
end
