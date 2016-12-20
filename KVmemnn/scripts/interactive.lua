-- Copyright 2004-present Facebook. All Rights Reserved.
-- @lint-skip-luachecker
-- Main function to evaluate.
-- Arguments: 1) Name of file which contains options.

local optFile = arg[1]

require('torch')
local interlib = require('library.interactive_lib')

if optFile ~= nil then
    -- Kill the first arg
    local args = {}
    for i = 2, #arg do
        local a = arg[i]
        args[#args + 1] = a
    end
    arg = args
    if optFile == 'model' then
        -- Load from model options instead.
        local cmdline = require('library.cmd')
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
    opt.allowSaving = true
    local f = io.open(opt.modelFilename)
    if f == nil then
        print('** cannot find model: ' .. opt.modelFilename)
    else
        f:close()
    end
    mlp = model:init_mlp(opt)
end

datalib = require(opt.dataClass)
g_train_data = datalib:create_data(opt.trainData, nil,
                                   opt, mlp.dict:get_shared())
if opt.testData then
    g_test_data = datalib:create_data(opt.testData, nil,
                                      opt, mlp.dict:get_shared())
end
g_inter = interlib:create(mlp, g_train_data)

function help()
    print("Commands:")
    print("- g_inter:run('some query text here')");
    print("- ex = g_inter:one_random_example(g_train_data);");
    print("- ex = g_inter:next_example(g_train_data);");
    print("- ex = g_inter:next_example_with_candidates(g_train_data);");
    print("- ex = g_inter:one_random_example(g_test_data);");
    print("- g_inter.eval:eval(g_test_data, 1000)");
    print("- x = g_inter:get_negative(ex);");
    print("- y = mlp:predict(ex);")
    print("- y = g_inter:run('query')")
    print("- y = g_inter:run_dialog('query')")
    print("- g_inter:print_detailed_prediction();")
    print("- help();");
end

help()
