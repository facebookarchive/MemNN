-- Copyright 2004-present Facebook. All Rights Reserved.
-- Main function to load and train -- some globals remain.
-- @lint-skip-luachecker
-----------------------------------------------------
-- Arguments:
-- 1) Name of file which contains options.
local optFile = arg[1]
local opt
require('torch')

if optFile ~= nil then
    -- Kill all the args apart from the ones after the first
    -- to skip the options class.
    local args = {}
    for i = 2, #arg do
        args[#args + 1] = arg[i]
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
else
    error('missing options file')
end

if opt.profi then
    print('WARNING: running slower because of profiling')
    ProFi = require('ProFi')
    ProFi:start()
end

model = require(opt.modelClass)

if mlp == nil then
    mlp = model:init_mlp(opt)
end

mlp:train()

if opt.profi then
   ProFi:stop()
   ProFi:writeReport('/tmp/MyProfilingReport.txt')
end
