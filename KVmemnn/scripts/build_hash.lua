-- Copyright 2004-present Facebook. All Rights Reserved.
-- Build long term memory hash.
--------------------------------------------------------------------------------
-- Arguments:
-- 1) Name of file which contains options.
local optFile = arg[1]
-- 2) Hash file out.
local hashFileOut = arg[2]
-- 3) File of memories to hash,
-- if not specified it's assuming you are loading the saved hash file.
local hashFileIn = arg[3]

require('torch')
if optFile ~= nil then
    -- Kill all the args apart from the ones after the first 3 to pass
    -- them to the options class.
    local args = {}
    for i = 4, #arg do
        args[#args + 1] = arg[i]
    end
    arg = args
end
opt = require(optFile)

parserlib = require('library.parse')
parser = parserlib:new(opt)
local model = require(opt.modelClass)
if mlp == nil then
    opt.dictFullLoading = true
    mlp = model:init_mlp(opt)
end
if hash == nil then
    hashlib = require('library.hash')
    hash = hashlib:new(opt, mlp.dict)
    valib = require('library.vector_array')
    if hashFileIn ~= nil and hashFileIn ~= '' then
        if g_train_data == nil then
            local datalib = require(mlp.opt.dataClass)
            opt.trainData = hashFileIn
            opt.memHashFile = nil
            g_train_data = datalib:create_data(
                hashFileIn, nil,
                opt, mlp.dict:get_shared())
        end
        hash:add_facts(g_train_data)
        if opt.dictUseUNKHashLookup then
            -- Not contiguous: cannot convert to vector array
            -- in this case.
            hash:save(hashFileOut)
        else
            print "done"
            hash:convert_to_vector_array()
            hash:save_va(hashFileOut)
        end
    else
        if opt.dictUseUNKHashLookup then
            -- Not contiguous: cannot use vector array in this case.
            hash:load(hashFileOut)
        else
            hash:load_va(hashFileOut)
        end
    end
end

function vec2txt(x, dict)
    local s = ''
    if x:dim() == 2 then x = x:t()[1]; end
    for i = 1, x:size(1) do
        local word
        if x[i] <= dict.num_symbols then
            word = dict:index_to_symbol(x[i])
        end
        if word == nil then
            word = x[i]
            if dict._unk[word] ~= nil then
                word = word .. '=' .. dict._unk[word]
            end
        end
        s = s .. word .. ' '
    end
    return s
end

-- Try your own example to see if it worked, e.g.:
-- hash.opt.hashDocWindow = 1
-- hash_example("who directed sleepy hollow?")
-- hash_example("what did audrey hepburn star in?")
function hash_example(s)
    ex = {}
    ex[1] = parser:parse_test_time(s, mlp.dict)
    print("hash key:" .. vec2txt(ex[1], mlp.dict))
    if ex ~= nil then
        c1, c2, cind = hash:get_candidate_set(ex[1])
        print("--- found " .. #c1 .. " hashes ---")
        for i = 1, math.min(#c1, 100) do
            local info = '[fact:' .. cind[i][1]
                .. ' doc:' .. cind[i][2]
                .. ' sent:' .. cind[i][3] .. ']'
            print(vec2txt(c1[i], mlp.dict), vec2txt(c2[i], mlp.dict), info)
            -- print(mlp.dict:vector_to_text(c1[i]))
        end
        print("[found " .. #c1 .. " hashes.]")
    end
end
