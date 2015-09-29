-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('cunn')
require('nngraph')
paths.dofile('LinearNB.lua')

local function build_memory(params, input, context, time)
    local hid = {}
    hid[0] = input
    local shareList = {}
    shareList[1] = {}

    local Ain_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Ain_t = nn.LookupTable(params.memsize, params.edim)(time)
    local Ain = nn.CAddTable()({Ain_c, Ain_t})

    local Bin_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Bin_t = nn.LookupTable(params.memsize, params.edim)(time)
    local Bin = nn.CAddTable()({Bin_c, Bin_t})

    for h = 1, params.nhop do
        local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[h-1])
        local MMaout = nn.MM(false, true):cuda()
        local Aout = MMaout({hid3dim, Ain})
        local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
        local P = nn.SoftMax()(Aout2dim)
        local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
        local MMbout = nn.MM(false, false):cuda()
        local Bout = MMbout({probs3dim, Bin})
        local C = nn.LinearNB(params.edim, params.edim)(hid[h-1])
        table.insert(shareList[1], C)
        local D = nn.CAddTable()({C, Bout})
        if params.lindim == params.edim then
            hid[h] = D
        elseif params.lindim == 0 then
            hid[h] = nn.ReLU()(D)
        else
            local F = nn.Narrow(2,1,params.lindim)(D)
            local G = nn.Narrow(2,1+params.lindim,params.edim-params.lindim)(D)
            local K = nn.ReLU()(G)
            hid[h] = nn.JoinTable(2)({F,K})
        end
    end

    return hid, shareList
end

function g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
    local time = nn.Identity()()
    local hid, shareList = build_memory(params, input, context, time)
    local z = nn.LinearNB(params.edim, params.nwords)(hid[#hid])
    local pred = nn.LogSoftMax()(z)
    local costl = nn.ClassNLLCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})
    local model = nn.gModule({input, target, context, time}, {cost})
    model:cuda()
    -- IMPORTANT! do weight sharing after model is in cuda
    for i = 1,#shareList do
        local m1 = shareList[i][1].data.module
        for j = 2,#shareList[i] do
            local m2 = shareList[i][j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
        end
    end
    return model
end
