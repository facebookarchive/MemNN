-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tds = require('tds')
local pl = require('pl.import_into')()
local Simulator=require('../simulator/Simulator')
local movieTeacher=require('../simulator/movieTeacher')
local movieStudent=require('../simulator/movieStudent')
local opt=require('../simulator/parse')
movieTeacher:Initial(opt)
movieStudent:Initial(opt)
Simulator:Initial(opt,{movieTeacher,movieStudent})

for i = 1, #movieTeacher.dataset do
    --print('')
    Simulator:roundtable()
    Simulator:print_log()
end
Simulator:close_files()
