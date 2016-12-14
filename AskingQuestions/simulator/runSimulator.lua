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
