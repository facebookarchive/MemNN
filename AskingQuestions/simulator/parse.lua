-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tds = require('tds')
local pl = require('pl.import_into')()
local cmd = torch.CmdLine()
cmd:option('-mode', 'train', 'mode of simulation: train|dev|test')
cmd:option('-task',1, "1,2,3,4,5,6,7,8,9")
cmd:option('-prob_correct_final_answer', 0.5, 'prob of being correct'
    .."when generating the final answer to the question")
cmd:option('-prob_correct_intermediate_answer', 0.5, 'prob of being correct'
    .."when generating the answers to the intermediate answers")
cmd:option('-homefolder','../data/movieQA_kb','homefolders to store the database')
cmd:option("-junk",true,"whether to incorporate junk i.e., random QA pairs")
cmd:option("-setting","AQ","the setting, takes value of AQ, QA or mix")
cmd:option("-randomQuestionNumTotal",5,"the number of junk QA pairs in history")
cmd:option("-output_dir","./","the output file name")
cmd:option("-output_fname","","the output file name")

local opt = cmd:parse(arg)

opt.dataset_file=opt.homefolder.."/"..opt.mode..'.txt'
opt.template_file=opt.homefolder.."/"..opt.mode.."_template.txt" -- question templates
opt.question_entity_file=opt.homefolder.."/"..opt.mode.."_entity.txt" -- the entity the teacher is asking about
opt.question_relation_file=opt.homefolder.."/"..opt.mode.."_relation.txt"
opt.kb_file=opt.homefolder.."/movie_kb.txt"
opt.typo_file=opt.homefolder.."/word_typo"
opt.all_entities=opt.homefolder.."/movie_entities.txt"
opt.question_template_file=opt.homefolder.."/question_templates.txt"
if opt.output_fname=="" then
    opt.output_fname=opt.output_dir.."/Task"..opt.task.."_"..opt.setting.."_"..opt.mode..".txt"
end

if opt.mode=="dev" or opt.mode=="test" then
    opt.CopyAllAnswers=true;
end

return opt
