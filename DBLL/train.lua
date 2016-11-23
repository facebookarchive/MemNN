require "torch"
require "cunn"
require "cutorch"
require "nngraph"
local params=require("parse")
local model=require("memmnet")
cutorch.manualSeed(123)
--cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
