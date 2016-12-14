require "torch"
require "cunn"
require "cutorch"
require "nngraph"
local params=require("../supervised/parse")
local model=require("../supervised/memmnet")
cutorch.manualSeed(123)
--cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
