require "torch"
require "cunn"
require "cutorch"
require "nngraph"
params=require("../reinforce/parse")
local model=require("../reinforce/RL_memmnet")

cutorch.manualSeed(123)
--cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
