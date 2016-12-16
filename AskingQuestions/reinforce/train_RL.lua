-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

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
