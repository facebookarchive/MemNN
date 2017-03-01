-- Copyright 2004-present Facebook. All Rights Reserved.
require('nn')

local WeightedLookupTableSkinny, parent =
    torch.class('nn.WeightedLookupTableSkinny', 'nn.Module')
WeightedLookupTableSkinny.__version = 1

local ffi = require('ffi')
local C = ffi.load('libmemnn.so')
ffi.cdef [[
void wlts_updateOutput(
    int dims, long* sizes, int input_dim,
    float* input, float* weight, float* output
);
int wlts_accGradParameters(
    float scale, int size, int dims, long* sizes, int input_dim, long* inputs,
    float* input, float* gradOutput, float* gradWeight, int* inputWeights
);
void wlts_accUpdateGradParameters(
    float lr, int size, int dims, long* sizes, int input_dim,
    float* input, float* gradOutput, float* weight
);
void wlts_updateParameters(
    float lr, long inputs, int dims, long* sizes,
    float* weight, float* gradWeight, int* inputWeights
);
]]

function WeightedLookupTableSkinny:__init(nIndex, embDim)
    parent.__init(self)

    self.size = torch.LongStorage(2)
    self.size[1] = nIndex or 1
    self.size[2] = embDim or 1

    self._input = torch.Tensor()
    self.weight = torch.Tensor(self.size)
    self.gradWeight = nil
    self.inputs = torch.LongStorage(1)
    self.inputWeights = torch.IntTensor(10000)

    -- set initial max number of word updates to queue to 10000 or fewer
    self.gradWeight = torch.Tensor(10000, embDim)

    self:reset()
end

function WeightedLookupTableSkinny:reset(stdv)
    stdv = stdv or 1
    if nn.oldSeed then
        self.weight:apply(function()
            return torch.normal(0, stdv)
        end)
    else
        self.weight:normal(0, stdv)
    end
end

function WeightedLookupTableSkinny:makeInputContiguous(input)
   -- make sure input is a contiguous torch Tensor
   if not input:isContiguous() then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

--[[
Parameters:
* `Input` should be an n x 2 tensor where the first column is dictionary indexes
and the second column is weights.
]]
function WeightedLookupTableSkinny:updateOutput(input)
    if not input then error('Nil input to WeightedLookupTableSkinny.') end
    local output_size = torch.LongStorage(self.size:size()):copy(self.size)
    output_size[1] = input:size(1)
    self.output:resize(output_size)

    local input_dim = input:dim()
    if input_dim > 2 then
        error('Input has unsupported dimensions.')
    end
    input = self:makeInputContiguous(input)

    C.wlts_updateOutput(
        output_size:size(),
        torch.data(output_size),
        input_dim,
        torch.data(input),
        torch.data(self.weight),
        torch.data(self.output)
    )

    return self.output
end

function WeightedLookupTableSkinny:zeroGradParameters()
    self.inputs[1] = 0
    if self.queueResize then
        self.queueResize = false
        local grd_wgt_sz = torch.LongStorage(self.size:size()):copy(self.size)
        -- double size or hit max size
        grd_wgt_sz[1] = self.gradWeight:size(1) * 2
        self.gradWeight:resize(grd_wgt_sz)
        self.inputWeights:resize(grd_wgt_sz[1])
    end
end

function WeightedLookupTableSkinny:accGradParameters(
    input, gradOutput, scale)
    if not input then error('Nil input to WeightedLookupTableSkinny.') end
    if not gradOutput then
        error('Nil gradOutput to WeightedLookupTableSkinny.')
    end
    scale = scale or 1

    local input_dim = input:dim()
    if input_dim > 2 then
        error('Input has unsupported dimensions.')
    end
    input = self.copiedInput and self._input or input
    if not gradOutput:isContiguous() then
        self._gradOutput = self._gradOutput or gradOutput.new()
        self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
        gradOutput = self._gradOutput
    end

    self.queueResize = 1 == C.wlts_accGradParameters(
        scale,
        input:size(1),
        self.gradWeight:size():size(),
        torch.data(self.gradWeight:size()),
        input_dim,
        torch.data(self.inputs),
        torch.data(input),
        torch.data(gradOutput),
        torch.data(self.gradWeight),
        torch.data(self.inputWeights)
    )
end

function WeightedLookupTableSkinny:accUpdateGradParameters(
    input, gradOutput, lr)
    if not input then error('Nil input to WeightedLookupTableSkinny.') end
    if not gradOutput then
        error('Nil gradOutput to WeightedLookupTableSkinny.')
    end
    if input:size(1) ~= gradOutput:size(1)
    or self.weight:size(2) ~= gradOutput:size(2) then
        error('Bad dimensions for gradOutput in WeightedLookupTableSkinny.')
    end
    lr = lr or 1
    if lr == 0 then return end

    local input_dim = input:dim()
    if input_dim > 2 then
        error('Input has unsupported dimensions.')
    end
    input = self.copiedInput and self._input or input
    if not gradOutput:isContiguous() then
        self._gradOutput = self._gradOutput or gradOutput.new()
        self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
        gradOutput = self._gradOutput
    end

    C.wlts_accUpdateGradParameters(
        lr,
        input:size(1),
        self.size:size(),
        torch.data(self.size),
        input_dim,
        torch.data(input),
        torch.data(gradOutput),
        torch.data(self.weight)
    )
end

function WeightedLookupTableSkinny:updateParameters(learningRate)
    learningRate = learningRate or 1
    if learningRate == 0 then return end
    C.wlts_updateParameters(
        learningRate,
        self.inputs[1],
        self.weight:size():size(),
        torch.data(self.weight:size()),
        torch.data(self.weight),
        torch.data(self.gradWeight),
        torch.data(self.inputWeights)
    )
end
