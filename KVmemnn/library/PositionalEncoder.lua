-- Copyright 2004-present Facebook. All Rights Reserved.
require('torch')
require('nn')

local PositionalEncoder, parent =
    torch.class('nn.PositionalEncoder', 'nn.Module')

-- import C implementation of main functionality
local ffi = require('ffi')
local C = ffi.load('libmemnn.so')
ffi.cdef [[
    void pe_transform(int d, int size, float* input, float* len, float* result);
]]

--[[
- For more details, see page 5 of this paper: http://arxiv.org/abs/1503.08895
- forwardLen specifies whether to also return the len field of the input
--]]
function PositionalEncoder:__init(forwardLen)
    parent.__init(self)
    self.forwardLen = forwardLen
    if self.forwardLen then
        self.output = {torch.Tensor(), torch.Tensor()}
    else
        self.output = torch.Tensor()
    end
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

--[[
- Computes the forward operation using the C transform.
- If precomputed matrix is available (see lowMemoryMode), uses it.
--]]
function PositionalEncoder:updateOutput(input)
    local inputs = input[1]
    local len = input[2]
    if len:type() == 'torch.DoubleTensor' then
        len = len:type('torch.FloatTensor')
    end

    local output
    if self.forwardLen then
        output = self.output[1]
        self.output[2] = len:clone()
    else
        output = self.output
    end
    output:resizeAs(inputs)

    C.pe_transform(
        inputs:size(2),
        len:size(1),
        torch.data(inputs),
        torch.data(len),
        torch.data(output)
    )

    return self.output
end

--[[
  - Computes the backward operation using the C transform.
  - If precomputed matrix is available (see lowMemoryMode), uses it.
--]]
function PositionalEncoder:updateGradInput(input, gradOutput)
    local inputs = input[1]
    local len = input[2]
    if len:type() == 'torch.DoubleTensor' then
        len = len:type('torch.FloatTensor')
    end

    if type(gradOutput) == 'table' then
        gradOutput = gradOutput[1]
    end

    -- contains the useful output
    self.gradInput[1]:resizeAs(inputs)
    -- empty, just maintains API boundaries
    self.gradInput[2] = len:clone()

    C.pe_transform(
        inputs:size(2),
        len:size(1),
        torch.data(gradOutput),
        torch.data(len),
        torch.data(self.gradInput[1])
    )
    return self.gradInput
end
