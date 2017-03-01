-- Copyright 2004-present Facebook. All Rights Reserved.

local SumVecarr, parent = torch.class('nn.SumVecarr', 'nn.Module')

local ffi = require('ffi')
local C = ffi.load('libmemnn.so')
ffi.cdef [[
    void sum_doubles(double* data, float* len, int size, int dim,
        double* result);
    void grad_doubles(double* gradOutput, float* len, int outputSize, int dim,
        double* gradInput);
    void sum_floats(float* data, float* len, int size, int dim, float* result);
    void grad_floats(float* gradOutput, float* len, int outputSize, int dim,
        float* gradInput);
]]

function SumVecarr:__init()
    parent.__init(self)
    self.gradInput = {}
end

function SumVecarr:updateOutput(input)
    local data, len = unpack(input)
    if len:type() == 'torch.DoubleTensor' then
        len = len:type('torch.FloatTensor')
    end

    self.output:resize(len:size(1), data:size(2))

    -- C implementation
    if data:type() == 'torch.DoubleTensor' then
        C.sum_doubles(
            torch.data(data),
            torch.data(len),
            len:size(1),
            data:size(2),
            torch.data(self.output)
        )
    else
        C.sum_floats(
            torch.data(data),
            torch.data(len),
            len:size(1),
            data:size(2),
            torch.data(self.output)
        )
    end
    return self.output
end

function SumVecarr:updateGradInput(input, gradOutput)
    local data, len = unpack(input)
    if len:type() == 'torch.DoubleTensor' then
        len = len:type('torch.FloatTensor')
    end
    if self.gradInput[1] == nil then
        self.gradInput[1] = torch.Tensor()
    end
    self.gradInput[1]:resizeAs(data)
    self.gradInput[2] = len:clone()

    -- C implementation
    if data:type() == 'torch.DoubleTensor' then
        C.grad_doubles(
            torch.data(gradOutput),
            torch.data(len),
            gradOutput:size(1),
            data:size(2),
            torch.data(self.gradInput[1])
        )
    else
        C.grad_floats(
            torch.data(gradOutput),
            torch.data(len),
            gradOutput:size(1),
            data:size(2),
            torch.data(self.gradInput[1])
        )
    end
    return self.gradInput
end
