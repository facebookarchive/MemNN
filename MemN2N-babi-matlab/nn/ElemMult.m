% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef ElemMult < Module
    properties
        weight
    end
    methods
        function obj = ElemMult(w)
            obj = obj@Module();
            obj.weight = w;
        end
        function output = fprop(obj, input)
            obj.output = bsxfun(@times, input, obj.weight);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = bsxfun(@times, grad_output, obj.weight);
            grad_input = obj.grad_input;
        end
    end
end