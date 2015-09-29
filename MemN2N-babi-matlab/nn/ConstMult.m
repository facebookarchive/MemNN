% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef ConstMult < Module
    properties
        c
    end
    methods
        function obj = ConstMult(c)
            obj = obj@Module();
            obj.c = c;
        end
        function output = fprop(obj, input)
            obj.output = obj.c * input;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = obj.c * grad_output;
            grad_input = obj.grad_input;
        end
    end
end