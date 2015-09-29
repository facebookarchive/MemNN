% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Sigmoid < Module
    properties
    end
    methods
        function obj = Sigmoid()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            obj.output = 1 ./ (1 + exp(-input));
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            grad_input = grad_output .* obj.output .* (1 - obj.output);
        end
    end
end