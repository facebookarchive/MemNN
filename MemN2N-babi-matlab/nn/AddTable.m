% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef AddTable < Module
    properties
    end
    methods
        function obj = AddTable()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            obj.output = input{1};
            for i = 2:length(input)
                obj.output = obj.output + input{i};
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(input)
                obj.grad_input{i} = grad_output;
            end
            grad_input = obj.grad_input;
        end
    end
end