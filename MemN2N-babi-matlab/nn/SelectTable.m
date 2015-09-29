% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef SelectTable < Module
    properties
        index
    end
    methods
        function obj = SelectTable(index)
            obj = obj@Module();
            obj.index = index;
        end
        function output = fprop(obj, input)
            obj.output = input{obj.index};
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(input)
                if i == obj.index
                    obj.grad_input{i} = grad_output;
                else
                    obj.grad_input{i} = zeros(size(input{i}), 'single');
                end
            end
            grad_input = obj.grad_input;
        end
    end
end