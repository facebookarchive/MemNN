% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Parallel < Contrainer
    properties
    end
    methods
        function obj = Parallel()
            obj = obj@Contrainer();
        end
        function output = fprop(obj, input)
            obj.output = {};
            for i = 1:length(obj.modules)
                obj.output{i} = obj.modules{i}.fprop(input{i});
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            for i = 1:length(obj.modules)
                obj.grad_input{i} = obj.modules{i}.bprop(input{i}, grad_output{i});
            end
            grad_input = obj.grad_input;
        end
    end
end