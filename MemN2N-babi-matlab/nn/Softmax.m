% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Softmax < Module
    properties
        skip_bprop = false;
    end
    methods
        function obj = Softmax()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            input = bsxfun(@minus, input, max(input,[],1)) + 1;
            a = exp(input);
            obj.output = bsxfun(@rdivide, a, sum(a, 1));
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            if obj.skip_bprop == false
                z = bsxfun(@minus, grad_output, sum(obj.output .* grad_output));
                obj.grad_input = obj.output .* z;
            else
                obj.grad_input = grad_output;
            end
            grad_input = obj.grad_input;
        end
    end
end