% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Linear < Module
    properties
        in_dim;
        out_dim;
        weight;
        bias;
    end
    methods
        function obj = Linear(input_dim, output_dim)
            obj = obj@Module();
            obj.in_dim = input_dim;
            obj.out_dim = output_dim;
            obj.weight = Weight([output_dim, input_dim]);
            obj.bias = Weight([output_dim, 1]);
        end
        function output = fprop(obj, input)
            if ndims(input) > 2
                sz = size(input);
                input = reshape(input, size(input,1),[]);
            end
            obj.output = obj.weight.D * input;
            obj.output = bsxfun(@plus, obj.output, obj.bias.D);
            if exist('sz') == 1
                sz(1) = size(obj.output,1);
                obj.output = reshape(obj.output, sz);
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            if ndims(input) > 2
                sz = size(input);
                input = reshape(input, size(input,1),[]);
                grad_output = reshape(grad_output, size(grad_output,1),[]);
            end
            obj.weight.grad = obj.weight.grad + grad_output * input';
            obj.bias.grad = obj.bias.grad + sum(grad_output,2);
            obj.grad_input = obj.weight.D' * grad_output;
            if exist('sz') == 1
                obj.grad_input = reshape(obj.grad_input, sz);
            end
            grad_input = obj.grad_input;
        end
        function update(obj, params)
            obj.weight.update(params);
            obj.bias.update(params);
        end
        function share(obj, m)
            obj.weight = m.weight;
            obj.bias = m.bias;
        end
    end
end