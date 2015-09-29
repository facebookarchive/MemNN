% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef LookUpTable < Module
    properties
        sz;
        out_dim;
        weight;
    end
    methods
        function obj = LookUpTable(sz, output_dim)
            obj = obj@Module();
            obj.sz = sz;
            obj.out_dim = output_dim;
            obj.weight = Weight([output_dim, sz]);
        end
        function output = fprop(obj, input)
            obj.output = obj.weight.D(:,input(:));    
            obj.output = squeeze(reshape(obj.output, [obj.out_dim, size(input)]));
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output) 
%             input = reshape(input, 1, []);
%             for i = 1:size(input,2)
%                 k = input(i);
%                 obj.weight.grad(:,k) = obj.weight.grad(:,k) + grad_output(:,i);
%             end

            c = unique(input(:));
            for i = c'
                obj.weight.grad(:,i) = obj.weight.grad(:,i) + ...
                    sum(grad_output(:,input(:) == i), 2);
            end            
            
            obj.grad_input = [];
            grad_input = obj.grad_input;
        end
        function update(obj, params)
            obj.weight.update(params);
        end
        function share(obj, m)
            obj.weight = m.weight;
        end
    end
end