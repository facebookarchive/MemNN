% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef MatVecProd < Module
    properties
        do_transpose;
    end
    methods
        function obj = MatVecProd(do_transpose)
            obj = obj@Module();
            obj.do_transpose = do_transpose;
        end
        function output = fprop(obj, input)
            M = input{1};
            V = input{2};
            if obj.do_transpose
                obj.output = zeros(size(M,2), size(M,3), 'single');
                for i = 1:size(M,3)
                    obj.output(:,i) = M(:,:,i)' * V(:,i);
                end
            else
                obj.output = zeros(size(M,1), size(M,3), 'single');
                for i = 1:size(M,3)
                    obj.output(:,i) = M(:,:,i) * V(:,i);
                end
            end
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            M = input{1};
            V = input{2};
            gradM = zeros(size(M), 'single');
            gradV = zeros(size(V), 'single');
            for i = 1:size(M,3)
                if obj.do_transpose
                    gradM(:,:,i) = V(:,i) * grad_output(:,i)';
                    gradV(:,i) = M(:,:,i) * grad_output(:,i);
                else
                    gradM(:,:,i) = grad_output(:,i) * V(:,i)';
                    gradV(:,i) = M(:,:,i)' * grad_output(:,i);
                end
            end
            obj.grad_input = {gradM, gradV};
            grad_input = obj.grad_input;
        end
    end
end