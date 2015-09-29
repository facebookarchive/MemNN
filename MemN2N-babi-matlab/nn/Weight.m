% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Weight < handle
    properties
        sz
        D
        grad                
    end
    methods
        function obj = Weight(sz)
            obj.sz = sz;
            obj.D = 0.1 * randn(sz, 'single');
            obj.grad = zeros(sz, 'single');
        end
        function update(obj, params)
            if isfield(params, 'max_grad_norm') && params.max_grad_norm > 0
                if norm(obj.grad) > params.max_grad_norm
                    obj.grad = obj.grad * params.max_grad_norm / norm(obj.grad);
                end
            end
            obj.D = obj.D - params.lrate * obj.grad;
            obj.grad(:) = 0;
        end
        function m = clone(obj)
            m = Weight(obj.sz);
            m.D = obj.D;
            m.grad = obj.grad;
        end
    end
end