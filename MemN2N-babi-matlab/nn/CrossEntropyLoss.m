% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef CrossEntropyLoss < Loss
    properties
        do_softmax_brop = false;
        eps = 0.0000001;
        size_average = true;
    end
    methods
        function obj = CrossEntropyLoss()
            obj = obj@Loss();
        end
        function cost = fprop(obj, input, target)
            z = sub2ind(size(input), target, 1:length(target));
            cost = sum(-log(input(z)));
            if obj.size_average
                cost = cost / size(input,2);
            end
        end
        function grad_input = bprop(obj, input, target)
            z = sub2ind(size(input), target, 1:length(target));
            if obj.do_softmax_brop
                % better numberical stability
                grad_input = input; 
                grad_input(z) = grad_input(z) - 1;
            else
                grad_input = zeros(size(input), 'single');
                grad_input(z) = -1./(input(z) + obj.eps);
            end
            if obj.size_average
                grad_input = grad_input / size(input,2);
            end
        end
        function error = get_error(obj, input, target)
            [~,y] = max(input,[],1);
            error = sum(y ~= target);
        end
    end    
end