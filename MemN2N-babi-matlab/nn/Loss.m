% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Loss < handle
    properties
    end
    methods
        function obj = Loss()
            obj = obj@handle();
        end
        function cost = fprop(obj, input, target)
            assert(false, obj.name)
        end
        function grad_input = bprop(obj, input, target)
            assert(false, obj.name)
        end
    end
end