% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Contrainer < Module
    properties
        modules = {};
    end
    methods
        function obj = Contrainer()
            obj = obj@Module();
        end
        function add(obj, m)
            obj.modules{end+1} = m;
        end
        function update(obj, params)            
            for i = 1:length(obj.modules)
                obj.modules{i}.update(params);
            end
        end        
        function share(obj, m)
            for i = 1:length(obj.modules)
                obj.modules{i}.share(m.modules{i});
            end
        end
    end
end