% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef MemoryL < Memory
    properties
    end
    methods
        function obj = MemoryL(config)
            obj = obj@Memory(config);
            obj.data = zeros(config.max_words, obj.sz, config.bsz, 'single');
        end
        function initQueryModule(obj)
            obj.emb_query = LookUpTable(obj.voc_sz ,obj.in_dim);            
            S = Sequential();
            S.add(obj.emb_query);
            S.add(ElemMult(obj.config.weight))
            S.add(Sum(2));
            P = Parallel();
            P.add(S);
            P.add(Identity());
            obj.mod_query = Sequential();
            obj.mod_query.add(P);            
            obj.mod_query.add(MatVecProd(true));            
            obj.mod_query.add(Softmax());            
        end
        function initOutputModule(obj)
            obj.emb_out = LookUpTable(obj.voc_sz ,obj.out_dim);            
            S = Sequential();
            S.add(obj.emb_out);
            S.add(ElemMult(obj.config.weight))
            S.add(Sum(2));
            P = Parallel();
            P.add(S);
            P.add(Identity());
            obj.mod_out = Sequential();
            obj.mod_out.add(P);            
            obj.mod_out.add(MatVecProd(false));            
        end
    end
end