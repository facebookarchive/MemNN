% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Memory < Module
    properties
        sz;
        voc_sz;
        in_dim;
        out_dim;
        data;
        emb_query;
        emb_out;
        mod_query;
        mod_out;
        probs;
        nil_word;
        config;        
    end
    methods
        function obj = Memory(config)
            obj = obj@Module();
            obj.sz = config.sz;
            obj.voc_sz = config.voc_sz;
            obj.in_dim = config.input_dim;
            obj.out_dim = config.out_dim;            
            obj.nil_word = config.voc_sz;
            obj.config = config;
            
            obj.data = zeros(obj.sz, config.bsz, 'single');
            obj.initQueryModule();
            obj.initOutputModule();
        end
        function initQueryModule(obj)
            obj.emb_query = LookUpTable(obj.voc_sz ,obj.in_dim);            
            P = Parallel();
            P.add(obj.emb_query);
            P.add(Identity());
            obj.mod_query = Sequential();
            obj.mod_query.add(P);            
            obj.mod_query.add(MatVecProd(true));            
            obj.mod_query.add(Softmax());            
        end
        function initOutputModule(obj)
            obj.emb_out = LookUpTable(obj.voc_sz ,obj.out_dim);            
            P = Parallel();
            P.add(obj.emb_out);
            P.add(Identity());
            obj.mod_out = Sequential();
            obj.mod_out.add(P);            
            obj.mod_out.add(MatVecProd(false));            
        end
        function reset(obj)
            obj.data(:) = obj.nil_word;
        end
        function put(obj, data)
            obj.data(2:end,:) = obj.data(1:end-1,:);           
            obj.data(1,:) = data;
        end
        function output = fprop(obj, input)
            obj.probs = obj.mod_query.fprop({obj.data, input});
            obj.output = obj.mod_out.fprop({obj.data, obj.probs});
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            g1 = obj.mod_out.bprop({obj.data, obj.probs}, grad_output);
            g2 = obj.mod_query.bprop({obj.data, input}, g1{2});
            obj.grad_input = g2{2};
            grad_input = obj.grad_input;
        end
        function update(obj, params)
            obj.mod_out.update(params);
            obj.mod_query.update(params);
            obj.emb_out.weight.D(:,obj.nil_word) = 0;
        end
    end
end