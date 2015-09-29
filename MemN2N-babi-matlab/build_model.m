% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

% construct model
if use_bow == false
    config.weight = ones(config.input_dim, config.max_words, 'single');
    for i = 1:config.input_dim
        for j = 1:config.max_words
            config.weight(i,j) = (i-(config.input_dim+1)/2)*(j-(config.max_words+1)/2);
        end
    end
    config.weight = 1 + 4 * config.weight / config.input_dim / config.max_words;
end

memory = {};
model = Sequential();
model.add(LookUpTable(config.voc_sz, config.input_dim));
if use_bow == false
    if enable_time
        model.add(ElemMult(config.weight(:,1:end-1)));
    else
        model.add(ElemMult(config.weight));
    end
end
model.add(Sum(2))
proj = {};
for i = 1:nhops
    if use_bow
        memory{i} = MemoryBoW(config);
    else
        memory{i} = MemoryL(config);
    end
    memory{i}.nil_word = dict('nil');
    model.add(Duplicate());
    P1 = Parallel();
    P1.add(memory{i});
    if add_proj
        proj{i} = LinearNB(config.input_dim,config.input_dim);
        P1.add(proj{i});        
    else
        P1.add(Identity());
    end
    model.add(P1);
    model.add(AddTable());
    if add_nonlin
        model.add(ReLU());
    end
end
model.add(LinearNB(config.out_dim, config.voc_sz, true));
model.add(Softmax());

% share weights
if share_type == 1
    memory{1}.emb_query.share(model.modules{1});
    for i = 2:nhops
        memory{i}.emb_query.share(memory{i-1}.emb_out);
    end
    model.modules{end-1}.share(memory{end}.emb_out);
elseif share_type == 2
    for i = 2:nhops
        memory{i}.emb_query.share(memory{1}.emb_query);
        memory{i}.emb_out.share(memory{1}.emb_out);
    end
end
if add_proj
    for i = 2:nhops
        proj{i}.share(proj{1});
    end
end
% cost
loss = CrossEntropyLoss();
loss.size_average = false;
loss.do_softmax_brop = true;
model.modules{end}.skip_bprop = true;