% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

params = {};
params.lrate = config.init_lrate;
params.max_grad_norm = config.max_grad_norm;
for ep = 1:nepochs
    if mod(ep, lrate_decay_step) == 0
        params.lrate = params.lrate * 0.5;
    end
    total_err = 0;
    total_cost = 0;
    total_num = 0;
    for k = 1:floor(length(train_range)/batch_size)
        batch = train_range(randi(length(train_range), batch_size,1));
        input = zeros(size(story,1),batch_size,'single');
        target = questions(3,batch);
        memory{1}.data(:) = dict('nil');
        offset = zeros(1,batch_size,'single');
        for b = 1:batch_size
            d = story(:,1:questions(2,batch(b)),questions(1,batch(b)));
            offset(b) = max(0,size(d,2)-config.sz);
            d = d(:,1+offset(b):end);
            memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
            if enable_time
                if randomize_time > 0
                    nblank = randi([0,ceil(size(d,2) * randomize_time)]);
                    rt = randperm(size(d,2) + nblank);
                    rt(rt > config.sz) = config.sz; % vocabulary limit
                    memory{1}.data(end,1:size(d,2),b) = sort(rt(1:size(d,2)),'descend') + length(dict);
                else
                    memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict);
                end
            end
            input(:,b) = qstory(:,batch(b));
        end
        for i = 2:nhops
            memory{i}.data = memory{1}.data;
        end
        
        out = model.fprop(input);
        total_cost = total_cost + loss.fprop(out, target);
        total_err = total_err + loss.get_error(out, target);
        total_num = total_num + batch_size;
        grad = loss.bprop(out, target);
        model.bprop(input, grad);
        model.update(params);            
        for i = 1:nhops
            memory{i}.emb_query.weight.D(:,1) = 0;
        end
    end
            
    total_val_err = 0;
    total_val_cost = 0;
    total_val_num = 0;
    for k = 1:floor(length(val_range)/batch_size)
        % do validation
        batch = val_range((1:batch_size) + (k-1) * batch_size);
        input = zeros(size(story,1),batch_size,'single');
        target = questions(3,batch);
        memory{1}.data(:) = dict('nil');
        for b = 1:batch_size
            d = story(:,1:questions(2,batch(b)),questions(1,batch(b)));
            d = d(:,max(1,size(d,2)-config.sz+1):end);
            memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
            if enable_time
                memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict);
            end
            input(:,b) = qstory(:,batch(b));
        end
        for i = 2:nhops
            memory{i}.data = memory{1}.data;
        end
        
        out = model.fprop(input);
        total_val_cost = total_val_cost + loss.fprop(out, target);
        total_val_err = total_val_err + loss.get_error(out, target);
        total_val_num = total_val_num + batch_size;
    end    
    train_error = total_err/total_num;
    val_error = total_val_err/total_val_num;
    disp([num2str(ep), ' | train error: ', num2str(train_error), ' | val error: ', num2str(val_error)]);
end