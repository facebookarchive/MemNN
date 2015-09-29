% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

batch_size = 32;
nhops = 3;
nepochs = 60;
lrate_decay_step = 15;

rp = randperm(size(questions,2));
train_range = rp(1:floor(0.9 * size(questions,2)));
val_range = rp((floor(0.9 * size(questions,2))+1):size(questions,2));
enable_time = true;
use_bow = false;
share_type = 1;
randomize_time = 0.1;
add_proj = false;
add_nonlin = false;

config = {};
config.init_lrate = 0.01;
config.max_grad_norm = 40;
config.input_dim = 50;
config.out_dim = 50;

linear_start = true;
if linear_start
    ls_nepochs = 30;
    ls_lrate_decay_step = 31;
    ls_init_lrate = 0.01/2;
    config.init_lrate = 0.01/2;
end

config.sz = min(50, size(story,2));
config.voc_sz = length(dict);
config.bsz = batch_size;
config.max_words = size(story,1);
if enable_time 
   config.voc_sz = config.voc_sz + config.sz;
   config.max_words = config.max_words + 1; % +1 for time words
end