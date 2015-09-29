% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

rng('shuffle')
addpath nn;
addpath memory;
base_dir = 'tasks_1-20_v1-2 2/en/'; % path to data
t = 1; % task ID

% parse data
f = dir(fullfile(base_dir,['qa',num2str(t),'_*_train.txt']));
data_path = {fullfile(base_dir,f(1).name)};
f = dir(fullfile(base_dir,['qa',num2str(t),'_*_test.txt']));
test_data_path = {fullfile(base_dir,f(1).name)};
dict = containers.Map;
dict('nil') = 1;
[story, questions,qstory] = parseBabiTask(data_path, dict, false);
[test_story, test_questions, test_qstory] = parseBabiTask(test_data_path, dict, false);

% train and test
config_babi;
build_model;
if linear_start
    train_linear_start;
else
    train;
end
test;