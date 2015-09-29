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

% parse data
data_path = {};
for t = 1:20
    f = dir(fullfile(base_dir,['qa',num2str(t),'_*_train.txt']));
    data_path{t} = fullfile(base_dir,f(1).name);        
end
dict = containers.Map;
dict('nil') = 1;
[story, questions,qstory] = parseBabiTask(data_path, dict, false);
for t = 1:20
    f = dir(fullfile(base_dir,['qa',num2str(t),'_*_test.txt']));
    test_data_path = {fullfile(base_dir,f(1).name)};
    [test_story, test_questions, test_qstory] = parseBabiTask(test_data_path, dict, false);
end

% train on all tasks
config_babi_joint;
build_model;
if linear_start
    train_linear_start;
else
    train_babi;
end

% test on each task
for t = 1:20
    f = dir(fullfile(base_dir,['qa',num2str(t),'_*_test.txt']));
    test_data_path = {fullfile(base_dir,f(1).name)};
    dc = dict.Count;
    [test_story, test_questions, test_qstory] = parseBabiTask(test_data_path, dict, false);
    assert(dc == dict.Count);
    disp(['task ', num2str(t)]);
    test;
end