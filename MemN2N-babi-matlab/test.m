% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

total_test_err = 0;
total_test_num = 0;
for k = 1:floor(size(test_questions,2)/batch_size)
    batch = (1:batch_size) + (k-1) * batch_size;
    input = zeros(size(story,1),batch_size,'single');
    target = test_questions(3,batch);
    input(:) = dict('nil');
    memory{1}.data(:) = dict('nil');
    for b = 1:batch_size
        d = test_story(:,1:test_questions(2,batch(b)),test_questions(1,batch(b)));
        d = d(:,max(1,end-config.sz+1):end);
        memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
        if enable_time
            memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict); % time words
        end
        input(1:size(test_qstory,1),b) = test_qstory(:,batch(b));
    end
    for i = 2:nhops
        memory{i}.data = memory{1}.data;
    end
    
    out = model.fprop(input);
    cost = loss.fprop(out, target);
    total_test_err = total_test_err + loss.get_error(out, target);
    total_test_num = total_test_num + batch_size;
end

test_error = total_test_err/total_test_num;
disp(['test error: ', num2str(test_error)]);
