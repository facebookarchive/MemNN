-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('paths')

local stringx = require('pl.stringx')
local file = require('pl.file')

function g_read_words(fname, vocab, ivocab)
    local data = file.read(fname)
    local lines = stringx.splitlines(data)
    local c = 0
    for n = 1,#lines do
        local w = stringx.split(lines[n])
        c = c + #w + 1
    end
    local words = torch.Tensor(c, 1)
    c = 0
    for n = 1,#lines do
        local w = stringx.split(lines[n])
        for i = 1,#w do
            c = c + 1
            if not vocab[w[i]] then
                ivocab[#vocab+1] = w[i]
                vocab[w[i]] = #vocab+1
            end
            words[c][1] = vocab[w[i]]
        end
        c = c + 1
        words[c][1] = vocab['<eos>']
    end
    print('Read ' .. c .. ' words from ' .. fname)
    return words
end
