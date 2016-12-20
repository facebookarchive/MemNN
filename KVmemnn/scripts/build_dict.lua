-- Copyright 2004-present Facebook. All Rights Reserved.
-- Build dictionary from training file(s).
-----------------------------------------------------
-- Arguments:
-- 1) Existing options, if any, to load.
local optclass = arg[1]
-- 2) Comma-separated list of input data text files.
local dpathin = arg[2]
-- 3) Path of output.
local dpathout = arg[3] or '/tmp/'
-- 4) Existing dictionary, if any, to add to.
local dictin = arg[4]
if dictin == '' then dictin = nil end


if dictin ~= nil then
    print('[loading existing dict from:' .. dictin .. ']')
else
    print('[not using an existing dictionary]')
end
print('[building dict from:' .. dpathin .. ']')
print('[building dict to:' .. dpathout .. ']')

require('torch')
local opt
if optclass ~= nil and optclass ~= '' then
    -- Kill all the args apart from the ones after the first 4 to pass
    -- them to the options class.
    local args = {}
    for i = 5, #arg do
        args[#args + 1] = arg[i]
    end
    arg = args
    opt = require(optclass)
else
    opt = {}
    opt.dictNumUNKs = 0
    opt.numThreads = 1
end
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local pl = require('pl.import_into')()
local tds = require('tds')
local parserlib = require('library.parse')
local parser = parserlib:new(opt)
local util = require('library.util')

-- local functions for multithreading
local function add_word(self, word, freq)
    if word ~= nil and word:len() > 0 then
        if self.dict[word] ~= nil then
            local start = word:sub(1,4)
            if start ~= '<NUL' and start ~= '<EOS' and start ~= '<UNK' then
                self.cnt[word] = self.cnt[word] + freq
            end
        else
            self.dictsz = self.dictsz + 1
            self.dict[word] = self.dictsz
            self.idict[self.dictsz] = word
            self.cnt[word] = freq
        end
    end
end

local function init(self, tds, parser)
    self.start_time = os.time()
    self.curr_time = os.time()
    self.dict = tds.hash()
    self.idict = tds.hash()
    self.cnt = tds.hash()
    self.dictsz = 0
    -- We hard code the NULL as the first item
    -- of the dictionary for convenience.
    local max = 10000000000
    add_word(self, '<NULL>', max + 2)
    add_word(self, '<EOS>', max + 1)
    add_word(self, '<UNK>', max)
    for i = 1, parser.opt.dictNumUNKs do
        add_word(self, '<UNK' .. i .. '>', max - i)
    end
    -- parser:set_add_callback(add_word, self)
end

local function process_str(self, s, pl, parser)
    -- First, get rid of the index at the beginning
    -- this doesn't go in the dictionary.
    local i1 = s:find(' ')
    if i1 ~= nil then
        s = s:sub(i1 + 1, -1)
    end

    -- Pull out x, y, and cands
    local all = pl.utils.split(s, '\t')
    local x = #all >= 1 and all[1]
    local y = #all >= 2 and all[2]
    -- don't process supporting facts (all[3])
    local c = #all >= 4 and all[4]

    -- Now go through the text finding n-grams.
    if x then parser:parse_internal(x, {}, self.dict, add_word, self) end
    if y then parser:parse_internal(y, {}, self.dict, add_word, self) end
    if c then
        local cands = pl.utils.split(c, '|')
        for _, c in pairs(cands) do
            parser:parse_internal(c, {}, self.dict, add_word, self)
        end
    end
end

local function load_dict(self, dictin, pl, parser, doPrint, zeroCounts)
    local beg = os.time()
    local f = io.open(dictin)
    if f == nil then
        print("[dict not found: " .. dictin .. "]")
        return
    end
    while true do
        local s = f:read("*line")
        if s == nil then break; end
        local w = pl.utils.split(s, '\t')
        local name = w[1]
        if name ~= nil and name ~= "" then
            if name:sub(1, 1) ~= '<' then
                if parser.opt.dictLowercaseData then
                    name = name:lower()
                end
                if name:sub(-1, -1) == ' ' then name = name:sub(1, -2); end
                if name:sub(-1, -1) == ' ' then name = name:sub(1, -2); end
                if name:sub(1, 1) == ' ' then name = name:sub(2, -1); end
                if name:sub(1, 1) == ' ' then name = name:sub(2, -1); end
            end
            local count = 0
            if #w == 2 then count = tonumber(w[2]); end
            add_word(self, name, zeroCounts and 0 or count)
        end
    end
    local fin = os.time()
    if doPrint then
        print(string.format(
            '[loaded: %d dict entries in %d seconds]', self.dictsz, fin - beg
        ))
    end
end

local build = {}

-- Calculate the number of lines.
function build:count(fname)
    local sz = 0
    for _ in io.lines(fname) do sz = sz + 1 end
    return sz
end

function build:add_word(word, freq)
    add_word(self, word, freq)
end

function build:process_str(s)
    process_str(self, s, pl, parser)
end

-- Build dictionary.
function build:process_singlethreaded(fname)
    if fname:sub(-4, -1) ~= '.txt' then
        print(string.format('WARNING: skipping %s, expected .txt file.', fname))
        return
    end
    local total_lines = self:count(fname)
    local lines = 0
    local start_time = os.time()
    for s in io.lines(fname) do
        build:process_str(s)
        lines = lines + 1
        util.log_progress(lines, total_lines, start_time, opt.logEveryNSecs)
    end
end

-- Build dictionary with multiple threads.
function build:process_multithreaded(fname, dictin)
    if fname:sub(-4, -1) ~= '.txt' then
        print(string.format('WARNING: skipping %s, expected .txt file.', fname))
        return
    end

    local pool = threads.Threads(opt.numThreads)

    local processed = 0
    local total_lines = self:count(fname)
    local job_size = math.ceil(total_lines / opt.numThreads)
    local jobs_rem = opt.numThreads
    for j = 1, opt.numThreads do
        local job_start = math.floor((j - 1) * job_size) + 1
        local job_end = math.min(math.floor(j * job_size), total_lines)
        pool:addjob(
            function(jobid)
                if opt.debugMode then
                    print(string.format(
                        'Job %d started processing lines %d-%d.',
                        jobid, job_start, job_end
                    ))
                end
                -- imports
                local parserlib = require('library.parse')
                local parser = parserlib:new(opt)
                local pl = require('pl.import_into')()
                local tds = require('tds')
                local util = require('library..util')

                local bld = {}
                init(bld, tds, parser)
                if dictin ~= nil then
                    load_dict(bld, dictin, pl, parser, jobid == 1, true)
                end

                local lines_processed = 0
                local read = assert(io.open(fname, 'r'))
                for i = 1, job_start - 1 do read:read('*line') end
                local start_time = os.time()
                for i = job_start, job_end do
                    local s = read:read('*line')
                    if s == nil then
                        break
                    elseif s:len() > 0 then
                        process_str(bld, s, pl, parser)
                    end
                    lines_processed = lines_processed + 1
                    if jobid == 1 then
                        util.log_progress(
                            lines_processed,
                            job_end - job_start + 1,
                            start_time,
                            opt.logEveryNSecs
                        )
                    end
                end

                return jobid, bld.dict, bld.cnt, lines_processed
            end,
            function(jobid, dict, cnt, lp)
                jobs_rem = jobs_rem - 1
                if jobid == 1 or opt.debugMode then
                    print(string.format(
                        'Finished thread %02d, waiting on %02d other threads.',
                        jobid, jobs_rem
                    ))
                end
                for w, _ in pairs(dict) do
                    self:add_word(w, cnt[w])
                end
                processed = processed + lp
            end,
            j
        )
    end
    pool:synchronize()
    pool:terminate()
    assert(processed == total_lines, 'Error: did not process all lines.')
    print('Finished building dictionary with ' .. self.dictsz  .. ' entries.')
end

function build:sort_dict()
    local tbl = {}
    for i = 1, self.dictsz do
        table.insert(tbl, {i, -self.cnt[self.idict[i]], self.idict[i]})
    end
    table.sort(tbl,
        function(t1, t2)
            local _a, b, c = table.unpack(t1)
            local _x, y, z = table.unpack(t2)
            if b ~= y then return b < y
            else return c < z end
        end
    )
    local ind = torch.Tensor(self.dictsz)
    for i, t in ipairs(tbl) do
        ind[i] = t[1]
    end
    self.freq_ind = ind
end

function build:save_dict(fname)
    -- Save out the dictionary.
    local fw = io.open(fname, 'w')
    assert(fw, 'Could not open file for writing. Check permissions/filename: '
               .. tostring(fname))
    local load_cnt = 0
    for i = 1, self.dictsz do
        if self.freq_ind ~= nil then
            local ind = self.freq_ind[i]
            local cnt = self.cnt[self.idict[ind]]
            if opt.dictMinOcc and cnt >= opt.dictMinOcc then
                load_cnt = load_cnt + 1
            end
            fw:write(self.idict[ind] .. '\t' .. cnt .. '\n')
        else
            -- Save unsorted dictionary.
            local cnt = self.cnt[self.idict[i]]
            if opt.dictMinOcc and cnt >= opt.dictMinOcc then
                load_cnt = load_cnt + 1
            end
            fw:write(self.idict[i] .. '\t' .. cnt .. '\n')
        end
    end
    if opt.dictMinOcc and opt.dictMinOcc > 0 then
        print(string.format(
            'INFO: With your current setting of dictMinOcc (%d), only %d items'
            .. ' will be loaded out of %d total.',
            opt.dictMinOcc, load_cnt, self.dictsz
        ))
    end
    fw:close()
end

function build:init()
    init(self, tds, parser)
end

function build:load_dict(dictin, print)
    load_dict(self, dictin, pl, parser, print)
end

function build:build()
    self:init()
    if dictin ~= nil then self:load_dict(dictin, true) end
    local t2 = os.time()
    -- Process the files.
    local files = pl.utils.split(dpathin, ',')
    for i,f in pairs(files) do
        print(i,f)
        if f:find('/%*') == nil then
            print('processing: ' .. f .. ' with '
                .. opt.numThreads .. ' thread(s).')
            if opt.numThreads and opt.numThreads > 1 then
                self:process_multithreaded(f, dictin)
            else
                self:process_singlethreaded(f)
            end
        else
            f = f:gsub('/%*', '/')
            for f2 in paths.files(f) do
                if f2:sub(-4) == ".txt" then
                    f2 = f .. f2
                    print('processing: ' .. f2 .. ' with '
                        .. opt.numThreads .. ' thread(s).')
                    if opt.numThreads and opt.numThreads > 1 then
                        self:process_multithreaded(f2, dictin)
                    else
                        self:process_singlethreaded(f2)
                    end
                end
            end
        end
    end
    local t3 = os.time()
    print(string.format('Processed all files in %d seconds.', t3 - t2))
    -- Sort the dictionary.
    if opt.dictSort then
        self:sort_dict()
    end
    -- And save it.
    if dpathout:sub(-4,-1) == '.txt' then
        self:save_dict(dpathout)
    else
        self:save_dict(dpathout .. '/dict.txt')
    end
    local t4 = os.time()
    print(string.format('Sorted and saved dictionary in %d seconds.', t4 - t3))
    return self.dict
end

-- In case you want to look at the results afterwards!!
g_build = build

-- Build the dictionary.
g_dict = build:build()
