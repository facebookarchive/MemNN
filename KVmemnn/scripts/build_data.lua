-- Copyright 2004-present Facebook. All Rights Reserved.
-- Build data from training file(s), given that the
-- dictionary is already built using build_dict.
--------------------------------------------------------------------------------
-- Arguments:
-- 1) Existing options, if any, to load.
local optclass = arg[1]
-- 2) Comma-separated list of input data text files.
local dpathin = arg[2]
-- 3) Path of output.
local dpathout = arg[3] or '/tmp/'
-- 4) Existing dictionary.
local dictin = arg[4]

print('[loading existing dict from:' .. dictin .. ']')
print('[building data from:' .. dpathin .. ']')
print('[building data to:' .. dpathout .. ']')

require('torch')
math.randomseed(os.time())
local opt
if optclass ~= nil then
    -- Kill all the args apart from the ones after the first 4 to pass
    -- them to the options class.
    local args = {}
    for i = 5, #arg do
        args[#args + 1] = arg[i]
    end
    arg = args
    opt = require(optclass)
end
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')
local tds = require('tds')
local pl = require('pl.import_into')()
local VectorArray = require(
    'library.vector_array')
local parserlib = require('library.parse')
local parser = parserlib:new(opt)
local util = require('library.util')

local function count_internal(s, parser, pl)
    local lines = 0
    local qfloats = 0
    local afloats = 0
    local totalsf = 1
    local totalcands = 0
    local candfloats = 0
    if s ~= nil and s:len() > 0 then
        lines = lines + 1
        -- process question
        local ind = s:find('\t')
        if ind ~= nil then
            local q = s:sub(1, ind - 1)
            s = s:sub(ind + 1, -1)
            local num_words = #parser:split(q,
                opt.dictWhiteSpace,
                opt.dictRemoveWhiteSpace
            )
            qfloats = qfloats + math.max(num_words, 1)

            -- process answer
            ind = s:find('\t')
            if ind ~= nil then
                local a = s:sub(1, ind - 1)
                s = s:sub(ind + 1, -1)
                local num_words = #parser:split(a,
                    opt.dictWhiteSpace,
                    opt.dictRemoveWhiteSpace
                )
                afloats = afloats + math.max(num_words, 1)

                -- process supporting facts
                ind = s:find('\t')
                if ind ~= nil then
                    local sf = s:sub(1, ind - 1)
                    s = s:sub(ind + 1, -1)
                    if sf:len() > 0 then
                        -- supporting facts are delimited by spaces
                        local num_facts = pl.stringx.count(sf, ' ') + 1
                        totalsf = num_facts
                    end

                    -- process candidates
                    local num_cands = 1 + pl.stringx.count(s, '|')
                    if num_cands > 0 then
                        totalcands = totalcands + num_cands
                        local num_words = #parser:split(s,
                            opt.dictWhiteSpace,
                            opt.dictRemoveWhiteSpace
                        )
                        candfloats = candfloats + num_words + num_cands
                    end
                else
                    if s:len() > 0 then
                        local num_facts = pl.stringx.count(s, ' ') + 1
                        totalsf = num_facts
                    end
                end
            else
                local num_words = #parser:split(s,
                    opt.dictWhiteSpace,
                    opt.dictRemoveWhiteSpace
                )
                afloats = afloats + math.max(num_words, 1)
            end
        else
            local num_words = #parser:split(s,
                opt.dictWhiteSpace,
                opt.dictRemoveWhiteSpace
            )
            qfloats = qfloats + math.max(num_words, 1)
            -- answer always contains at least the NULL symbol
            afloats = afloats + 1
        end
    end
    return lines, qfloats, afloats, totalsf, totalcands, candfloats
end

local build = {}
build.parser = parser

-- To load the dictionary.
function build:load_dict(fname)
    if fname == nil then return {}; end
    local f = io.open(fname)
    if f == nil then
        error("cannot load dictionary: " .. fname)
    end
    local dict = tds.hash()
    local cnt = 0
    while true do
        local s = f:read("*line")
        if s == nil then break; end
        local i1 = s:find('\t')
        s = s:sub(1, i1 - 1)
        cnt = cnt + 1
        dict[cnt] = s
        dict[s] = cnt
    end
    dict.num_symbols = cnt
    f:close()
    print('[loaded ' .. cnt .. ' dictionary entries.]')
    return dict
end

function build:count_internal(s)
    return count_internal(s, parser, pl)
end

-- Calculate the number of examples needed.
function build:count(fname)
    local lines = 0
    local qfloats = 0
    local afloats = 0
    local totalsf = 0
    local totalcands = 0
    local candfloats = 0
    -- check if multithreading is desired
    for s in io.lines(fname) do
        local l, q, a, sf, cc, cf = build:count_internal(s)
        lines = lines + l
        qfloats = qfloats + q
        afloats = afloats + a
        totalsf = totalsf + sf
        totalcands = totalcands + cc
        candfloats = candfloats + cf
    end
    return lines, qfloats, afloats, totalsf, totalcands, candfloats
end

local function gen_from_ex(ex, sizes, data, VA)
    if data.DSF == nil then
        data.DSF = VA:new(sizes.sfcnt, sizes.sfcnt, 1, false)
    end
    if data.DC == nil and ex[4] then
        data.DC = VA:new(
            sizes.candmsz, sizes.candcnt, 1, false
        )
        data.DCI = torch.FloatTensor(sizes.maxcnt, 2):fill(0)
        data.cand_cnt = 1
    end
end

local function process_ex(ex, data)
    if #ex[1] > 0 then
        if #ex[2] == 0 then
            -- Fill the example with the special NULL symbol.
            ex[2][#ex[2] + 1] = 1
        end
        local int = torch.FloatTensor(ex[1])
        local outt = torch.FloatTensor(ex[2])
        data.DX:add(int)
        data.DY:add(outt)
        data.cnt = data.cnt + 1
        local conv_index = ex[1][1]
        if conv_index == 1 then
            -- Start of a conversation. Record that.
            data.DST:add(torch.DoubleTensor({data.cnt}))
        end
        -- saving supporting facts
        if ex[3] and ex[3]:dim() > 0 then
            data.DSF:add(ex[3])
        else
            data.DSF:add(torch.FloatTensor({0}))
        end

        -- saving candidate sets
        if ex[4] then
            data.DCI[data.cnt][1] = data.cand_cnt
            for i = 1, #ex[4] do
                data.cand_cnt = data.cand_cnt + 1
                data.DC:add(ex[4][i])
            end
            data.DCI[data.cnt][2] = data.cand_cnt
        end
    end
end

local function save_vecarrays(fname_out, data)
    data.DX:clip()
    print('DX.len:size(1)', data.DX.len:size(1))
    print('DX.data:size(1)', data.DX.data:size(1))
    data.DY:clip()
    print('DY.len:size(1)', data.DY.len:size(1))
    print('DY.data:size(1)', data.DY.data:size(1))
    data.DST:clip()
    data.DX:save(fname_out .. '.x')
    data.DY:save(fname_out .. '.y')
    data.DST:save(fname_out .. '.st')
    data.DSF:clip()
    data.DSF:save(fname_out .. '.sf')
    if data.DC and data.DCI then
        data.DC:clip()
        data.DC:save(fname_out .. '.c')
        torch.save(fname_out .. '.ci', data.DCI)
    end
    -- In case you want to see the results afterwards!
    g_last_vecarray_x = data.DX
    g_last_vecarray_y = data.DY
    g_last_vecarray_st = data.DST
    if data.DSF then
        g_last_vecarray_sf = data.DSF
    end
    if data.DC and data.DCI then
        g_last_vecarray_c = data.DC
        g_last_vecarray_ci = data.DCI
    end
end

function build:multithreaded_onefile(fname_in, fname_out, dict)
    print('[build_data: ' .. fname_in .. ' with '
        .. opt.numThreads ..' threads ]')

    local total_examples = 0
    for line in io.lines(fname_in) do
        if line:sub(1, 2) == '1 ' then
            total_examples = total_examples + 1
        end
    end

    local job_seek_pos = {}
    local job_size = math.floor(total_examples / opt.numThreads)
    print('Total examples to process: ' .. total_examples)
    print('Allocating examples to threads...')

    local line
    local curr_ex = 1
    local j = 2
    local read = assert(io.open(fname_in, 'r'))
    job_seek_pos[1] = read:seek()
    repeat
        line = read:read('*line')
        while line ~= nil and line:sub(1, 2) ~= '1 ' do
            line = read:read('*line')
        end
        curr_ex = curr_ex + 1
        if curr_ex % job_size == 0 then
            -- We store the position where this line *began*
            job_seek_pos[j] = read:seek() - line:len() - 1
            j = j + 1
        end
    until line == nil or j == opt.numThreads + 1
    job_seek_pos[j] = read:seek('end')

    local mutex
    local mutex_id
    if opt.dictUseUNKHash then
        mutex = threads.Mutex()
        mutex_id = mutex:id()
    end
    local pool = threads.Threads(opt.numThreads)

    -- add jobs
    local results = {}
    local jobs_rem = opt.numThreads
    for j = 1, opt.numThreads do
        pool:addjob(
            function(jobid)
                if opt.debugMode then
                    print(string.format('Starting job %d.', jobid))
                end
                -- imports
                local parserlib =
                    require('library.parse')
                local parser = parserlib:new(opt)
                local pl = require('pl.import_into')()
                local VectorArray = require(
                    'library.vector_array')
                local util =
                    require('library.util')

                local mutex
                if opt.dictUseUNKHash then
                    local threads = require('threads')
                    mutex = threads.Mutex(mutex_id)
                end

                -- count properties for vector array allocation
                local sizes = {}
                sizes.maxcnt = 0
                sizes.maxqsz = 0
                sizes.maxasz = 0
                sizes.sfcnt = 0
                sizes.candcnt = 0
                sizes.candmsz = 0
                local read = assert(io.open(fname_in, 'r'))
                read:seek('set', job_seek_pos[jobid])
                while read:seek() < job_seek_pos[jobid + 1] do
                    local s = read:read('*line')
                    if s ~= nil and s:len() > 0 then
                        local n, q, a, sf, cc, cf =
                            count_internal(s, parser, pl)
                        sizes.maxcnt = sizes.maxcnt + n
                        sizes.maxqsz = sizes.maxqsz + q
                        sizes.maxasz = sizes.maxasz + a
                        sizes.sfcnt = sizes.sfcnt + sf
                        sizes.candcnt = sizes.candcnt + cc
                        sizes.candmsz = sizes.candmsz + cf
                    end
                end

                local data = {}
                data.DX = VectorArray:new(
                    sizes.maxqsz, sizes.maxcnt, 1, false
                )
                data.DY = VectorArray:new(
                    sizes.maxasz, sizes.maxcnt, 1, false
                )
                -- Starting indices of each "conversation" (begin with index 1).
                -- Need to be doubles otherwise precision is lost.
                data.DST = VectorArray:new(
                    sizes.maxcnt, sizes.maxcnt, 1, true
                )
                data.cnt = 0

                read:seek('set', job_seek_pos[jobid])
                local start_time = os.time()
                while read:seek() < job_seek_pos[jobid + 1] do
                    local s = read:read('*line')
                    if s ~= nil and s:len() > 0 then
                        if opt.dictUseUNKHash then mutex:lock() end
                        local ex = parser:parse(s, dict)
                        if opt.dictUseUNKHash then mutex:unlock() end
                        gen_from_ex(ex, sizes, data, VectorArray)
                        process_ex(ex, data)
                    end
                    if jobid == 1 then
                        util.log_progress(
                            data.cnt, sizes.maxcnt,
                            start_time, opt.logEveryNSecs
                        )
                    end
                end
                read:close()
                data.DX:clip()
                data.DY:clip()
                data.DST:clip()
                if data.DSF then
                    data.DSF:clip()
                end
                if data.DC then
                    data.DC:clip()
                end
                return jobid, data
            end,
            function(jobid, data)
                results[jobid] = data
                jobs_rem = jobs_rem - 1
                if jobid == 1 or opt.debugMode then
                    print(string.format(
                        'Finished thread %02d, waiting on %02d other threads.',
                        jobid, jobs_rem
                    ))
                end
            end,
            j
        )
    end
    pool:synchronize()
    pool:terminate()

    print('All threads finished, concatenating results.')

    -- calculate how big all the data from the threads is
    local maxcnt = 0
    for _, data in pairs(results) do
        maxcnt = maxcnt + data.DX.len:size(1)
    end

    local newData = results[1]
    setmetatable(newData.DX, {__index = VectorArray})
    setmetatable(newData.DY, {__index = VectorArray})
    setmetatable(newData.DST, {__index = VectorArray})
    if newData.DSF then setmetatable(newData.DSF, {__index = VectorArray}) end
    if newData.DC then setmetatable(newData.DC, {__index = VectorArray}) end
    local dci_offset = 0
    if newData.DCI then
        local oldDCI = newData.DCI
        local oldsz = oldDCI:size(1)
        newData.DCI = torch.FloatTensor(maxcnt, 2):fill(0)
        newData.DCI:narrow(1, 1, oldsz):add(oldDCI)
        dci_offset = oldsz
    end

    -- concatenate results from threads
    for ind, data in pairs(results) do
        collectgarbage()
        results[ind] = nil
        if ind > 1 then
            -- this aligns dialog start indices
            data.DST.data:add(newData.DX:size())

            newData.DX:add_vecarr(data.DX)
            newData.DY:add_vecarr(data.DY)
            newData.DST:add_vecarr(data.DST)
            if data.DSF then
                newData.DSF:add_vecarr(data.DSF)
            end
            if data.DC and data.DCI then
                -- this aligns candidate indices
                data.DCI:add(newData.DC:size())

                newData.DC:add_vecarr(data.DC)
                -- copy dci tensor
                newData.DCI:narrow(
                    1, dci_offset + 1, data.DCI:size(1)
                ):add(data.DCI)
                dci_offset = dci_offset + data.DCI:size(1)
            end
        end
    end
    save_vecarrays(fname_out, newData)
end

-- Build data for one file.
function build:singlethread_one_file(fname_in, fname_out, dict)
    print('[building task: ' .. fname_in .. ' with a single thread]')
    local sizes = {}
    sizes.maxcnt, sizes.maxqsz, sizes.maxasz, sizes.sfcnt,
        sizes.candcnt, sizes.candmsz = build:count(fname_in)
    print('Lines in data file:', sizes.maxcnt)
    print('X data points:', sizes.maxqsz)
    print('Y data points:', sizes.maxasz)
    print('Supporting facts data points:', sizes.sfcnt)
    print('Number of candidates:', sizes.candcnt)
    print('Candidate data points:', sizes.candmsz)

    local data = {}
    data.DX = VectorArray:new(sizes.maxqsz, sizes.maxcnt, 1, false)
    data.DY = VectorArray:new(sizes.maxasz, sizes.maxcnt, 1, false)
    -- Starting indices of each "conversation" (beginning with index 1).
    -- Need to be doubles otherwise precision is lost.
    data.DST = VectorArray:new(sizes.maxcnt, sizes.maxcnt, 1, true)
    data.DSF = VectorArray:new(sizes.sfcnt, sizes.sfcnt, 1, false)
    data.cnt = 0

    local start_time = os.time()
    for s in io.lines(fname_in) do
        if s:len() > 0 then
            local ex = self.parser:parse(s, dict)
            gen_from_ex(ex, sizes, data, VectorArray)
            process_ex(ex, data)
            util.log_progress(
                data.cnt, sizes.maxcnt, start_time, opt.logEveryNSecs
            )
        end
    end

    save_vecarrays(fname_out, data)
end

function build:print(x, dict)
    local s = ''
    for i = 1, x:size(1) do
        local word =  dict[x[i]]
        if word == nil then
            word = x[i]
            if g_dict._unk[word] ~= nil then
                word = word .. '=' .. g_dict._unk[word]
            end
        end
        s = s .. word .. '|'
    end
    print(s)
end

function build:process(f, dict)
    local f_split = pl.stringx.split(f, '/')
    local fout = dpathout .. '/' .. f_split[#f_split] .. '.vecarray'
    local tmp = io.open(fout)
    if tmp == nil then
        if opt.numThreads and opt.numThreads > 1 then
            build:multithreaded_onefile(f, fout, dict)
        else
            build:singlethread_one_file(f, fout, dict)
        end
    else
        print("[" .. fout .. " already exists.]")
        tmp:close()
    end
end

-- quick safety check to make sure dialog start indices are aligned properly
local function check_dialog_align()
    for n = 1, 100 do
        local ind = n
        if n > math.min(5, g_last_vecarray_st:size()) then
            ind = math.random(g_last_vecarray_st:size())
        end
        local ex_ind = g_last_vecarray_st:get(ind)[1]
        assert(
            g_last_vecarray_x:get(ex_ind)[1] == 1,
            string.format(
                'Dialog #%d (example #%d) has index %d. (Should be 1 always.)'
                .. ' Something broke during the build process.',
                ind,
                ex_ind,
                g_last_vecarray_x:get(ex_ind)[1]
            )
        )
    end
end

-- quick safety check to make sure candidate sets include the answer
local function check_candidates_answers()
    if g_last_vecarray_c then
        local ind = 0
        local num_checked = 0
        while num_checked < 15 do
            if num_checked < 3 then
                ind = ind + 1
            else
                ind = math.random(g_last_vecarray_y:size())
            end
            local ans = g_last_vecarray_y:get(ind)
            local ci = g_last_vecarray_ci[ind]
            local strt, fnsh = ci[1], ci[2]
            local found = true
            if fnsh - strt > 0 then
                found = false
                num_checked = num_checked + 1
                for i = strt, fnsh do
                    local c = g_last_vecarray_c:get(i)
                    if c:size(1) == ans:size(1)
                    and c:eq(ans):sum() == c:size(1) then
                        found = true
                        break
                    end
                end
            end
            assert(
                found,
                'Candidate set does not include answer at index ' .. ind
            )
        end
    end
end

local t1 = os.time()
g_dict = build:load_dict(dictin)
local t2 = os.time()
print(string.format('Loaded dictionary in %d seconds.', t2 - t1))
local files = pl.utils.split(dpathin, ',')
for i,f in pairs(files) do
    print(i,f)
    if f:find('/%*') == nil then
        print("processing:" .. f)
        build:process(f, g_dict)
        check_dialog_align()
        check_candidates_answers()
    else
        f = f:gsub('/%*', '/')
        for f2 in paths.files(f) do
            if f2:sub(-4) == ".txt" and f2 ~= 'dict.txt' then
                f2 = f .. f2
                print("processing:" .. f2)
                build:process(f2, g_dict)
                check_dialog_align()
                check_candidates_answers()
            end
        end
    end
end
local t3 = os.time()
print(string.format('Processed all data files in %d seconds.', t3 - t2))

-- If we are using unknown word hashes, we save that out as a separate set
-- as well.
if g_dict._unk ~= nil then
    local fw = io.open(dpathout .. '/dict_unkhash.txt', 'w')
    for i,k in pairs(g_dict._unk) do
        fw:write(i .. '\t' .. k .. '\n')
    end
    fw:close()
end

-- In case you want to look at the results afterwards!!
-- E.g. you can do:
--     g_build:print(g_last_vecarray_x:get(1), g_dict)
-- to look at one of the examples and see if it looks good.
-- NOTE: to debug hashing of unknowns, it is done with something like:
-- hash = require('hash')
-- hash.hash('hi')% 1e9 + g_dict.num_symbols) (and convert to float)
g_build = build

print("First example processed as:")
g_build:print(g_last_vecarray_x:get(1), g_dict)
g_build:print(g_last_vecarray_y:get(1), g_dict)
