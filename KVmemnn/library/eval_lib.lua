-- Copyright 2004-present Facebook. All Rights Reserved.
-- Main function to evaluate
require('torch')
local tds = require('tds')
local util = require('library.util')
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')
local eval = {}

function eval:create(mlp, opt)
    local new_eval = {}
    setmetatable(new_eval, { __index = self })
    new_eval.opt = opt
    new_eval.log = util.Log:new('eval', opt.debugMode)
    new_eval.mlp = mlp
    if opt.candidateFile ~= nil then
        new_eval:load_candidates()
    end
    new_eval:reset()
    return new_eval
end

function eval:reset()
    self.stats = {}
    self.stats.hpr_sizes = {1, 10, 100}
    self.stats.h = {}
    self.stats.r = {}
    self.stats.p = {}
    for i, j in pairs(self.stats.hpr_sizes) do
        self.stats.h[j] = 0
        self.stats.r[j] = 0
        self.stats.p[j] = 0
    end
    self.stats.mmr = 0
    self.stats.mrr = 0
    self.stats.map = 0
    self.stats.total = 0
    self.stats.ranks = {}
    self.stats.mem_clipped = 0
    self.stats.mem_empty = 0
    if self.opt.logPredictions then
        self.preds = tds.hash()
    end
    if self.opt.logResponses then
        self.resps = tds.hash()
    end
end

function eval:load_candidates()
    -- Load a candidate set to rank as negatives.
    local new_opt = {}
    for i,k in pairs(self.opt) do new_opt[i] = k; end
    new_opt.memHashFile = nil
    -- The candidate labels will be seen as inputs, so we shuffle this flag.
    new_opt.dictTFIDF = new_opt.dictTFIDFLabel
    local datalib = require(self.opt.dataClass)
    local cands_data =
        datalib:create_data(new_opt.candidateFile, nil,
                            new_opt, self.mlp.dict:get_shared())
    -- Store candidates in a tds hash ready for evaluation.
    local cands = tds.hash()
    while true do
        local ex, finished = cands_data:get_next_example()
        if finished then break; end
        cands[#cands + 1] = ex[1]:clone()
    end
    self.numFixedCands = #cands
    -- self.opt.evalParrotPenalty = 1
    if self.opt.evalParrotPenalty then
        for i = 1, self.opt.evalParrotPenalty do
            cands[#cands + 1] = self.mlp.NULL.x
        end
    end
    -- The last candidate is reserved for the correct answer.

    cands[#cands + 1] = self.mlp.NULL.x
    self.cands = cands
end

function eval:add_cmdline_options(opt)
    opt:option('maxExamplesEval', false, 'optional maximum number of '
                   .. 'examples to run evaluation on', 'eval')
end


function eval:print_save_predictions(ex, response, cands, scores,
                                     inds, min_rank, do_print)
    local mlp = self.mlp
    if not mlp.opt.dictFullLoading
        or (not do_print and not self.opt.debugMode
            and not self.opt.logPredictions and not self.opt.logResponses) then
            return
    end

    local txt_ex = ''
    local txt_pr = ''

    -- prefix each line in logs with "+" for correct predictions and
    -- "-" otherwise
    local pref = "- "
    if not min_rank then
        pref = ''
    elseif min_rank == 1 then
        pref = "+ "
    end

    if ex.memhx then
        for i = 1, #ex.memhx do
            txt_ex = txt_ex .. pref .. 'MHX' .. i .. ': '
                .. mlp.dict:vector_to_text(ex.memhx[i]) .. '\n'
            if ex.memhy[i] then
                txt_ex = txt_ex .. pref .. 'MHY' .. i .. ': '
                    .. mlp.dict:vector_to_text(ex.memhy[i]) .. '\n'
            end
            end
    end
    for i = 1, #ex.memx do
        txt_ex = txt_ex .. pref .. 'MX' .. i .. ': '
            .. mlp.dict:vector_to_text(ex.memx[i]) .. '\n'
        if ex.memy[i] then
            txt_ex = txt_ex .. pref .. 'MY' .. i .. ': '
                .. mlp.dict:vector_to_text(ex.memy[i]) .. '\n'
        end
    end
    txt_ex = txt_ex .. pref .. "Q: " ..
        mlp.dict:vector_to_text(ex[1]) .. '\n'
        .. pref .. "A: " .. mlp.dict:vector_to_text(ex[2])
    txt_pr = txt_pr .. pref .. "RESP:" ..
        mlp.dict:vector_to_text(response) .. '\n'
    -- Top 10 Predictions.
    for i = 1, math.min(10, inds:size(1)) do
        local name = self.mlp.dict:v2t(cands[inds[i]])
        local score = util.shortFloat(scores[inds[i]])
        txt_pr = txt_pr .. pref .. "PRED" .. i .. ": "
            .. name .. '\t[' .. inds[i] .. ',' .. score  .. ']\n'
    end
    txt_pr = txt_pr .. pref .. "min true rank: " .. min_rank .. '\n'

    if (do_print or self.opt.debugMode) and #txt_ex > 0 then
        print(txt_ex)
    end
    if (do_print or self.opt.debugMode) and #txt_pr > 0 then
        print(txt_pr)
    end
    if self.opt.logPredictions and #txt_pr > 0 then
        self.preds[#self.preds + 1] = txt_ex .. '\n' .. txt_pr
    end
    if self.opt.logResponses then
        local answers = ''
        if self.opt.rankLabelDocuments then
            -- logs question<tab>answer<tab>response
            answers = mlp.dict:v2t(ex[2])
        else
            -- logs question<tab>answer, answer, answer<tab>response
            for i = 1, ex[2]:size(1) do
                local word = ex[2][i][1]
                answers = answers .. ', ' .. mlp.dict:v2t(torch.Tensor{word})
            end
            answers = answers:sub(3, -1)
        end
        self.resps[#self.resps + 1] = mlp.dict:v2t(ex[1]) ..
            '\t' .. answers ..
            '\t' .. mlp.dict:v2t(response)
    end
end

function eval:eval_one_example(ex, do_print)
    if self.opt.onlyTestWithExWithCandidates and
    (ex.cands == nil or #ex.cands == 0) then
        if do_print then
            print "NO PREDICTION"
        end
        return
    end
    if self.opt.onlyTestOnIndex and
    ex.index ~= self.opt.onlyTestOnIndex then
            if do_print then
                print "NO PREDICTION"
            end
        return
    end
    local mlp = self.mlp
    if self.cands ~= nil then
        -- Replace the candidates with the candidates file.
        ex.cands = self.cands
        ex.numFixedCands = self.numFixedCands
        if self.opt.evalParrotPenalty then
            -- Add a parrot penalty.
            for i = 1, self.opt.evalParrotPenalty  do
                ex.cands[#ex.cands - i] = ex[1]
            end
        end
        if self.opt.addLabelAsLastCandidate then
        -- Add the true label as the last candidate.
            ex.cands[#ex.cands] = ex[2]
        end
    end
    local response, cands, scores = mlp:predict(ex)
    if cands == nil or #cands == 0 then
        if cands == nil then cands = {}; end
        cands[1] = response
        scores = torch.zeros(1)
    end
    if self.opt.evalDontScoreQueryWords then
        -- Blacklist words from the query itself.
        local x = ex[1]
        for i = 1, x:size(1) do
            local ind = x[i][1]
            scores[ind] = -1000
        end
    end
    local ranks = {}
    local min_rank = scores:size(1)
    local true_rank
    local vals, inds = torch.sort(scores, true)
    if (self.opt.useCandidateLabels or self.opt.testWithCandidateLabels)
    and #ex.cands > 0 then
        local ans = ex[2]
        if self.cands ~= nil and self.opt.addLabelAsLastCandidate then
            self.log:debug('using candidates from dataset')
            -- The last candidate is the correct answer in the case
            -- of a fixed candidate set for all test examples.
            _, true_rank = inds:eq(#self.cands):max(1)
            true_rank = true_rank[1]
            ranks[#ranks + 1] = true_rank
            min_rank = true_rank
        else
            -- Go through the candidates finding the right answer.
            self.log:debug('using candidates from model')
            for ci = 1, inds:size(1) do
                if util.same_tensor(cands[inds[ci]], ans) then
                    true_rank = ci
                    ranks[#ranks + 1] = true_rank
                    min_rank = true_rank
                    break
                end
            end
        end
    else
        self.log:debug('scoring the entire dictionary')
        -- Ranking the entire dictionary
        if inds:size(1) ~= mlp.dict.num_symbols then
            util.log_first_n(
                5, 'evaluation.lua',
                'warning: predictions are not over entire dictionary '
                    .. inds:size(1) .. ' vs. ' .. mlp.dict.num_symbols)
            min_rank = mlp.dict.num_symbols
        else
            for i = 1, ex[2]:size(1) do
                index = ex[2][i][1]
                _, true_rank = inds:eq(index):max(1)
                true_rank = true_rank[1]
                ranks[#ranks + 1] = true_rank
                if true_rank < min_rank then min_rank = true_rank; end
            end
        end
    end
    local stats = self.stats
    -- Hits calculation.
    for i in pairs(stats.h) do
        if min_rank <= i then
            stats.h[i] = stats.h[i] + 1
        end
    end
    -- Precision and Recall calculations.
    local p = {}
    local r = {}
    for _, j in pairs(stats.hpr_sizes) do
        r[j] = 0; p[j] = 0
    end
    for index = 1, #ranks do
        for i in pairs(r) do
            if ranks[index] <= i then
                r[i] = r[i] + 1; p[i] = p[i] + 1
            end
        end
    end
    for _, j in pairs(stats.hpr_sizes) do
        if r[j] > 0 and #ranks > 0 then
            stats.r[j] = stats.r[j] + (r[j] / #ranks)
        end
        stats.p[j] = stats.p[j] + (p[j] / j)
    end
    -- Average precision calculation.
    local p_avp = {}
    for _, j in pairs(ranks) do
        p_avp[j] = 0
    end
    for index = 1, #ranks do
        for i in pairs(p_avp) do
            if ranks[index] <= i then
                p_avp[i] = p_avp[i] + 1
            end
        end
    end
    local avp = 0
    for k in pairs(p_avp) do
        avp = avp + p_avp[k] / k
    end
    avp = avp / #ranks

    -- Mean Minimum rank calculation
    stats.mmr = stats.mmr + min_rank
    -- Mean Reciprocal Rank
    stats.mrr = stats.mrr + 1 / min_rank
    -- Mean Average Precision
    stats.map = stats.map + avp
    stats.total = stats.total + 1
    table.insert(stats.ranks, min_rank)
    if mlp.lastMemClipped then
        stats.mem_clipped = stats.mem_clipped + 1
    end
    if ex.memhx == nil or #ex.memhx == 0 then
        stats.mem_empty = stats.mem_empty + 1
    end

    -- potential debug printing
    self:print_save_predictions(ex, response, cands, scores, inds,
                                min_rank, do_print)
end

function eval:compute_median(stats)
    table.sort(stats.ranks)
    if #stats.ranks > 1 then
        return (stats.ranks[math.max(1, math.floor(#stats.ranks / 2))] +
                stats.ranks[math.floor(#stats.ranks / 2 + 1)]) / 2
    else return 0 end
end

function eval:metrics_to_string(metrics)
    local function shortstring(s)
        if type(s) == "number" then
            s = util.shortFloat(s)
        else
            s = tostring(s)
        end
        return s
    end
    local line =
        " tot:" .. shortstring(metrics['total'])
        .. " h1:" .. shortstring(metrics['h1'])
        .. " h10:".. shortstring(metrics['h10'])
        .. " h100:".. shortstring(metrics['h100'])
        .. " r1:" .. shortstring(metrics['r1'])
        .. " r10:".. shortstring(metrics['r10'])
        .. " r100:".. shortstring(metrics['r100'])
        .. " p1:" .. shortstring(metrics['p1'])
        .. " p10:".. shortstring(metrics['p10'])
        .. " p100:".. shortstring(metrics['p100'])
        .. " mmr:" .. shortstring(metrics['mmr'])
        .. " mrr:" .. shortstring(metrics['mrr'])
        .. " map:" .. shortstring(metrics['map'])
        .. " median:" .. shortstring(metrics['median'])
        .. " mem_clip:" .. shortstring(metrics['mem_clipped'])
        .. " mem_empty:" .. shortstring(metrics['mem_empty'])
    return line
end

function eval:calc_metrics()
    local metrics = {}
    metrics['total'] = self.stats.total
    for i, j in pairs(self.stats.hpr_sizes) do
        metrics['h' .. j] = self.stats.h[j] / self.stats.total
        metrics['r' .. j] = self.stats.r[j] / self.stats.total
        metrics['p' .. j] = self.stats.p[j] / self.stats.total
    end
    metrics['mmr'] = self.stats.mmr / self.stats.total
    metrics['mrr'] = self.stats.mrr / self.stats.total
    metrics['map'] = self.stats.map / self.stats.total
    metrics['median'] = self:compute_median(self.stats)
    metrics['mem_clipped'] = self.stats.mem_clipped / self.stats.total
    metrics['mem_empty'] = self.stats.mem_empty / self.stats.total
    return metrics
end

function eval:eval(test_data, max_exs, numThreads)
    numThreads = numThreads or self.opt.numThreads or 1
    self:reset()
    local totcnt = test_data:num_loaded_examples()
    if self.opt.maxExamplesEval then
        max_exs = self.opt.maxExamplesEval
    end
    if not max_exs or max_exs < 0 then max_exs = totcnt end
    test_data:reset_examples()

    print(string.format(
        '[evaluating up to %d examples with %d thread(s).]',
        max_exs,
        numThreads
    ))

    if numThreads and numThreads > 1 then
        return self:eval_multithreaded(test_data, max_exs, numThreads)
    else
        return self:eval_singlethreaded(test_data, max_exs, totcnt)
    end
end

function eval:eval_singlethreaded(test_data, max_exs, totcnt)
    local cnt = 0
    local start_time = os.time()
    local last_log_time = os.time()
    self.mlp:reset_prediction_cache()
    while cnt < totcnt and self.stats.total < max_exs do
        local ex, finished = test_data:get_next_example()
        if finished then break end -- No more data.
        cnt = cnt + 1
        if not test_data:is_null(ex[2]) then
            self:eval_one_example(ex)
        end
        local time = os.time()
        local since_last_log = time - last_log_time
        if since_last_log >= self.opt.logEveryNSecs then
            util.log_progress(cnt, totcnt, start_time)
            last_log_time = os.time()
            local metrics = self:calc_metrics()
            local line = self:metrics_to_string(metrics)
            print(line)
        end
    end
    local metrics = self:calc_metrics()
    local line = self:metrics_to_string(metrics)
    print(line)

    return metrics, self.stats
end

function eval:eval_multithreaded(test_data, max_exs, numThreads)
    local opt = self.opt
    opt.threadsShareData = true
    local evalClass = self.opt.evalClass
    local dataClass = self.opt.dataClass
    local modelClass = self.opt.modelClass
    local testData = opt.testData
    local shared_data = test_data:get_shared()
    local shared_mlp = self.mlp:get_shared()
    -- sometimes dict get_shared has been lost
    if self.mlp.dict.get_shared == nil then
        setmetatable(self.mlp.dict, {__index = require(opt.dictClass)})
    end
    local shared_dict = self.mlp.dict:get_shared()
    local pool = threads.Threads(
        numThreads,
        function(idx)
            require('torch')
            g_create_mlp_class = require(modelClass)
        end
    )

    local jobs_rem = numThreads
    local job_size = math.ceil(max_exs / numThreads)
    for j = 1, numThreads do
        pool:addjob(
            function(jobid)
                -- set up libraries and modules
                local evallib = require(evalClass)
                local datalib = require(dataClass)
                local mlp = g_create_mlp_class:clone_mlp(shared_mlp, opt)
                mlp:reset_prediction_cache()
                local eval = evallib:create(mlp, opt)
                local test_data = datalib:create_data(
                    testData, shared_data, opt, shared_dict)
                local util = require('library.util')

                -- process data
                local cnt = 0
                local start_time = os.time()
                local last_log_time = start_time
                while true do
                    local ex, finished = test_data:get_next_example_partitioned(
                        jobid, numThreads, max_exs)
                    if finished then break end -- No more data.
                    cnt = cnt + 1
                    if not test_data:is_null(ex[2]) then
                        eval:eval_one_example(ex)
                    end
                    if jobid == 1 then
                        local time = os.time()
                        local since_last_log = time - last_log_time
                        if since_last_log >= opt.logEveryNSecs then
                            util.log_progress(cnt, job_size, start_time)
                            last_log_time = os.time()
                            local metrics = eval:calc_metrics()
                            local line = eval:metrics_to_string(metrics)
                            print('Results from thread 1:' .. line)
                        end
                    end
                end
                return jobid, eval
            end,
            function(jobid, res)
                for k, v in pairs(res.stats) do
                    if type(v) == 'number' then
                        self.stats[k] = self.stats[k] + v
                    end
                end
                for _, j in pairs(self.stats.hpr_sizes) do
                    self.stats.h[j] = self.stats.h[j] + res.stats.h[j]
                    self.stats.r[j] = self.stats.r[j] + res.stats.r[j]
                    self.stats.p[j] = self.stats.p[j] + res.stats.p[j]
                end
                for _, r in pairs(res.stats.ranks) do
                    table.insert(self.stats.ranks, r)
                end
                if opt.logPredictions then
                    for _, pr in pairs(res.preds) do
                        self.preds[#self.preds + 1] = pr
                    end
                end
                if opt.logResponses then
                    for _, pr in pairs(res.resps) do
                        self.resps[#self.resps + 1] = pr
                    end
                end
                if jobid == 1 or opt.debugMode then
                    jobs_rem = jobs_rem - 1
                    print(string.format(
                        'Finished job %02d, %02d jobs remaining.',
                        jobid, jobs_rem
                    ))
                end
            end,
            j
        )
    end
    pool:synchronize()
    pool:terminate()

    local metrics = self:calc_metrics()
    local line = self:metrics_to_string(metrics)
    print('Finished combining thread results, final stats are:' .. line)

    return metrics, self.stats
end

function eval:save_metrics(filename, metrics, comment)
    local s = ''
    if comment ~= nil then
        s = s .. comment .. ' '
    end
    s = s .. os.date() .. ' '
    s = s .. self:metrics_to_string(metrics)
    local fw, err = io.open(filename, "a")
    if fw == nil then
        print("writing file failed: " .. filename)
        error('writing  failed! ' .. tostring(err))
    end
    fw:write(s .. "\n")
    fw:close()
    print("[saved eval metrics:" .. filename .. "]")
    if self.opt.logPredictions then
        local filenamep = filename .. '_predictions'
        local fw, err = io.open(filenamep, "w")
        if fw == nil then
            print("writing file failed: " .. filenamep)
            error('writing  failed! ' .. tostring(err))
        end
        for i = 1, #self.preds do
            fw:write(i .. ' ---------- \n' .. self.preds[i])
        end
        fw:close()
        print("[saved eval predictions: " .. filenamep .. "]")
    end
    if self.opt.logResponses then
        local filenamep = filename .. '_responses'
        local fw, err = io.open(filenamep, "w")
        if fw == nil then
            print("writing file failed: " .. filenamep)
            error('writing  failed! ' .. tostring(err))
        end
        for i = 1, #self.resps do
            fw:write(self.resps[i] .. '\n')
        end
        fw:close()
        print("[saved eval responses: " .. filenamep .. "]")
    end
end

-- functions below for validating while training
function g_eval_save_metrics(_, val_metrics, filename, comment, validMetric)
    validMetric = validMetric or 'h10'
    local s = os.date()
    s = s .. " " .. validMetric .. ": " .. val_metrics[validMetric]
    if file_extension == nil then
        file_extension = ".eval"
    end
    if file_extension:sub(1,1) ~= "." then
        file_extension = "." .. file_extension
    end
    local f = io.open(filename, "a")
    if comment == nil then comment = ''; end
    f:write(s .. " " .. comment .. "\n")
    f:close()
end

function g_eval_data_set(filename, dataset, mlp)
    local tmp_eval = g_eval_class:create(mlp, mlp.opt)
    local val_metrics, stats = tmp_eval:eval(dataset, mlp.opt.evalMaxTestEx, 1)
    return val_metrics, stats
end

return eval
