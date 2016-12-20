-- Copyright 2004-present Facebook. All Rights Reserved.
require('torch')
require('nn')
require('sys')
require('math')
require('os')
local util = require('library.util')

local baseNN = {}

baseNN.modelClass = 'library.base_model'

function baseNN:add_cmdline_options(opt)
    -- Model
    opt:option('modelFilename',
               nil,
               "file name where we save the model, " ..
               "if it exists we load it before training.", 'model')
    opt:option('continueTraining', false,
               'if true, allows to load a model, and continue training '
               .. 'otherwise training exits if the training log already exists',
               'model')
    opt:option('initModelFile',
               nil,
               "file name of a model we initialize *our* model with"
                   .. "when we first begin training.", 'model', true)
    opt:option('useDoubles', false, 'use doubles or floats', 'model',
               true)
    opt:option('numThreads', 1, 'number of threads; must be >= 1.', 'model')

    -- Dictionary
    opt:option('dictFile', '', 'Dictionary file mapping symbols to indices',
               'dict')
    opt:option('dictClass', 'library.dict',
               'dictionary class', 'dict', true)
    opt:option('dictFullLoading', true,
               'Whether to load the dictionary word to index map'
                   .. '(takes a little extra memory)',
               'dict')
    opt:option('dictSave', false, 'save dictionary with model', 'dict', true)
    opt:option('dictSort', true, 'sort dictionary', 'dict', true)

    -- Training and Logs
    opt:option('allowLoading', true, 'allow loading', 'model')
    opt:option('allowSaving', true, 'allow saving', 'model')
    opt:option('maxTrainTime', 60 * 60 * 24 * 30, -- default : one month
               'maximum training time in secs',
               'model')
    opt:option('maxTrainEpochs', 60 * 24 * 30, -- default: epoch/min for 1 month
               'maximum number of training examples to consider',
               'model')
    opt:option('logEveryNSecs', 60, 'log training stats every N secs', 'log',
               true)
    opt:option('saveEveryNSecs', 60 * 15, 'save model every N secs', 'log',
               true)
    opt:option('logTrainingFile', true,
               'log training stats to <modelFilename>.log',
               'log', true)
    opt:option('debugMode', false,
               'prints out various debugging information, '
               .. 'larger values print more',
               'log', true)
    opt:option('profi', false, 'profile using ProFi', 'log', false)

    -- Validation options.
    opt:option('evalClass',
               'library.eval_lib',
               'evaluation class to use', 'valid', true)
    opt:option('testData', nil, 'test data file', 'valid', true)
    opt:option('evalFileSuffix', '', 'appended to eval filename',
               'valid', 'true')
    opt:option('evalIgnoreMissingModelFile', false, '', 'valid', true)
    opt:option('evalDataClass', nil,
               'data class to use for validation', 'valid')
    opt:option('testWithCandidateLabels', false,
               'run validation using the candidates attached to examples',
               'eval', true)
    opt:option('onlyTestWithExWithCandidates', false,
               'if true, only compute eval '
                   .. 'on examples with candidate labels', 'valid', true)
    opt:option('onlyTestOnIndex', false,
               'if true, only compute eval '
                   .. 'on question with given index', 'valid', true)
    opt:option('evalMaxTestEx', false,
               'max number of test examples to eval on',
               'valid', true)
    opt:option('candidateFile', nil,
               'if specified used as a set of negatives for ranking loss',
               'valid')
    opt:option('addLabelAsLastCandidate', true, 'when a candidateFile is ' ..
                   'used add the label as the last candidate',
               'valid')
    opt:option('validData',
               nil,
               "filename of validation data, uses train data if set to 'train'",
               'valid', true)
    opt:option('validEveryNSecs', 60 * 30,
               'validate model every N secs', 'valid', true)
    opt:option('validMetric', 'p1',
               'name of metric to measure for validation', 'valid', true)
    opt:option('validOptMetric', 'max',
               'whether we are maximizing of minimizing the metric', 'valid',
               true)
    opt:option('printEveryNEvals', 10000,
               'print a log of the eval metrics every N examples', 'valid',
               true)
    opt:option('robustFileIO', false,
               'if true, try to carry on if file IO is failing', 'data',
               true)
    opt:option('sweeper', false, 'print json logs to be read by aisweeper',
               'valid', true)
    opt:option('logPredictions', false, 'save the predictions to a file',
               'valid')
    opt:option('logResponses', false,
               'during evaluation, save question<tab>answer<model reply> ' ..
               'triplets to file for each test example', 'valid')
end

function baseNN:init_mlp(options)
    local new_mlp
    if options.modelFilename == nil then
        new_mlp = self:create_mlp(options)
    else
        new_mlp = {}
        setmetatable(new_mlp, { __index = self })
        new_mlp.opt = options
        -- Possibly load a previous checkpoint.
        if not new_mlp:load(options.modelFilename) then
            print("[couldn't find mlp at '" .. options.modelFilename .. "']")
            if options.initModelFile == nil then
                print('[so, start training mlp from scratch.]')
                new_mlp = self:create_mlp(options)
            else
                print('[trying to initialize with: ' ..
                          options.initModelFile .. ']')
                if new_mlp:load(options.initModelFile) then
                    print('[..done]')
                else
                    error('failed to initialize')
                end
            end
        end
    end
    return new_mlp
end

function baseNN:Tensor(...)
    if self.opt.useDoubles then
        return torch.DoubleTensor(...)
    else
        return torch.FloatTensor(...)
    end
end

function baseNN:clone_mlp(shared_mlp, options)
    return baseNN:create_mlp(options)
end

function baseNN:create_mlp(options)
    error('not defined!')
end

-- Reset gets called on the model after being created, cloned or loaded
function baseNN:reset()
    torch.setdefaulttensortype("torch.FloatTensor")
    self.NULL = {}
    self.NULL.x = torch.Tensor({1})  -- special NULL character.
    self.NULL.index = 0
    self.log = util.Log:new('model', self.opt.debugMode)
end

-- Train one worker, can be used for hogwild training
-- or single-threaded.
function baseNN:do_train_worker()
    local exs_processed = 0
    local its = 0
    local loss = {}
    local batch = 0
    local start_time = sys.clock()
    local last_save_time = sys.clock()
    local last_log_time = sys.clock();
    local last_valid_time = sys.clock();
    local best_valid_metric
    local last_valid_metric
    while true do
        -- Train on an example.
        local ex = g_train_data:get_random_example()
        exs_processed = exs_processed + 1
        if ex ~= nil then
            local update_loss = self:update(ex)
            if update_loss ~= nil then
                for i, k in pairs(update_loss) do
                    if loss[i] == nil then loss[i] = 0; end
                    loss[i] = loss[i] + k
                end
                batch = batch + 1
            end
        end
        -- Logging of various types follows.
        local time = sys.clock()
        local total_time = time - start_time
        if self.opt.maxTrainTime ~= nil then
            if total_time > self.opt.maxTrainTime then break end
        end
        local num_epochs = exs_processed / g_train_data:num_loaded_examples()
        if self.opt.maxTrainEpochs ~= nil then
            if num_epochs > self.opt.maxTrainEpochs then break end
        end
        -- Save the model every N seconds.
        local save_time = time - last_save_time
        if (self.my_threadidx == 1 and self.opt.saveEveryNSecs ~= nil and
            save_time >= self.opt.saveEveryNSecs) then
            if self.opt.modelFilename ~= nil then
                self:save_all(self.opt.modelFilename)
            end
            last_save_time = sys.clock();
        end

        last_valid_time, best_valid_metric, last_valid_metric =
            self:validate_while_training(
                g_valid_data, best_valid_metric, last_valid_metric,
                time, last_valid_time)

        -- Log training error every N seconds.
        local log_time = time - last_log_time
        if (log_time >= self.opt.logEveryNSecs) then
            its = its + batch
            local thread_string = ''
            local doPrint = true
            if self.opt.numThreads > 1 then
                thread_string = 'thread:' .. self.my_threadidx .. ' ';
                if not self.opt.debugMode and self.my_threadidx > 1 then
                    doPrint = false;
                end
            end
            if doPrint then
                local log_string = '[' .. thread_string
                    .. 'exs:' .. its
                    .. ' epoch:' .. util.shortFloat(num_epochs)
                if loss.hinge_loss then
                    log_string = log_string .. ' hinge_loss:'
                        .. util.shortFloat(loss.hinge_loss / batch)
                end
                if loss.r then
                    log_string = log_string .. ' resp_loss:'
                        .. util.shortFloat(loss.r / batch)
                end
                if loss.rank_loss then
                    log_string = log_string .. ' rank_loss:'
                        .. util.shortFloat(loss.rank_loss / batch)
                end
                if loss.neg_sample_errs then
                    log_string = log_string .. ' neg_errs:'
                        .. util.shortFloat(loss.neg_sample_errs / batch)
                end
                if loss.mean_rank then
                    log_string = log_string .. ' mean_rank:'
                        .. util.shortFloat(loss.mean_rank / batch)
                end
                log_string = log_string
                    .. ' time:' .. math.floor(total_time) .. 's'
                    .. ' data:' .. g_train_data:num_loaded_examples()
                if loss.mems then
                    log_string = log_string .. ' mems:'
                    .. util.shortFloat(loss.mems / batch)
                end
                if last_valid_metric then
                    log_string = log_string
                        .. ' last_valid:' .. util.shortFloat(last_valid_metric)
                end
                if best_valid_metric then
                    log_string = log_string
                        .. ' best_valid:' .. util.shortFloat(best_valid_metric)
                end
                log_string = log_string .. "]"
                print(log_string)
                if self.opt.logTrainingFile and self.opt.modelFilename and
                self.opt.allowSaving then
                    local log_file = self.opt.modelFilename .. ".log"
                    local f = io.open(log_file, "a")
                    if f ~= nil then
                        f:write(os.date() .. ":" .. log_string .. "\n")
                        f:close()
                    else
                        if not self.opt.robustFileIO then
                            error('writing log: ' .. log_file .. ' failed!')
                        end
                    end
                end
            end
            collectgarbage()
            loss = {}
            batch = 0
            last_log_time = sys.clock();
        end
    end
    return loss, batch
end

-- Main training loop for serial (non-hogwild) trainer.
function baseNN:do_serial_train()
    -- Create the NN, and possibly load a previous checkpoint.
    print '[training...]'
    self.my_threadidx = 1
    local start_time = sys.clock()
    -- The actual training.
    local loss, batch = self:do_train_worker()
    -- Now print some metrics, and save.
    local total_time = sys.clock() - start_time
    print('[total time: ' .. total_time .. 's]')
    if self.opt.modelFilename ~= nil then
        self:save_all(self.opt.modelFilename)
    end
    local ave_loss = {}
    for k, v in pairs(loss) do
        ave_loss[k] = v / batch
    end
    return ave_loss
end

function baseNN:do_hogwild_train()
    local threads = require('threads')
    threads.serialization('threads.sharedserialize')

    local nworker = self.opt.numThreads
    local local_opt = self.opt
    local shared_dict = self.dict:get_shared()
    local shared_data = {}
    if g_train_data ~= nil and self.opt.threadsShareData then
        shared_data = g_train_data:get_shared()
    end
    local shared_mlp = self:get_shared()
    local modelClass = self.modelClass
    print('[starting ' .. nworker .. ' threads]')
    local workers = threads.Threads(
        nworker,
        function(threadidx)
            require('torch')
            require('sys')
            g_create_mlp_class = require(modelClass)
            torch.manualSeed(345678900 * threadidx)
            math.randomseed(345678900 * threadidx)
        end,
        function(threadidx)
            local opt = local_opt
            local data = require(opt.dataClass)
            if opt.threadsShareData then
                g_train_data = data:create_data(
                    opt.trainData, shared_data, opt, shared_dict)
            else
                g_train_data = data:create_data(opt.trainData, nil, opt,
                                                shared_dict)
            end
            g_my_threadidx = threadidx
            if threadidx == 1 and opt.validEveryNSecs ~= nil and
            opt.validData ~= nil then
                local eval_data
                if opt.evalDataClass then
                    eval_data = require(opt.evalDataClass)
                else
                    eval_data = require(opt.dataClass)
                end
                g_eval_class = require(opt.evalClass)
                g_valid_data = eval_data:create_data(opt.validData, nil, opt,
                                                     shared_dict)
            end
            collectgarbage()
        end)
    collectgarbage()
    print("[ok, let's go...]")
    local err = {}
    local batch = 0
    local start_time = sys.clock()
    local function run_worker()
        local opt = local_opt
        local mlp = g_create_mlp_class:clone_mlp(shared_mlp, opt)
        mlp.my_threadidx = g_my_threadidx
        return mlp:do_train_worker()
    end
    for w = 1, nworker do
        workers:addjob(
            run_worker,
            -- the end callback, ran in the main thread
            function(werr, wbatch)
                for k, v in pairs(werr) do
                    if err[k] then
                        err[k] = err[k] + v
                    else
                        err[k] = v
                    end
                end
                batch = batch + wbatch
            end
        )
    end
    while workers:hasjob() do
        workers:dojob()
        if workers:haserror() then
            for i = 1, #workers.errors do
                print(workers.errors[i])
            end
            error(#workers.errors .. ' occurred, terminating main process')
        end
    end
    local ave_err = {}
    for k, v in pairs(err) do
        ave_err[k] = v / batch
    end
    -- picks an existing loss measure to print
    local err_type, err_val = next(ave_err)
    print('[final its:' .. batch .. ' ' .. err_type .. ':' .. err_val .. ']')
    local total_time = sys.clock() - start_time
    print('[total time: ' .. total_time .. 's]')
    if self.opt.modelFilename ~= nil then
        self:save_all(self.opt.modelFilename)
    end
    return ave_err
end

function baseNN:validate_while_training(
                g_valid_data, best_valid_metric, last_valid_metric,
                time, last_valid_time)
    -- Compute validation error every N seconds.
    local valid_time = time - last_valid_time
    if (self.my_threadidx == 1 and self.opt.validEveryNSecs ~= nil and
        self.opt.validData ~= nil and
        valid_time >= self.opt.validEveryNSecs) then
        print '[running validation...]'
        if not self.firstValid then
            -- Delete valid_eval file as this is the first time.
            self.firstValid = true
            local fname = self.opt.modelFilename .. '.best_valid_eval'
            local f = io.open(fname, 'r')
            if f ~= nil then
                io.close(f)
                os.execute("rm " .. fname)
            end
        end
        -- Temporarily keep the weights frozen.
        if not self.freeze_weights then
            print('Need to implement freeze_weights to validate during '
                      .. 'training, or set validEveryNSecs=nil.')
            error('freeze_weights not implemented')
        end
        self:freeze_weights()

        local metrics = g_eval_data_set(nil, g_valid_data, self)
        local metric = metrics[self.opt.validMetric]
        if self.opt.validOptMetric == 'min' then metric = -metric; end
        if best_valid_metric == nil or metric > best_valid_metric then
            best_valid_metric = metric
            -- Save out the model, as this is the best we have found.
            print('[best validation so far with metric '
                      .. self.opt.validMetric .. ':'
                      .. metrics[self.opt.validMetric] .. ']')
            if self.opt.modelFilename ~= nil then
                g_eval_save_metrics(
                    nil, metrics, self.opt.modelFilename .. '.best_valid_eval',
                    '*', self.opt.validMetric)
                self:save_all(self.opt.modelFilename .. '.best_valid_model')
            end
        else
            print('[not as good as best metric ' .. self.opt.validMetric
                      .. ':' .. best_valid_metric .. ' vs. '
                      ..  metrics[self.opt.validMetric]  .. ']')
            if self.opt.modelFilename ~= nil then
                g_eval_save_metrics(
                    nil, metrics, self.opt.modelFilename .. '.best_valid_eval',
                    nil, self.opt.validMetric)
            end
        end

        -- Put the weights back to the shared ones for hogwild, etc.
        self:unfreeze_weights()
        last_valid_time = sys.clock();
        last_valid_metric = metric
    end

    return last_valid_time, best_valid_metric, last_valid_metric
end

function baseNN:train()
    if not self.opt.continueTraining and self.opt.modelFilename then
        -- Check if the model log file already exists.
        -- If it does, we do not continue training.
        local log_file = self.opt.modelFilename .. '.log'
        local f = io.open(log_file)
        if f ~= nil then
            print('[log file ' .. log_file ..
                    ' already exists: NOT training]')
            return 0
        end
    end
    local data = require(self.opt.dataClass)
    local last_err
    if self.opt.numThreads == 1 then
        g_train_data = data:create_data(self.opt.trainData, nil, self.opt,
                                        self.dict)
        if (self.opt.validEveryNSecs ~= nil and
            self.opt.validData ~= nil) then
            local eval_data
            if self.opt.evalDataClass then
                eval_data = require(self.opt.evalDataClass)
            else
                eval_data = require(self.opt.dataClass)
            end
            g_eval_class = require(self.opt.evalClass)
            g_valid_data = eval_data:create_data(self.opt.validData, nil,
                                                 self.opt, self.dict)
        end
        last_err = self:do_serial_train()
    else
        if self.opt.threadsShareData then
            g_train_data = data:create_data(self.opt.trainData, nil, self.opt,
                                            self.dict)
        end
        last_err = self:do_hogwild_train()
    end
    return last_err
end

return baseNN
