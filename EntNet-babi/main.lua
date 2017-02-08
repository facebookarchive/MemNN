require('paths')
require('nngraph')
require('cunn')
require('optim')
paths.dofile('params.lua')
paths.dofile('utils.lua')
paths.dofile('layers/Normalization.lua')
torch.setdefaulttensortype('torch.FloatTensor')
g_make_deterministic(123)

-- load the data
trdata = paths.dofile('data.lua')
tedata = paths.dofile('data.lua')
trdata:load(trfiles, opt.batchsize, opt.T)
tedata:load(tefiles, opt.batchsize, trdata.memsize, trdata.dict, trdata.idict)

-- split into train and validation sets
nquestions = trdata.questions:size(1)
train_range = torch.range(1, math.floor(0.9*nquestions))
val_range = torch.range(math.floor(0.9*nquestions) + 1, nquestions)

-- set some parameters based on the dataset
opt.nwords = #trdata.idict
opt.winsize = trdata.memory:size(3)
if opt.tied == 1 then 
   opt.memslots = trdata.entities:size(1)
   print('tying keys to entities -> ' .. opt.memslots .. ' memory slots')
end

-- build the model and loss
paths.dofile(opt.model .. '.lua')
model = build_model(opt)
print('\nmodel: ' .. paths.basename(opt.modelFilename))
print('#params = ' .. model.paramx:size(1))

function train()
    local train_err = {}
    local train_cost = {}
    local val_err = {}
    optstate = {learningRate = opt.sdt}

    for ep = 1, opt.epochs do
        if ep % opt.sdt_decay_step == 0 and opt.sdt_decay_step ~= -1 then
            optstate.learningRate = optstate.learningRate / 2
        end
        model:zeroNilToken()
        if opt.dropout > 0 then 
           model:setDropout('train')
        end
        local total_err, total_cost, total_num = 0, 0, 0
        local nBatches = math.floor(train_range:size(1)/opt.batchsize)
        for k = 1, nBatches do
           xlua.progress(k, nBatches)
            local err, cost
            local feval = function ()
                local batch = train_range:index(1, torch.randperm(train_range:size(1)):sub(1, opt.batchsize):long())
                local question, answer, story = trdata:getBatch(batch)
                err, cost = model:fprop(question, answer, story, graph)
                model:bprop(question, answer, story, sdt)
                return cost, model.paramdx
            end
            optimize(feval, model.paramx, optstate)
            model:zeroNilToken()
            total_cost = total_cost + cost
            total_err = total_err + err
            total_num = total_num + opt.batchsize

            if k % 10 == 0 then 
               collectgarbage()
               collectgarbage()
            end
        end
        train_err[ep] = total_err / total_num
        train_cost[ep] = total_cost / total_num
        val_err[ep] = evaluate('valid')

        local log_string = 'epoch = ' .. ep
            .. ' | train cost = ' .. g_f4(train_cost[ep])
            .. ' | train err = ' .. g_f4(train_err[ep])
            .. ' | valid err = ' .. g_f4(val_err[ep])
            .. ' | lr = ' .. optstate.learningRate

        print(log_string)
        collectgarbage()
    end
    return val_err[opt.epochs], train_err, val_err
end

function evaluate(split, display)
   if opt.dropout > 0 then 
      model:setDropout('test')
   end
    local total_err, total_cost, total_num = 0, 0, 0
    local N, indx
    if split == 'train' then
        N = train_range:size(1)
        indx = train_range
        data = trdata
    elseif split == 'valid' then
        N = val_range:size(1)
        indx = val_range
        data = trdata
    elseif split == 'test' then
        N = tedata.questions:size(1)
        indx = torch.range(1, N)
        data = tedata
    end
    local loss = torch.Tensor(N)
    for k = 1, math.floor(N/opt.batchsize) do
        local batch = indx:index(1, torch.range(1 + (k-1)*opt.batchsize, k*opt.batchsize):long())
        local question, answer, story, facts, graph = data:getBatch(batch)
        local err, cost, missed = model:fprop(question, answer, story, graph)
        total_cost = total_cost + cost
        total_err = total_err + err
        total_num = total_num + opt.batchsize
    end
    return total_err / total_num
 end

final_perf_train = {}
final_perf_val = {}
final_perf_test = {}
weights = {}

for i = 1, opt.runs do
    print('--------------------')
    print('RUN ' .. i)
    print('--------------------')
    -- reset the weights 
    g_make_deterministic(i)
    model:reset()
    -- train
    final_perf_val[i] = train()
    final_perf_train[i] = evaluate('train')
    final_perf_test[i] = evaluate('test')
    weights[i] = model.paramx:clone()
    print('test error = ' .. g_f4(final_perf_test[i]))
    print('val err')
    print(final_perf_val)
    print('test err')
    print(final_perf_test)

    if opt.save ~= '' then
       local log_string = 'run ' .. i 
          .. ' | train error = ' .. g_f4(final_perf_train[i]) 
          .. ' | valid error = ' .. g_f4(final_perf_val[i]) 
          .. ' | test error = ' .. g_f4(final_perf_test[i])
       write(opt.modelFilename .. '.log', log_string)
       torch.save(opt.modelFilename .. '.model', {final_perf_val = final_perf_val, 
                     final_perf_test = final_perf_test, 
                     model = model, 
                     optstate = optstate, 
                     weights = weights})
    end

    if final_perf_val[i] == 0 then
       -- we will pick this run and don't need more
       break
    end
end

-- pick test error based on validation performance
_, best = torch.Tensor(final_perf_val):min(1)

if opt.save ~= '' then
    write(opt.modelFilename .. '.log', 'final test error = ' .. final_perf_test[best[1]])
    torch.save(opt.modelFilename .. '.model', {final_perf_val = final_perf_val, 
                  final_perf_test = final_perf_test, 
                  model = model, 
                  optstate = optstate, 
                  weights = weights})
end