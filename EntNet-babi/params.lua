--------------------------
-- hyperparameters
--------------------------

cmd = torch.CmdLine()
-- dataset parameters
cmd:option('-task', 1)
cmd:option('-dataPath', 'data/tasks_1-20_v1-2/en-10k/')
-- model parameters
cmd:option('-model', 'entnet')
cmd:option('-embed', 'icmul+bow', 'bow | icmul+bow')
cmd:option('-tied', 0, 'tie keys to entities')
cmd:option('-edim', 100, 'embedding dimensions')
cmd:option('-memslots', 20, 'memory slots')
cmd:option('-nhop', 1, 'number of hops in output module')
cmd:option('-init_std', 0.1, 'std used for weight initialization')
cmd:option('-dropout', 0)
-- optimization parameters
cmd:option('-optim', 'adam', 'sgd | adam')
cmd:option('-batchsize', 32)
cmd:option('-sdt', 0.01, 'initial learning rate')
cmd:option('-sdt_decay_step', 25, 'how often to reduce learning rate')
cmd:option('-maxgradnorm', 40, 'max gradient norm')
cmd:option('-epochs', 200)
-- other 
cmd:option('-gpu', 1, 'which gpu to use')
cmd:option('-runs', 10)
cmd:option('-save', 'outputs/')

opt = cmd:parse(arg or {})
cutorch.setDevice(opt.gpu)
opt.use_time = (opt.use_time == 1)
print(opt)

if not paths.dirp(opt.save) then
    os.execute('mkdir -p ' .. opt.save)
end

trfiles = {}
tefiles = {}
for f in paths.files(opt.dataPath, 'qa' .. opt.task .. '_') do
   if string.match(f, 'train') then
      table.insert(trfiles, opt.dataPath .. f)
   elseif string.match(f, 'test') then
      table.insert(tefiles, opt.dataPath .. f)
   end
end

-- max number of timesteps
if opt.task == 3 then 
   opt.T = 150 
else
   opt.T = 70
end

-- set filename based on parameters
opt.modelFilename = opt.save
    .. 'task=' .. opt.task
    .. '-model=' .. opt.model
    .. '-embed=' .. opt.embed
    .. '-edim=' .. opt.edim
    .. '-initstd=' .. opt.init_std
    .. '-sdt_decay=' .. opt.sdt_decay_step
    .. '-optim=' .. opt.optim
    .. '-sdt=' .. opt.sdt
    .. '-nhop=' .. opt.nhop
    .. '-tied=' .. opt.tied
   .. '-T=' .. opt.T

if opt.tied == 0 then 
   opt.modelFilename = opt.modelFilename
      .. '-memslots=' .. opt.memslots
end

if opt.optim == 'sgd' then
    optimize = optim.sgd
elseif opt.optim == 'adam' then
    optimize = optim.adam
end