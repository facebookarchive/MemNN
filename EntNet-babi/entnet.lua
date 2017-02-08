--------------------------
-- create the EntNet
--------------------------

local vocabsize = opt.nwords
if opt.tied == 0 then
   -- add extra words to the vocabulary representing keys
   vocabsize = vocabsize + opt.memslots
end

-- function to take a set of word embeddings and produce a fixed-size vector
function input_encoder(opt, input, model, label)
   local input = nn.View(-1, opt.winsize * opt.edim)(input)
    if model == 'bow' then
       return nn.Sum(2)(nn.View(opt.batchsize, opt.winsize, opt.edim)(input))
    elseif model == 'icmul+bow' then
       input = nn.Dropout(opt.dropout)(input):annotate{name = 'dropout'}
       input = nn.View(-1, opt.winsize * opt.edim)(input)
       input = nn.CMul(opt.winsize * opt.edim)(input):annotate{name = label}
       return nn.Sum(2)(nn.View(opt.batchsize, opt.winsize, opt.edim)(input))
    else
       error('encoder not recognized')
    end
end

-- output layer
local function output_module(opt, hops, x, M)
   local hid = {}
   local s = nn.LookupTable(vocabsize, opt.edim)(x):annotate{name = 'E'}
   hid[0] = input_encoder(opt, s, opt.embed, 'q_embed1')
   for h = 1, hops do
      local hid3dim = nn.View(opt.batchsize, 1, opt.edim)(hid[h-1])
      local MMaout = nn.MM(false, true)
      local Aout = MMaout({hid3dim, M})
      local Aout2dim = nn.View(opt.batchsize, -1)(Aout)
      local P = nn.SoftMax()(Aout2dim)
      local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
      local MMbout = nn.MM(false, false)
      local Bout = nn.View(opt.batchsize, -1)(MMbout({probs3dim, M}))
      local C = nn.Linear(opt.edim, opt.edim, false)(Bout):annotate{name = 'H'}
      local D = nn.CAddTable()({hid[h-1], C})
      hid[h] = nn.PReLU(opt.edim)(D):annotate{name = 'prelu'}
   end
   local z = nn.Linear(opt.edim, vocabsize, false)(hid[hops]):annotate{name = 'z'}
   local pred = nn.LogSoftMax()(nn.Narrow(2, 1, opt.nwords)(z))
   return pred
end

-- dynamic memory layer
local function update_memory(opt, keys, sentence, memories, t, mask)
   -- reshape everything to 2D so it can be fed to nn.Linear
   local sentence = nn.Replicate(opt.memslots, 2, 3)(nn.View(opt.batchsize, 1, opt.edim)(sentence))
   sentence       = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(sentence))
   local keys     = nn.Replicate(opt.batchsize, 1, 3)(nn.View(1, opt.memslots, opt.edim)(keys))
   keys           = nn.View(opt.batchsize * opt.memslots, opt.edim)(nn.Contiguous()(keys))
   local mask     = nn.Replicate(opt.memslots, 2, 2)(mask)
   local memories = nn.View(opt.batchsize * opt.memslots, opt.edim)(memories)

   local function DotBias(a, b)
      return nn.Add(opt.memslots)(nn.View(opt.batchsize, opt.memslots)(nn.DotProduct(){a, b})):annotate{name = 'gate_bias'}
   end

   -- compute the gate activations (mask indicates end of story which forces gates to close)
   local gate = nn.Sigmoid()(DotBias(nn.CAddTable(){memories, keys}, sentence))
   gate = nn.CMulTable(){gate, mask}:annotate{name = 'gate' .. t}
   gate = nn.Replicate(opt.edim, 2, 2)(nn.View(opt.batchsize * opt.memslots, 1)(gate))

   -- compute candidate memories
   local U = nn.Linear(opt.edim, opt.edim)(memories):annotate{name = 'U'}
   local V = nn.Linear(opt.edim, opt.edim, false)(sentence):annotate{name = 'V'}
   local W = nn.Linear(opt.edim, opt.edim, false)(keys):annotate{name = 'W'}
   local candidate_memories = nn.PReLU(opt.edim)(nn.CAddTable(){U, V, W}):annotate{name = 'prelu'}

   -- update the memories
   local updated_memories = nn.CAddTable(){memories, nn.CMulTable(){gate, candidate_memories}}

   -- normalize to the sphere
   updated_memories = nn.Normalization()(updated_memories)

   return nn.View(opt.batchsize, opt.memslots, opt.edim)(updated_memories)
end

-- build the nngraph module
local function build_network(opt)
   local question = nn.Identity()()
   local story    = nn.Identity()()
   local keys     = nn.Identity()()
   local mask     = nn.Identity()()
   local memories = {}

   local initmems = nn.Replicate(opt.batchsize, 1, 2)(keys)
   memories[0]    = nn.LookupTable(vocabsize, opt.edim)(initmems):annotate{name = 'E'}
   local keyvecs  = nn.LookupTable(vocabsize, opt.edim)(keys):annotate{name = 'E'}
   for i = 1, opt.T do
      local sentence = input_encoder(opt, nn.LookupTable(vocabsize, opt.edim)(nn.Select(2, i)(story)):annotate{name = 'E'}, opt.embed, 's_embed1')
      memories[i]    = update_memory(opt, keyvecs, sentence,  memories[i - 1], i, nn.Select(2, i)(mask))
   end
   local pred = output_module(opt, opt.nhop, question, memories[opt.T])
   return nn.gModule({question, story, keys, mask}, {pred})
end

-- build the final model
function build_model(opt)
   local model = {}
   model.network = build_network(opt)
   model.network = model.network:cuda()

   if opt.tied == 0 then
      model.keys = torch.range(opt.nwords + 1, opt.nwords + opt.memslots)
   else
      model.keys = trdata.entities
   end
   model.keys = model.keys:cuda()

   -- share the clones across timesteps
   share_modules({get_module(model.network, 'prelu')})
   share_modules({get_module(model.network, 'gate_bias')})
   share_modules({get_module(model.network, 'U')})
   share_modules({get_module(model.network, 'V')})
   share_modules({get_module(model.network, 'W')})
   share_modules({get_module(model.network, 'q_embed1')})
   share_modules({get_module(model.network, 's_embed1')})
   share_modules({get_module(model.network, 'E')})
   share_modules({get_module(model.network, 'H')})

   model.paramx, model.paramdx = model.network:getParameters()
   model.loss = nn.ClassNLLCriterion():cuda()
   model.loss.sizeAverage = false

   function model:reset()
      -- initialize weight to a Gaussian
      self.paramx:normal(0, opt.init_std)

      -- initialize PReLU slopes to 1
      local prelus = get_module(self.network, 'prelu')
      for i = 1, #prelus do
         prelus[i].weight:fill(1)
      end

      -- initialize encoder mask weights to 1 (i.e. BoW)
      if opt.embed == 'icmul+bow' then
         local icmul = get_module(self.network, 'q_embed1')
         for i = 1, #icmul do
            local w = icmul[i].weight
            w:fill(1)
         end
      end
   end

   function model:zeroNilToken()
      local G = get_module(self.network, 'E')
      local Z = get_module(self.network, 'z')
      for i = 1, #G do G[i].weight[1]:zero() end
      for i = 1, #Z do Z[i].weight[1]:zero() end
   end

   function model:setDropout(split)
      local drop = get_module(self.network, 'dropout')
      for i = 1, #drop do
         drop[i].train = (split == 'train')
      end
   end

   function model:fprop(question, answer, story)
      self.mask = story:ne(1):sum(3):select(3,1):ne(0):cuda()
      self.logprob = self.network:forward({question, story, self.keys, self.mask})
      local cost = self.loss:forward(self.logprob, answer)
      local _, pred = self.logprob:max(2)
      pred = pred:cuda()
      local missed = pred:ne(answer)
      return missed:sum(), cost, missed, pred
   end

   function model:bprop(question, answer, story)
      self.network:zeroGradParameters()
      local grad = self.loss:updateGradInput(self.logprob, answer)
      self.network:backward({question, story, self.keys, self.mask}, grad)
      local gradnorm = self.paramdx:norm()
      if gradnorm > opt.maxgradnorm then
         self.paramdx:mul(opt.maxgradnorm / gradnorm)
      end
      self:zeroNilToken()
   end

   return model
end
