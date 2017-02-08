
-- return a module annotated with the given name from an nngraph network.
function get_module(net, name)
    local m = {}
    for k, v in pairs(net.forwardnodes) do
        if v.data.annotations.name == name then
            table.insert(m, v.data.module)
        end
    end
    return m
 end

-- module_list is a table of tables containing modules, share them all
function share_modules(module_list)
   local modules = {}
   for i = 1, #module_list do
      for j = 1, #module_list[i] do
         table.insert(modules, module_list[i][j])
      end
   end
   local m1 = modules[1]
   for i = 2, #modules do
      local m2 = modules[i]
      m2:share(m1, 'weight', 'bias', 'gradWeight', 'gradBias')
   end
end

-- set random seed
function g_make_deterministic(seed)
    torch.manualSeed(seed)
    cutorch.manualSeed(seed)
    torch.zeros(1, 1):cuda():uniform()
end

-- format strings
function g_f4(x)
    return string.format('%0.4f', x)
end

function g_f5(x)
    return string.format('%0.5f', x)
end

-- write a string to a text file
function write(filename, s, mode)
   local mode = mode or "a"
   local ok = false
   local wait = 10
   while not ok do 
      if pcall(function ()
                  local f = assert(io.open(filename, mode))
                  f:write(os.date() .. ":" .. s .. "\n")
                  f:close() end
            ) then 
         ok = true
      else
         print('write failed, retrying in ' .. wait .. ' seconds')
         os.execute("sleep " .. wait)
      end
   end
end
