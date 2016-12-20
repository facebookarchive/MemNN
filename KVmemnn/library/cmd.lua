-- Copyright 2004-present Facebook. All Rights Reserved.
-- Command line parameters for models.

local cmd = {}
cmd.__index = cmd

function cmd:reset()
    self.opt = {}
end

-- Create new option.
function cmd:option(name, default_value, description, option_type, hideDefault)
    if self.opt == nil then self.opt = {}; end
    if self.opt[name] ~= nil then
        print("[WARNING - creating option that already exists:" .. name .. ']')
    end
    if option_type == nil then option_type = "xtra"; end
    if hideDefault == nil then hideDefault = false; end
    self.opt[name] = {default_value, description, option_type, hideDefault}
end

-- Reset default of existing option.
function cmd:reset_default(name, default_value)
    if self.opt[name] == nil then
        error(name .. " does not exist as an option yet")
    end
    self.opt[name][1] = default_value
end

function cmd:error(s)
    print("Failure to parse argument: " .. s)
    print("Options:")
    for i,k in pairs(self.opt) do
        print("-" .. i .. " : ", k[2], " [default:", k[1], "]")
    end
    os.exit(1)
end

-- Iterate through a table, creating options and setting defaults.
function cmd:set_defaults_from_table(t)
    if self.opt == nil then self.opt = {}; end
    for name, value in pairs(t) do
        if self.opt[name] ~= nil then
            self:reset_default(name, value)
        else
            self:option(name, value, name)
        end
    end
end

-- Print the options for each type (data, model, logs), each
-- in sorted order. If hideDefaults is true, values that are
-- set to defaults (that are labeled as hidden) are not shown,
-- this makes the output more readable.
function cmd:print(opt, hideDefaults)
    local opt_type = {}
    for i, k in pairs(opt) do
        local ot = "xtra"
        if self.opt[i] ~= nil then
            ot = self.opt[i][3]
        end
        local hide = false
        if self.opt[i][4] and k == self.opt[i][1] then
            -- Hide option that is not changed from the default.
            if hideDefaults ~= false then hide = true; end
        end
        if not hide then
            if opt_type[ot] == nil then opt_type[ot] = {}; end
            local t = opt_type[ot]
            t[#t + 1] = i
        end
    end
    local sorted_types = {}
    for i, k in pairs(opt_type) do
        sorted_types[#sorted_types + 1] = i;
    end
    table.sort(sorted_types)
    for i = 1, #sorted_types do
        local name = sorted_types[i]
        print('--- ' .. name .. ' options ---')
        table.sort(opt_type[name])
        print('{')
        for i = 1, #opt_type[name] do
            local key = opt_type[name][i]
            print("  " .. key, opt[key])
        end
        print('}')
    end
end

function cmd:parse(arg0, ignoreErrors)
    local opt = {}
    -- First, fill in the defaults.
    if self.opt then
        for i,k in pairs(self.opt) do
            opt[i] = k[1]
        end
    end
    local arg = {}
    local pos = 1
    while pos <= #arg0 do
        if arg0[pos]:find('=') == nil then
            arg[#arg + 1] = arg0[pos] .. '=' .. arg0[pos + 1]
            pos = pos + 2
        else
            arg[#arg + 1] = arg0[pos]
            pos = pos + 1
        end
    end
    for i = 1, #arg do
        local s = arg[i]
        if s ~= '-i' then
            local i1 = s:find('=')
            if i1 == nil and not ignoreErrors then
                self:error(s)
            end
            local key = s:sub(1, i1 - 1)
            local value = s:sub(i1 + 1, -1)
            -- Strip the key of leading hyphens.
            while key:sub(1,1) == '-' do
                key = key:sub(2, -1)
            end
            -- Possibly convert the value to a number or bool
            if tonumber(value) ~= nil then value = tonumber(value); end
            if value == "true" then value = true; end
            if value == "false" then value = false; end
            if value == "nil" then value = nil; end
            if self.opt[key] == nil and not ignoreErrors then
                self:error(s)
            end
            opt[key] = value
        end
    end
    return opt
end

-- Parser specific to loading a model and a data class.
function cmd:parse_mlp(arg)
    -- Get the model and data choices.
    local opt = self:parse(arg, true)
    -- Define the data and model and add their command line options.
    local mlp = require(opt.modelClass)
    cmd:option('modelClass', opt.modelClass, 'model')
    mlp:add_cmdline_options(self)
    local data = require(opt.dataClass)
    cmd:option('dataClass', opt.dataClass, 'data')
    data:add_cmdline_options(self)
    -- Now reparse now we know the specific options for the model choice.
    return self:parse(arg)
end

function cmd:parse_from_modelfile_and_override_with_args(arg)
    -- Find the modelFilename in the arguments.
    local opt1 = self:parse(arg, true)
    -- Load the options of that model.
    local fopt = torch.load(opt1.modelFilename .. '.opt')

    -- Set up the default arguments for that model/data pair.
    local farg = {}
    if opt1.modelClass then
        farg[1] = 'modelClass=' .. opt1.modelClass
    else
        farg[1] = 'modelClass=' .. fopt.modelClass
    end
    farg[2] = 'dataClass=' .. fopt.dataClass
    self:parse_mlp(farg)
    -- Set params from the file.
    self:set_defaults_from_table(fopt)
    -- Reparse arguments, possibly overriding defaults from the model file.
    local opt = self:parse(arg)
    return opt
end

function cmd:new(useDefaults)
    -- Set standard default options.
    -- Set useDefaults to false to not set any defaults at all.
    local opt = {}
    setmetatable(opt, self)
    if useDefaults == false then
        return opt
    end
    opt:option('modelClass',
               'library.base_model',
               'model class to use', 'model')
    opt:option('dataClass', 'library.data',
               'data class to use', 'data')
    return opt
end

return cmd
