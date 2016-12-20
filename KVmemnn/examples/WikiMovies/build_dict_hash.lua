-- Copyright 2004-present Facebook. All Rights Reserved.
-- Modify the dictionary so only entities have low frequency
-- (so we only hash on entities).

local out_dict = arg[1]
local in_dict = arg[2]
local in_entities = arg[3]
local in_freq = arg[4]

-- load original entities
f = io.open(in_entities)
ent = {}
cnt = 0
tot = 0
while true do
    s = f:read("*line")
    if s == nil then break; end
    if ent[s:lower()] ~= nil then
        print("repeat:" .. s)
    end
    ent[s:lower()] = true
    cnt = cnt + 1
end
print("found entities: ", cnt)


f = io.open(in_dict)
fw = io.open(out_dict, "w")
cnt = 0
entz = {}
while true do
    s = f:read("*line")
    if s == nil then break; end
    i1 = s:find('\t')
    name = s:sub(1, i1 - 1)
    if ent[name] ~= nil then
        fw:write(s .. "\n")
        cnt = cnt + 1
        entz[name] = 1
    else
        fw:write(name .. '\t ' .. in_freq .. '\n')
    end
end
fw:close()
print("matched dict entries:", cnt)
print("input dict:", in_dict)
print("output dict:", out_dict)
