-- Copyright 2004-present Facebook. All Rights Reserved.
-- Class for storing an array of vectors (d x n matrices) efficiently.

require('torch')
local thread_utils =
    require('library.thread_utils')

local VectorArray = {}
VectorArray.__index = VectorArray

-- Create an array of vectors.
-- Arguments:
-- maxsz: maximum amount of floats/doubles to be stored.
-- maxcnt: maximum number of vectors in the array.
-- n: size(2) of the vectors, i.e. each vector is d x n, where d is
--  variable, but n is fixed (usually set to 1 or 2).
-- useDoubles: true=double or false=float (default).
function VectorArray:new(maxsz, maxcnt, n, useDoubles)
   local va = {}
   setmetatable(va, {__index = VectorArray})
   va.n = n or 1
   va.useDoubles = useDoubles or false
   if useDoubles then
       va.storage = 'DoubleStorage*'
   else
       va.storage = 'FloatStorage*'
   end
   -- Contains all the data.
   va.data = va:Tensor(maxsz, n)
   -- Tells us where to point inside idata to get the i-th entry.
   va.idx = torch.DoubleTensor(maxcnt)
   -- The dimensions of the i-th entry.
   va.len = torch.DoubleTensor(maxcnt)
   -- Right now there are no entries.
   va.cnt = torch.DoubleTensor(1)
   va.sz = torch.DoubleTensor(1)
   va:clear()
   return va
end

-- resizes if necessary, ensuring at least enough extra space for 'items' more
-- additions of length 'len'
function VectorArray:resize(len, items)
    items = items or 1
    len = len or items
    local currcnt = self.cnt[1]
    local maxcnt = self.len:size(1)
    local currsz = self.sz[1]
    local maxsz = self.data:size(1)

    local GROWTH_FACTOR = (1 + math.sqrt(5)) / 2 -- golden ratio
    -- add more indices if you're full
    local chk1, chk2
    if maxcnt - currcnt < items then
        if currcnt > 0 then chk1 = self.idx[1]; chk2 = self.idx[currcnt] end
        self.idx:resize(math.ceil((maxcnt + items) * GROWTH_FACTOR))
        -- safety checks
        if currcnt > 0 then
            assert(self.idx[1] == chk1)
            assert(self.idx[currcnt] == chk2)
            chk1 = self.len[1]
            chk2 = self.len[currcnt]
        end
        self.len:resize(math.ceil((maxcnt + items) * GROWTH_FACTOR))
        if currcnt > 0 then
            assert(self.len[1] == chk1)
            assert(self.len[currcnt] == chk2)
        end
    end
    -- add more space if you're full
    if maxsz - currsz < len then
        if currcnt > 0 then chk1 = self:get(1); chk2 = self:get(currcnt) end
        if self.n > 1 then
            self.data:resize(math.ceil((maxsz + len) * GROWTH_FACTOR), self.n)
        else
            self.data:resize(math.ceil((maxsz + len) * GROWTH_FACTOR))
        end
        if currcnt > 0 then
            assert(self:get(1):eq(chk1):sum() == chk1:numel())
            assert(self:get(currcnt):eq(chk2):sum() == chk2:numel())
        end
    end
end

-- Clip tensors to the current sizes, this is used when
-- the data will not get any bigger to save memory.
function VectorArray:clip()
    local curr_cnt = self.cnt[1]
    local chk1, chk2

    if curr_cnt > 0 then chk1 = self.idx[1]; chk2 = self.idx[curr_cnt] end
    self.idx:resize(curr_cnt)
    -- safety checks
    if curr_cnt > 0 then
        assert(self.idx[1] == chk1)
        assert(self.idx[curr_cnt] == chk2)
        chk1 = self.len[1]
        chk2 = self.len[curr_cnt]
    end
    self.len:resize(curr_cnt)
    if curr_cnt > 0 then
        assert(self.len[1] == chk1)
        assert(self.len[curr_cnt] == chk2)

        chk1 = self:get(1)
        chk2 = self:get(curr_cnt)
    end
    if self.n > 1 then
        self.data:resize(self.sz[1], self.n)
    else
        self.data:resize(self.sz[1])
    end
    if curr_cnt > 0 then
        assert(self:get(1):eq(chk1):sum() == chk1:numel())
        assert(self:get(curr_cnt):eq(chk2):sum() == chk2:numel())
    end
end

function VectorArray:Tensor(d, n)
    if n == nil then n = 1; end
    if self.useDoubles then
        if n > 1 then
            return torch.DoubleTensor(d, self.n)
        else
            return torch.DoubleTensor(d)
        end
    else
        if n > 1 then
            return torch.FloatTensor(d, self.n)
        else
            return torch.FloatTensor(d)
        end
    end
end

function VectorArray:get_shared()
    local t = {}
    t.data = thread_utils.get_shared_ptr(self.data, self.storage)
    t.idx = thread_utils.get_shared_ptr(self.idx, 'DoubleStorage*')
    t.len = thread_utils.get_shared_ptr(self.len, 'DoubleStorage*')
    t.cnt = thread_utils.get_shared_ptr(self.cnt, 'DoubleStorage*')
    t.sz = thread_utils.get_shared_ptr(self.sz, 'DoubleStorage*')
    t.n = self.n
    t.useDoubles = self.useDoubles
    t.storage = self.storage
    return t
end

function VectorArray:new_shared(shared_sv)
   local self = {}
   self.storage = shared_sv.storage
   self.data = thread_utils.create_from_shared_ptr(shared_sv.data,
                                                   self.storage)
   self.idx = thread_utils.create_from_shared_ptr(shared_sv.idx,
                                                  'DoubleStorage*')
   self.len = thread_utils.create_from_shared_ptr(shared_sv.len,
                                                  'DoubleStorage*')
   self.cnt = thread_utils.create_from_shared_ptr(shared_sv.cnt,
                                                  'DoubleStorage*')
   self.sz = thread_utils.create_from_shared_ptr(shared_sv.sz,
                                                 'DoubleStorage*')
   self.n = shared_sv.n
   setmetatable(self, {__index = VectorArray})
   return self
end

function VectorArray:clear()
    self.cnt[1] = 0
    self.sz[1] = 0
end

function VectorArray:size()
    return self.cnt[1]
end

function VectorArray:get(i)
    -- A narrow gives us the data with no copy.
    return self.data:narrow(1, self.idx[i], self.len[i])
end

-- ensures enough space is available for new items
function VectorArray:check_and_resize(len, items)
    if self.sz[1] + len > self.data:size(1)
    or self.cnt[1] + items > self.idx:size(1) then
        self:resize(len, items)
    end
end

function VectorArray:add(x)
    if self.useDoubles then
        if torch.typename(x) ~= 'torch.DoubleTensor' then
            error('expected torch.DoubleTensor')
        end
    else
        if torch.typename(x) ~= 'torch.FloatTensor' then
            error('expected torch.FloatTensor')
        end
    end
    local len = x:size(1) -- Size of the current entry.
    self:check_and_resize(len, 1) -- resize if more space is needed
    local offset = self.sz[1] + 1 -- Where are we now inside data.
    -- Copy the data.
    self.data:narrow(1, offset, len):copy(x)

    local cnt = self.cnt[1] + 1
    self.idx[cnt] = offset
    self.len[cnt] = len
    self.sz[1] = self.sz[1] + len
    self.cnt[1] = cnt
    return true
end

function VectorArray:add_vecarr(va)
    if not va then return false end
    local len = va.sz[1]
    local items = va.cnt[1]
    self:check_and_resize(len, items) -- resize if more space is needed
    local data_offset = self.sz[1] + 1  -- Where are we now inside data.
    local otherData = va.data:narrow(1, 1, va.sz[1])
    self.data:narrow(1, data_offset, len):copy(otherData)

    local cnt_offset = self.cnt[1] + 1
    local otherIdx = va.idx:narrow(1, 1, va.cnt[1])
    local otherLen = va.len:narrow(1, 1, va.cnt[1])
    self.idx:narrow(1, cnt_offset, items):copy(otherIdx):add(data_offset - 1)
    self.len:narrow(1, cnt_offset, items):copy(otherLen)
    self.sz[1] = self.sz[1] + len
    self.cnt[1] = self.cnt[1] + items
    return true
end

function VectorArray:load(filename)
    print("[loading VectorArray: " .. filename .. "]")
    local va = torch.load(filename)
    setmetatable(va, {__index = VectorArray})
    return va
end

function VectorArray:save(filename)
    print('[saving VectorArray:' .. filename .. ']')
    torch.save(filename, self)
    return true
end

function VectorArray:create_from_tds_hash(hash)
    local max_key = 0
    for i, k in pairs(hash) do
        if i > max_key then max_key = i; end
    end
    local maxcnt = max_key
    assert(maxcnt > 0)

    local maxsz = 0
    for i = 1, maxcnt do
        if hash[i] ~= nil and hash[i]:dim() > 0 then
            maxsz = maxsz + hash[i]:size(1)
        else
            maxsz = maxsz + 1
        end
    end
    local n = 1
    local useDoubles = false
    for i = 1, maxcnt do
        if hash[i] ~= nil and hash[i]:dim() > 0 then
            if hash[i]:dim() == 2 then
                n = hash[i]:size(2)
            end
            useDoubles = hash[i].__typename == 'torch.DoubleTensor'
            break
        end
    end
    local holder = self:new(maxsz, maxcnt, n, useDoubles)

    for i = 1, maxcnt do
        if hash[i] ~= nil and hash[i]:dim() > 0 then
            holder:add(hash[i])
        else
            local tmpvec
            if useDoubles then
                tmpvec = torch.DoubleTensor({0})
            else
                tmpvec = torch.FloatTensor({0})
            end
            holder:add(tmpvec)
        end
    end
    return holder
end

function VectorArray:save_atomic_vector(filename, atom_vec)
    assert(#atom_vec > 0)
    local maxcnt = #atom_vec
    local maxsz = 0
    for i = 1, maxcnt do
        if atom_vec[i]:dim()>0 then
            maxsz = maxsz + atom_vec[i]:size(1)
        else
            maxsz = maxsz + 1
        end
    end
    local n = atom_vec[1]:size(2)
    local useDoubles = atom_vec[1].__typename == 'torch.DoubleTensor'

    local holder = self:new(maxsz, maxcnt, n, useDoubles)
    for i = 1, maxcnt do
        if atom_vec[i]:dim()>0 then
            holder:add(atom_vec[i])
        else
            local tmpvec
            if useDoubles then
                tmpvec = torch.DoubleTensor(1, n)
            else
                tmpvec = torch.FloatTensor(1, n)
            end
            tmpvec[1][1] = i
            holder:add(tmpvec)
        end
    end
    holder:save(filename)
end

return VectorArray
