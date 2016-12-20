-- Copyright 2004-present Facebook. All Rights Reserved.
local ffi = require('ffi')

ffi.cdef[[

typedef struct
{
   double *data;
   long size;
   int refcount;
   char flag;
} THDoubleStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THDoubleStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THDoubleTensor;

]]

local thdt = ffi.typeof('THDoubleTensor*')
function torch.DoubleTensor.data(self)
   local ptr = ffi.cast(thdt, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end

ffi.cdef[[

typedef struct
{
   float *data;
   long size;
   int refcount;
   char flag;
} THFloatStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THFloatStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THFloatTensor;

]]

local thft = ffi.typeof('THFloatTensor*')
function torch.FloatTensor.data(self)
   local ptr = ffi.cast(thft, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end

ffi.cdef[[

typedef struct
{
   unsigned char *data;
   long size;
   int refcount;
   char flag;
} THByteStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THByteStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THByteTensor;

]]

local thbt = ffi.typeof('THByteTensor*')
function torch.ByteTensor.data(self)
   local ptr = ffi.cast(thbt, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end

ffi.cdef[[

typedef struct
{
   char *data;
   long size;
   int refcount;
   char flag;
} THCharStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THCharStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THCharTensor;

]]

local thct = ffi.typeof('THCharTensor*')
function torch.CharTensor.data(self)
   local ptr = ffi.cast(thct, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end


ffi.cdef[[

typedef struct
{
   int *data;
   long size;
   int refcount;
   char flag;
} THIntStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THIntStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THIntTensor;

]]

local thit = ffi.typeof('THIntTensor*')
function torch.IntTensor.data(self)
   local ptr = ffi.cast(thit, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end

ffi.cdef[[

typedef struct
{
   long *data;
   long size;
   int refcount;
   char flag;
} THLongStorage;

typedef struct
{
   long *size;
   long *stride;
   int nDimension;
   THLongStorage *storage;
   long storageOffset;
   int refcount;
   char flag;
} THLongTensor;

]]

local thlt = ffi.typeof('THLongTensor*')
function torch.LongTensor.data(self)
   local ptr = ffi.cast(thlt, torch.pointer(self))
   return ptr.storage.data+ptr.storageOffset
end

local thread_utils = {}

function thread_utils.get_shared_ptr_ronan(x)
    local x_p = x:storage():cdata()
    return { x, x_p, x:size() }
end

function thread_utils.create_from_shared_ptr_ronan(x)
    -- x is table containing: (x, x_p, x_size, x_dim)
    local s = x[1]:storage().new()
    local s_p = s:cdata()
    s_p.data = x[2].data
    s_p.size = torch.LongTensor(x[3]):prod(1):squeeze()
    s_p.flag = 0
    s_p.refcount = 0
    local z = x[1].new(s, 1, x[3])
    return z
end

function thread_utils.get_shared_ptr(x, ttype)
    if ttype == nil then ttype = 'THDoubleStorage*'; end
    if ttype == 'DoubleStorage*' then ttype = 'THDoubleStorage*'; end
    if ttype == 'FloatStorage*' then ttype = 'THFloatStorage*'; end
    if ttype == 'LongStorage*' then ttype = 'THLongStorage*'; end
    local x_storage_ptr =
        tonumber(ffi.cast(
                     'intptr_t',
                     ffi.cast(ttype,
                              torch.pointer(x:storage())).data))
    local res = {}
    res[1] = x_storage_ptr
    res[2] = x:size()
    res[3] = x:nDimension()
    return res
end

function thread_utils.create_from_shared_ptr(x, ttype)
    local x2
    if ttype == nil then ttype = "THFloatStorage*"; end
    if ttype == 'DoubleStorage*' then ttype = 'THDoubleStorage*'; end
    if ttype == 'FloatStorage*' then ttype = 'THFloatStorage*'; end
    if ttype == 'LongStorage*' then ttype = 'THLongStorage*'; end
    if ttype == 'THFloatStorage*' then
        x2 = torch.FloatTensor(x[2])
    elseif ttype == 'THLongStorage*' then
       -- Default constructor of LongTensor does not consider LongStorage as its
       -- size. Rather, it uses the passed in LongStorage as its storage.
       x2 = torch.LongTensor(
          torch.LongStorage(torch.LongTensor(x[2]):prod())):reshape(x[2])
    else
        -- Assume double.
        x2 = torch.DoubleTensor(x[2])
    end
    local storage = x2:storage()
    local storage_ptr
    if ttype == "THFloatStorage*" then
        storage_ptr = ffi.cast(ttype, torch.pointer(storage))
        storage_ptr.data = ffi.cast('float*', x[1])
    elseif ttype == "THLongStorage*" then
        storage_ptr = ffi.cast(ttype, torch.pointer(storage))
        storage_ptr.data = ffi.cast('long*', x[1])
    else
        -- Assume double.
        storage_ptr = ffi.cast(ttype, torch.pointer(storage))
        storage_ptr.data = ffi.cast('double*', x[1])
    end
    storage_ptr.flag = 0
    storage_ptr.refcount = 0
    return x2
end

-- Reset a tensor t to use the shared storage defined in x.
function thread_utils.reset_to_shared_ptr(t, x, ttype)
    local ptype = ttype or "double*"
    if ptype == 'THDoubleStorage*' then ptype = 'double*'; end
    if ptype == 'THFloatStorage*' then ptype = 'float*'; end
    if ptype == 'THLongStorage*' then ptype = 'long int*'; end
    local data_t = x[1]
    local size_t = x[2]
    local dim = x[3]
    local size = torch.LongTensor(dim)
    for i= 1,dim do
        size[i] = tonumber(size_t[i])
    end
    local s = t:storage().new()
    local s_t = s:cdata()
    s_t.data = ffi.cast(ptype, data_t)
    s_t.size = size:prod(1):squeeze()
    s_t.flag = 0
    t:set(s, 1, size:storage())
end

return thread_utils
