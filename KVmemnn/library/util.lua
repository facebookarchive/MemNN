-- Copyright 2004-present Facebook. All Rights Reserved.
-- Some useful utility functions.

local util = {}

function util.shortFloat(t)
    if t > 100 or math.floor(t) == t then
        return string.format('%g', t)
    else
        return string.format('%.4g', t)
    end
end

-- Log a 'message' the first 'n' times a particular piece of code
-- is exectuted, indicated by the string 'code_location'.
local log = {}
function util.log_first_n(n, code_location, message)
    if log.code_count == nil then
        log.code_count = {}
    end
    if log.code_count[code_location] == nil then
        log.code_count[code_location] = 0
    end
    if log.code_count[code_location] < n then
        print('[' .. code_location .. '|' .. message .. ']')
        log.code_count[code_location] = log.code_count[code_location] + 1
    end
end

function util.same_tensor(y1, y2)
    if y1:dim() ~= y2:dim() then return false; end
    if y1:size(1) ~= y2:size(1) then return false; end
    if y1:dim() == 2 then
        y1 = y1:t()[1]
        y2 = y2:t()[1]
    end
    for i = 1, y1:size(1) do
        if y1[i] ~= y2[i] then
            return false
        end
    end
    return true
end

function util.basename(str)
    local name = string.gsub(str, "(.*/)(.*)", "%2")
    return name
end

local last_log_time
function util.log_progress(
        curr_done, total, start_time, log_interval, message)
    local now = os.time()
    if log_interval then
        if last_log_time == nil then
            -- this is first call, save the log time and move on
            last_log_time = now
            return
        elseif now - last_log_time < log_interval then
            -- logged recently--return to caller
            return
        else
            -- logging now, save the time first
            last_log_time = now
        end
    end

    local frac = curr_done / total
    local total_time = now - start_time
    local left_time = (total_time / curr_done * (total - curr_done))

    local total_hrs_dec = total_time / 60 / 60
    local total_hrs_rnd = math.floor(total_hrs_dec)
    local total_min_dec = (total_hrs_dec - total_hrs_rnd) * 60
    local total_min_rnd = math.floor(total_min_dec)
    local total_sec_dec = (total_min_dec - total_min_rnd) * 60
    local total_sec_rnd = math.floor(total_sec_dec)

    local left_hrs_dec = left_time / 60 / 60
    local left_hrs_rnd = math.floor(left_hrs_dec)
    local left_min_dec = (left_hrs_dec - left_hrs_rnd) * 60
    local left_min_rnd = math.floor(left_min_dec)
    local left_sec_dec = (left_min_dec - left_min_rnd) * 60
    local left_sec_rnd = math.floor(left_sec_dec)

    local message = message and message .. ' ' or ''
    print(string.format(
        '%sApprox. %.2f %% done. Time passed: '
        .. '%dh %02dm %02ds. ETA: %dh %02dm %02ds.',
        message,
        frac * 100,
        total_hrs_rnd,
        total_min_rnd,
        total_sec_rnd,
        left_hrs_rnd,
        left_min_rnd,
        left_sec_rnd
    ))
end

-- A very simple log: Info and debug correspond to levels 0 and 1
-- Messages are printed in the form: log_name:filename:line_number:message
local Log = {}

function Log:new(name, level)
    local obj = {name=name, level=level or 0}
    setmetatable(obj, self)
    self.__index = self
    return obj
end

function Log:verbose(level, message, ...)
    if self.level >= level then
        local info = debug.getinfo(2)
        print(self.name .. ':' .. util.basename(info.short_src) .. ':' ..
              info.currentline .. ':' .. message:format(...))
    end
end

function Log:info(message, ...)
    self:verbose(0, message, ...)
end

function Log:debug(message, ...)
    self:verbose(1, message, ...)
end

util.Log = Log

return util
