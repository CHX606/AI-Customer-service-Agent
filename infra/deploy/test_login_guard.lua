-- Pure Lua regression tests: no network, real passwords, or production state.
local script = assert(arg[1], "pass the login_guard.lua path")
local now, store, full, phase, checks = 0, {}, false, "rewrite", 0
local dict = {}
function dict:get(key)
    local item = store[key]
    if item and item.deadline <= now then store[key] = nil; item = nil end
    return item and item.value
end
function dict:safe_add(key, value, ttl)
    if self:get(key) ~= nil then return false, "exists" end
    if full then return nil, "no memory" end
    store[key] = {value = value, deadline = now + ttl}
    return true
end
function dict:incr(key, amount)
    local value = self:get(key)
    if value == nil then return nil, "not found" end
    store[key].value = value + amount
    return value + amount
end
function dict:ttl(key)
    if self:get(key) == nil then return nil, "not found" end
    return store[key].deadline - now
end
function dict:expire(key, ttl)
    if self:get(key) == nil then return nil, "not found" end
    store[key].deadline = now + ttl
    return true
end
local function check(condition, message)
    checks = checks + 1
    assert(condition, message)
end
local function reset()
    now, store, full = 0, {}, false
end
local function request(options)
    options = options or {}
    phase = "rewrite"
    _G.ngx = {
        var = {acs_admin_route = options.public and "0" or "1",
               binary_remote_addr = options.ip or "client-a",
               http_authorization = options.anonymous and nil or "Basic test-only",
               http_x_forwarded_for = options.forwarded or "untrusted-forwarded-address"},
        shared = {acs_login_guard = dict}, ctx = {}, header = {}, status = 200,
        ERR = "error", get_phase = function() return phase end,
        log = function() end,
        say = function(body) ngx.body = body end,
        exit = function(status) ngx.exited = status; return status end,
    }
    if options.anonymous then ngx.var.http_authorization = nil end
    dofile(script)
    if ngx.exited then return ngx.exited end
    ngx.status = options.status or 200
    ngx.var.upstream_status = options.upstream or tostring(ngx.status)
    phase = "header_filter"
    dofile(script)
    if options.double_filter then dofile(script) end
    return ngx.status
end

reset()
for _ = 1, 12 do request({anonymous = true, status = 401}) end
check(dict:get("client-a") == nil, "anonymous challenges counted as wrong passwords")
for _ = 1, 12 do request({status = 200}) end
check(dict:get("client-a") == 0, "successful requests counted as failures")
for _, status in ipairs({400, 403, 404, 429, 500, 502}) do request({status = status}) end
check(dict:get("client-a") == 0, "non-authentication response counted")
request({status = 401, upstream = "403"})
check(dict:get("client-a") == 0, "non-upstream 401 counted")
request({status = 401, double_filter = true})
check(dict:get("client-a") == 1, "failure recorded more than once")

reset()
for _ = 1, 9 do check(request({status = 401}) == 401, "blocked before threshold") end
check(request({status = 401}) == 401, "threshold response changed")
check(dict:get("client-a") == 10 and dict:ttl("client-a") == 900, "threshold does not establish 15 minute block")
check(request({status = 200}) == 429, "blocked IP accepted with correct password")
check(ngx.header["Retry-After"] == "900", "missing useful retry delay")
check(request({anonymous = true}) == 429, "blocked IP bypassed ban anonymously")
check(request({forwarded = "a-different-IP-every-time"}) == 429, "forwarded header bypassed ban")
check(request({ip = "client-b", status = 200}) == 200, "ban affected another IP")
check(request({public = true, status = 200}) == 200, "ban affected public chat/site")
now = 890
check(request() == 429 and ngx.header["Retry-After"] == "10", "blocked attempts extended expiry")
now = 901
check(request() == 200, "ban did not automatically expire")

reset()
for _ = 1, 9 do request({status = 401}) end
now = 601
request({status = 401})
check(dict:get("client-a") == 1, "failure window did not expire")
full = true
check(request({ip = "client-c"}) == 503, "full counter store failed open")
check(request({ip = "client-c", public = true}) == 200, "full store blocked public chat")
check(dict:get("client-a") == 1, "full store evicted an existing counter")
print(string.format("PASS: login protection %d assertions", checks))
