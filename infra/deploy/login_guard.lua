-- Only this site's management routes are guarded. Existing 1Panel WAF access
-- and log handlers remain inherited: use rewrite + header-filter phases here.
-- Never read/log credentials or parse forwarded IP headers. The nginx real-IP
-- module restores addresses only from the site's trusted Cloudflare peers.
local FAILURE_WINDOW = 600
local MAX_FAILURES = 10
local BLOCK_SECONDS = 900

if ngx.var.acs_admin_route ~= "1" then
    return
end

local guard = ngx.shared.acs_login_guard
local key = ngx.var.binary_remote_addr
local phase = ngx.get_phase()

local function reject(status, retry_after)
    ngx.status = status
    ngx.header["Retry-After"] = tostring(math.max(1, math.ceil(retry_after)))
    ngx.header["Cache-Control"] = "no-store"
    ngx.header.content_type = "application/json; charset=utf-8"
    ngx.say('{"detail":"后台登录尝试过多，请稍后再试。"}')
    return ngx.exit(status)
end

if phase == "rewrite" then
    local failures = guard:get(key)
    if failures and failures >= MAX_FAILURES then
        return reject(429, guard:ttl(key) or BLOCK_SECONDS)
    end
    -- Anonymous browser authentication challenges do not count as bad passwords.
    if ngx.var.http_authorization and ngx.var.http_authorization ~= "" then
        -- Reserve a counter without evicting unexpired bans. Fail closed for
        -- management only if the bounded shared store is exhausted.
        local ok, err = guard:safe_add(key, 0, FAILURE_WINDOW)
        if not ok and err ~= "exists" then
            ngx.log(ngx.ERR, "ACS login protection store unavailable")
            return reject(503, 60)
        end
        ngx.ctx.acs_login_candidate = true
    end
elseif phase == "header_filter" then
    ngx.header["Cache-Control"] = "no-store"
    -- Only actual upstream authentication rejection counts. A successful
    -- authenticated API request, 404, 5xx, or our own 429 must not add failures.
    if not ngx.ctx.acs_login_candidate or ngx.ctx.acs_login_counted
            or ngx.status ~= 401 or ngx.var.upstream_status ~= "401" then
        return
    end
    ngx.ctx.acs_login_counted = true
    local failures, err = guard:incr(key, 1)
    if not failures then
        ngx.log(ngx.ERR, "ACS login failure counter unavailable: ", err)
        return
    end
    if failures == MAX_FAILURES then
        local ok = guard:expire(key, BLOCK_SECONDS)
        if not ok then
            ngx.log(ngx.ERR, "ACS login protection expiry update failed")
        end
    end
end
