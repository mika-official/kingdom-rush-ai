-- Sound selection must not advance the combat RNG. Sound queue timing follows
-- real time even when simulation is stepped manually. Keep the native combat
-- random implementation; only sound_db's random draws use a separate stream.
local native_random = math.random
local native_seed = math.randomseed
local sound_state = 1
local calls = {sound=0, gameplay=0}
local sources = {}
local seeds = {}
local trace = {}
local sound_depth=0
-- A sound function may tail-call a helper, erasing its stack frame. Scope the
-- already loaded sound API as well as checking callers for queued callbacks.
local sound_db=package.loaded.sound_db
if sound_db then
    for name,fn in pairs(sound_db) do
        if type(fn)=="function" then
            sound_db[name]=function(...)
                sound_depth=sound_depth+1
                local function packed(...) return {n=select('#',...),...} end
                local result=packed(pcall(fn,...))
                sound_depth=sound_depth-1
                if not result[1] then error(result[2],0) end
                return unpack(result,2,result.n)
            end
        end
    end
end
-- Native waves_active is keyed by coroutine objects. pairs() visits them in
-- pointer/hash order, which changes after resets and between processes.
local native_create = coroutine.create
local native_pairs = pairs
local thread_rank = setmetatable({}, {__mode="k"})
local next_rank = 0
coroutine.create = function(...)
    local thread=native_create(...)
    next_rank=next_rank+1;thread_rank[thread]=next_rank
    return thread
end
pairs = function(t)
    local s=game and game.store
    if not s or t~=s.waves_active then return native_pairs(t) end
    local keys={}
    for key in native_pairs(t) do
        assert(type(key)=="thread" and thread_rank[key],"Untracked native wave spawner")
        keys[#keys+1]=key
    end
    table.sort(keys,function(a,b) return thread_rank[a]<thread_rank[b] end)
    local index=0
    return function()
        index=index+1
        local key=keys[index]
        if key then return key,t[key] end
    end,t,nil
end
local function sound_random(a,b)
    sound_state = (sound_state * 16807) % 2147483647
    local unit = (sound_state-1)/2147483646
    if a == nil then return unit end
    if b == nil then b,a = a,1 end
    a,b = math.floor(a),math.floor(b)
    assert(a<=b,"interval is empty")
    return a+math.floor(unit*(b-a+1))
end
math.random = function(...)
    local source = debug.getinfo(2,"S").source
    local sound = sound_depth>0 or source:find("sound_db.lua",1,true)
    if not sound then
        for depth=3,8 do
            local caller=debug.getinfo(depth,"S")
            if not caller then break end
            if caller.source:find("sound_db.lua",1,true) then sound=true;break end
        end
    end
    if sound then
        calls.sound=calls.sound+1
        return sound_random(...)
    end
    calls.gameplay=calls.gameplay+1
    sources[source]=(sources[source] or 0)+1
    local result=native_random(...)
    if AI.rng_trace then trace[#trace+1]={source=source,args={...},value=result} end
    return result
end
math.randomseed = function(seed)
    sound_state = math.floor(math.abs(seed)) % 2147483646 + 1
    calls.sound,calls.gameplay=0,0
    sources={};trace={};seeds[#seeds+1]={seed=seed,source=debug.getinfo(2,"S").source}
    return native_seed(seed)
end
AI.rng_isolation_version = 2
AI.rng_status = function() return {version=2,sound=calls.sound,gameplay=calls.gameplay,sources=sources,seeds=seeds,trace=AI.rng_trace and trace or nil} end
return "OK"
