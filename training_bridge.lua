-- Process-local training storage and asynchronous director transition support.
-- Loaded only by the multi-level client. Keep the storage overlay until process exit:
-- restoring disk writes while game objects retain training data would be unsafe.
assert(AI and AI.VERSION == 2 and AI.persistent, "Re-inject this project's boot.lua (director hook required)")
if AI.training then return "OK" end
local storage = require("storage")
local signal = require("hump.signal")
local initial_encode = (ENV and ENV.json) or require("json").encode
local function json(value) return ((ENV and ENV.json) or initial_encode)(value) end
local function copy(v)
    if type(v) ~= "table" then return v end
    local out = {}
    for k, x in pairs(v) do out[k] = copy(x) end
    return out
end
local load_original = storage.load_lua
local baseline = copy(assert(storage:load_slot(), "No active save slot"))
local overlay = {}
local pending, frozen_sim, frozen_update
storage.load_lua = function(self, filename, force)
    if overlay[filename] ~= nil then return copy(overlay[filename]) end
    return copy(load_original(self, filename, force))
end
storage.write_lua = function(self, filename, data) overlay[filename] = copy(data); return true end
storage.remove = function() error("Deleting saves is disabled in training process") end
-- Avoid slot-saved notifications which can request a platform/cloud sync.
storage.save_slot = function(self, data, idx)
    idx = idx or self.active_slot_idx
    assert(idx, "No active save slot")
    return self:write_lua(string.format(self.SLOT_FILE_FMT, idx), data)
end
AI.training = true
AI.training_snapshot = function() return json(baseline) end
local loadout = "original"
local hero_unlocks = {}
AI.training_loadout = function(name)
    assert(name == "original" or name == "full_elora" or name == "full_ignus", "Unknown training loadout")
    loadout = name
    local data = require("data.map_data")
    for _, hero in ipairs(data.hero_data) do
        if hero_unlocks[hero.name] == nil then hero_unlocks[hero.name] = hero.available_level end
        hero.available_level = name ~= "original" and 0 or hero_unlocks[hero.name]
    end
    return json({loadout=loadout})
end
local function release_freeze()
    if frozen_sim then frozen_sim.update = frozen_update end
    frozen_sim, frozen_update = nil, nil
end
AI.training_attach = function()
    release_freeze()
    ENV.manual(true)
end
signal.register("game-start", function(s)
    if pending and s.level_idx == pending.level and s.level_mode == pending.mode then
        pending.started = true
        frozen_sim = game.simulation
        frozen_update = rawget(frozen_sim, "update")
        frozen_sim.update = function() end
    end
end)
AI.switch_level = function(level, mode, difficulty)
    assert(not pending, "Previous level transition has not finished")
    assert(type(level) == "number" and level == math.floor(level) and level >= 1 and level <= 26, "Bad level")
    assert(mode == 1 or mode == 2 or mode == 3, "Bad mode")
    assert(difficulty == 1 or difficulty == 2 or difficulty == 3, "Bad difficulty")
    local d = rawget(_G, "director") or package.loaded.director
    local active = d.active_item
    assert(active and active.done_callback, "No initialized screen to leave")
    assert(not active.done_callback_called, "Game is already leaving this screen")
    -- Restore the same baseline for every episode; victories cannot accumulate upgrades.
    overlay = {}
    local slot = copy(baseline)
    if loadout ~= "original" then
        slot.upgrades = {}
        for _, kind in ipairs({"archers","barracks","mages","engineers","rain","reinforcements"}) do
            slot.upgrades[kind] = 5
        end
        slot.heroes = slot.heroes or {}
        slot.heroes.status = slot.heroes.status or {}
        for _, hero in ipairs(require("data.map_data").hero_data) do
            slot.heroes.status[hero.name] = slot.heroes.status[hero.name] or {xp=0,skills={}}
        end
        slot.heroes.selected = loadout == "full_ignus" and "hero_ignus" or "hero_elora"
    end
    slot.difficulty = difficulty
    slot.last_victory = nil
    storage:save_slot(slot)
    if ENV then ENV.uninstall(); ENV = nil end
    release_freeze()
    pending = {level = level, mode = mode, started = false}
    -- Direct game->game transitions otherwise reuse the module table. The
    -- director loads the new store before destroying the old item, whose destroy
    -- path clears that same store. Force distinct game/GUI module instances.
    d.force_reload = true
    active.done_callback({next_item_name = "game", level_idx = level,
                        level_mode = mode, level_difficulty = difficulty})
    return json({requested = true})
end
AI.switch_status = function()
    local d = rawget(_G, "director") or package.loaded.director
    local ready = pending and pending.started and d.active_item == game
        and game.store.level_idx == pending.level and game.store.level_mode == pending.mode
    local result = {ready = ready and true or false, pending = pending ~= nil}
    if ready then
        result.level, result.mode = pending.level, pending.mode
        pending = nil
    end
    return json(result)
end
return "OK"
