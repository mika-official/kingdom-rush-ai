-- kr_env.lua : RL environment layer for Kingdom Rush (v1)
-- Hot-loaded by kr_client.py through the injected socket; no re-injection needed.
-- Scope: build / upgrade / sell towers, call next wave, manual tick stepping, observation,
-- powers (rain of fire / reinforcements / hero ultimate), hero movement, barracks rally points.

local previous_route_tracker=rawget(_G,'ENV') and ENV.route_tracker
if rawget(_G, "ENV") and ENV.uninstall then pcall(ENV.uninstall) end
ENV = {}
ENV.VERSION = 15
ENV.auto_wave = false     -- true: call the next wave only when its countdown is about to run out
ENV.wave_margin = 1.0     -- seconds before expiry at which the auto wave call happens
ENV.lead_front = 10       -- path nodes AHEAD of the front-most enemy where 'front' targets land (blocking point)
ENV.lead_cluster = 0      -- path nodes ahead of the densest enemy cluster

local E = require("entity_db")
local P = require("path_db")
local signal = require("hump.signal")
ENV.route_tracker=RouteEvents.install(game,signal,previous_route_tracker)
local tower_menus = require("data.tower_menus_data")
local bit = require("bit")
local U = require("utils")
local GR = require("grid_db")
local V = require("klua.vector")
local _okU, UPGR = pcall(require, "upgrades")
local _okD, DI = pcall(require, "difficulty")
local _okS, storage = pcall(require, "storage")

local KINDS = {
	archer = "tower_build_archer",
	barrack = "tower_build_barrack",
	mage = "tower_build_mage",
	engineer = "tower_build_engineer",
}
local combat_catalog_cache
local SNAP_TOL = 15        -- px, match a tower entity to its holder position
local SCAN_EVERY = 5       -- ticks between kill-accounting scans (dead_lifetime is ~120 ticks)

-------------------------------------------------------------------------------
-- JSON encoder (compact). Empty tables encode as [].
-------------------------------------------------------------------------------
local ESC = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }
local function esc(s)
	local r = s:gsub('[%c"\\]', function(c)
		return ESC[c] or string.format("\\u%04x", c:byte())
	end)
	return '"' .. r .. '"'
end

local function is_array(t)
	local n = 0
	for k in pairs(t) do
		if type(k) ~= "number" then return false end
		n = n + 1
	end
	return n == #t
end

local function jenc(v)
	local t = type(v)
	if t == "nil" then return "null"
	elseif t == "boolean" then return v and "true" or "false"
	elseif t == "number" then
		if v ~= v or v == math.huge or v == -math.huge then return "null" end
		return string.format("%.14g", v)
	elseif t == "string" then return esc(v)
	elseif t == "table" then
		local out = {}
		if is_array(v) then
			for i = 1, #v do out[i] = jenc(v[i]) end
			return "[" .. table.concat(out, ",") .. "]"
		end
		for k, x in pairs(v) do out[#out + 1] = esc(tostring(k)) .. ":" .. jenc(x) end
		return "{" .. table.concat(out, ",") .. "}"
	end
	return "null"
end
ENV.json = jenc

-------------------------------------------------------------------------------
-- Helpers
-------------------------------------------------------------------------------
local function store() return game.store end

local function is_locked(tpl)
	local lt = store().level and store().level.locked_towers
	if lt then
		for _, n in ipairs(lt) do
			if n == tpl then return true end
		end
	end
	return false
end

-- Same price lookup the GUI uses: follow build_name first.
local function price_of(tpl)
	local nt = E:get_template(tpl)
	if not nt then return nil end
	if nt.build_name then nt = E:get_template(nt.build_name) end
	return nt and nt.tower and nt.tower.price
end

local function alive(e)
	return e.enemy and e.health and not e.health.dead and e.health.hp > 0
end

local function sell_value(e)
	local s = store()
	local spent = e.tower.spent or 0
	if s.wave_group_number == 0 then return spent end
	return math.floor((e.tower.refund_factor or 0) * spent + 0.5)
end

local function can_call_wave()
	local b = game.game_gui and game.game_gui.next_wave_button
	if not b then return false end
	local ok, dis = pcall(b.is_disabled, b)
	return ok and not dis
end

-------------------------------------------------------------------------------
-- Game-over detection
-------------------------------------------------------------------------------
local flags = { victory = false, defeat = false }
local W = {}
local function wave_clear() for k in pairs(W) do W[k] = nil end end
local h_win = function() flags.victory = true end
local h_lose = function() flags.defeat = true end
local h_start = function() flags.victory = false; flags.defeat = false; wave_clear() end
signal.register("game-victory", h_win)
signal.register("game-defeat", h_lose)
signal.register("game-start", h_start)

-- Next-wave countdown. The game's own GUI (game_gui:show_wave_flags) shows a timer of
--   duration = group.interval / FPS   seconds, only for groups with group_idx > 1 (the first wave has no timer).
-- We register for the same signals and keep diagnostics (early_total / auto_gap) so the formula can be verified.
local function now() return game.store.ts or 0 end

local h_wready = function(group)
	W.ready, W.ready_ts, W.group, W.sent_by_us, W.early = true, now(), group, false, false
end
local h_wearly = function(group, reward, remaining)
	W.early, W.last_reward, W.last_remaining = true, reward, remaining
	if W.ready_ts then W.early_total = (now() - W.ready_ts) + (remaining or 0) end
end
local h_wsent = function(group)
	if W.ready and W.ready_ts and not W.early and not W.sent_by_us then W.auto_gap = now() - W.ready_ts end
	W.ready = false
end
signal.register("next-wave-ready", h_wready)
signal.register("early-wave-called", h_wearly)
signal.register("next-wave-sent", h_wsent)

local function wave_total_est()
	local g = W.group
	if not (W.ready and g and g.interval and (g.group_idx or 1) > 1) then return nil end
	return g.interval / (FPS or 30)
end

local function wave_left()
	local tot = wave_total_est()
	if not tot then return nil end
	return tot - (now() - W.ready_ts)
end

local function outcome()
	local s = store()
	local go = s.game_outcome
	if flags.victory or (go and go.victory) then return "victory" end
	if flags.defeat or (go and go.victory == false) or (s.lives or 1) <= 0 then return "defeat" end
	return nil
end

-------------------------------------------------------------------------------
-- Tower spots (holders), matched to entities by position
-------------------------------------------------------------------------------
local spots, spots_rc
local pending_cost = 0
local replacement, replacement_result

local function rebuild_spots()
	spots = {}
	local seen = {}
	for _, e in pairs(store().entities) do
		if e.tower and e.tower.holder_id and e.pos then
			local hid = tonumber(e.tower.holder_id)
			if hid and not seen[hid] then
				seen[hid] = true
				spots[#spots + 1] = { hid = hid, x = e.pos.x, y = e.pos.y }
			end
		end
	end
	table.sort(spots, function(a, b) return a.hid < b.hid end)
	spots_rc = store().restart_count or 0
end

local function ensure_spots()
	if not spots or #spots == 0 or spots_rc ~= (store().restart_count or 0) then rebuild_spots() end
end

local function spot_entities()
	ensure_spots()
	local res = {}
	for _, e in pairs(store().entities) do
		if e.tower and e.pos then
			for i, sp in ipairs(spots) do
				if math.abs(e.pos.x - sp.x) + math.abs(e.pos.y - sp.y) < SNAP_TOL then
					if not res[i] then res[i] = e end
					break
				end
			end
		end
	end
	return res
end

local function spot_state(e)
	if not e then return "missing" end
	local n = e.template_name or ""
	local tt = e.tower
	if n:find("^tower_holder") then
		if tt.type == "holder" then return "empty" end
		return "blocked"
	end
	if n:find("^tower_build") or tt.upgrade_to then return "building" end
	return "tower"
end

local function options_for(e, st)
	local out = {}
	if st ~= "empty" and st ~= "tower" then return out end
	local by_type = tower_menus[e.tower.type]
	local tm = by_type and by_type[e.tower.level]
	if not tm then return out end
	for _, item in pairs(tm) do
		if item.action == "tw_upgrade" then
			out[#out + 1] = { tpl = item.action_arg, price = price_of(item.action_arg),
				locked = is_locked(item.action_arg) }
		end
	end
	table.sort(out, function(a, b) return a.tpl < b.tpl end)
	return out
end

-------------------------------------------------------------------------------
-- Observation
-------------------------------------------------------------------------------
local function safe_progress(e)
	local n=e.nav_path
	if not n then return nil end
	local ok,a,b=pcall(function() return P:get_start_node(n.pi),P:get_end_node(n.pi) end)
	if ok and a and b then return math.max(0,math.min(1,(n.ni-a)/math.max(1,b-a))) end
end

local function enemies_list()
	local out = {}
	for _, e in pairs(store().entities) do
		if alive(e) and e.pos then
			local n = e.nav_path
			local togo
			local ahead
			if n then
				local ok, r = pcall(P.nodes_to_goal, P, n.pi, n.spi, n.ni)
				if ok then togo = r end
				if ENV.planning then
					local good,point=pcall(P.node_pos,P,n.pi,n.spi,n.ni+1)
					if good and point then ahead={x=point.x,y=point.y} end
				end
			end
			out[#out + 1] = {
				id = e.id, name = e.template_name, x = e.pos.x, y = e.pos.y,
				hp = e.health.hp, hp_max = e.health.hp_max,
				armor = e.health.armor or 0, magic_armor = e.health.magic_armor or 0,
				flying = e.vis and F_FLYING and bit.band(e.vis.flags or 0, F_FLYING) ~= 0 or false,
				pi = n and n.pi, spi = n and n.spi, ni = n and n.ni, to_goal = togo,
				gold = e.enemy.gold, lives_cost = e.enemy.lives_cost,
				spawn_wave=e._route_spawn_wave,
				speed = e.motion and e.motion.max_speed, flags = e.vis and e.vis.flags,
				ahead = ahead,
				mechanics = ENV.strategy and Strategy.mechanics(E,e) or nil,
				progress = ENV.strategy and safe_progress(e) or nil,
			}
		end
	end
	table.sort(out, function(a, b) return (a.to_goal or 1e9) < (b.to_goal or 1e9) end)
	return out
end


-------------------------------------------------------------------------------
-- v2: targets, powers, hero, barracks rally
-------------------------------------------------------------------------------
local MAX_ANCHORS = 12          -- fixed number of path anchors exposed as targets
local CLUSTER_R = 60            -- px, radius for the "densest enemy cluster" target
local DENSE_STEP = 4            -- nodes between dense path samples (used for rally candidates)
local N_TARGETS = 2 + MAX_ANCHORS   -- [1]=front enemy, [2]=enemy cluster, [3..]=path anchors

local anchors, dense, anchors_rc
local rally_cache = {}
local resolve_cache = {}
local cover_cache = {}
local SNAP_OFFSETS = { 0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15 }
local FRONT_OFFSETS = { 0, 3, 6, 9, 12, -3, -6, -9, -12 }   -- blocking points prefer to stay AHEAD of the enemy

local function safe(f, ...)
	local ok, r = pcall(f, ...)
	if ok then return r end
	return nil
end

local function node_xy(pi, ni)
	for _, spi in ipairs({ 2, 1, 3 }) do
		local ok, p = pcall(P.node_pos, P, pi, spi, ni)
		if ok and p and p.x then return p.x, p.y end
	end
end

local function build_anchors()
	anchors, dense, rally_cache, resolve_cache, cover_cache = {}, {}, {}, {}, {}
	for pi = 1, 16 do
		local ok, active = pcall(P.is_path_active, P, pi)
		if ok and active then
			local ok1, n0 = pcall(P.get_start_node, P, pi)
			local ok2, n1 = pcall(P.get_end_node, P, pi)
			if ok1 and ok2 and n0 and n1 then
				for ni = n0 + 10, n1 - 10, DENSE_STEP do
					local x, y = node_xy(pi, ni)
					if x then dense[#dense + 1] = { x = x, y = y, pi = pi, ni = ni, prog = (ni - n0) / math.max(1, n1 - n0) } end
				end
			end
		end
	end
	if #dense <= MAX_ANCHORS then
		for i, d in ipairs(dense) do anchors[i] = d end
	else
		for k = 0, MAX_ANCHORS - 1 do
			anchors[#anchors + 1] = dense[math.floor(k * #dense / MAX_ANCHORS) + 1]
		end
	end
	anchors_rc = store().restart_count or 0
end

local function ensure_anchors()
	if not anchors or anchors_rc ~= (store().restart_count or 0) then build_anchors() end
end

local function cluster_point(es)
	local best, bc, bhp = nil, 0, 0
	for _, a in ipairs(es) do
		local c, hp = 0, 0
		for _, b in ipairs(es) do
			local dx, dy = a.x - b.x, a.y - b.y
			if dx * dx + dy * dy <= CLUSTER_R * CLUSTER_R then c = c + 1; hp = hp + (b.hp or 0) end
		end
		if c > bc or (c == bc and hp > bhp) then best, bc, bhp = a, c, hp end
	end
	return best, bc, bhp
end

local function targets_list(es)
	ensure_anchors()
	local t = {}
	local function from_enemy(kind, e, lead, n, hp)
		if not e then return { kind = kind } end
		return { kind = kind, x = e.x, y = e.y, pi = e.pi, ni = e.ni and (e.ni + lead), n = n, hp = hp }
	end
	t[1] = from_enemy("front", es[1], ENV.lead_front or 0, 1, es[1] and es[1].hp)
	local cp, cn, chp = cluster_point(es)
	t[2] = from_enemy("cluster", cp, ENV.lead_cluster or 0, cn, chp)
	for i = 1, MAX_ANCHORS do
		local a = anchors[i]
		t[#t + 1] = a and { kind = "anchor", x = a.x, y = a.y, pi = a.pi, ni = a.ni } or { kind = "anchor" }
	end
	if ENV.candidates and #es>0 then
		-- Candidate policy can choose urgent, durable and poorly covered threats.
		local durable,uncovered=es[1],es[1]
		local least=math.huge
		for _,enemy in ipairs(es) do
			if enemy.hp>durable.hp then durable=enemy end
			local coverage=0
			for _,tower in pairs(store().entities) do
				if tower.tower and tower.attacks and tower.pos and not tower.tower.blocked then
					local off=tower.tower.range_offset or {x=0,y=0}
					if U.is_inside_ellipse(V.v(enemy.x,enemy.y),V.v(tower.pos.x+off.x,tower.pos.y+off.y),tower.attacks.range or 0) then coverage=coverage+1 end
				end
			end
			if coverage<least then least=coverage;uncovered=enemy end
		end
		for _,pair in ipairs({{'urgent',es[1]},{'durable',durable},{'unprotected',uncovered}}) do
			local enemy=pair[2]
			t[#t+1]={kind=pair[1],x=enemy.x,y=enemy.y,pi=enemy.pi,ni=enemy.ni,n=1,hp=enemy.hp}
		end
	end
	return t
end

local function hero_entity()
	local gg = game.game_gui
	local h = gg and gg.heroes and gg.heroes[1]
	if not h then return nil end
	return store().entities[h.hero_id]
end

local function hero_alive(e)
	return e and e.health and not e.health.dead
end

-- Same landing rules the GUI applies in PickView:on_down
local function power_point_ok(n, x, y)
	if not x then return false end
	local s = store()
	if n == 1 then
		if GR:cell_is(x, y, TERRAIN_CLIFF) or GR:cell_is(x, y, TERRAIN_FAERIE) then return false end
		if P:valid_node_nearby(x, y, 1.4285714285714286, NF_POWER_1) then return true end
		local lv = s.level
		if lv and lv.fn_can_power and lv:fn_can_power(s, GUI_MODE_POWER_1, V.v(x, y)) then return true end
		return GR:cell_is(x, y, TERRAIN_WATER) and true or false
	elseif n == 2 then
		return (P:valid_node_nearby(x, y, nil, NF_RALLY) and GR:cell_is_only(x, y, bit.bor(TERRAIN_LAND, TERRAIN_ICE))) and true or false
	elseif n == 3 then
		local he = hero_entity()
		if not hero_alive(he) then return false end
		local un = he.hero.skills.ultimate
		local ut = E:get_template(un.controller_name)
		return (not ut.can_fire_fn or ut:can_fire_fn(x, y, s)) and true or false
	end
	return false
end

-- Resolve a target to a LEGAL landing point for power n. Powers may only land on legal path points, so we
-- snap onto the enemy lane/path centre line (t.pi, t.ni) and search +-15 nodes for the first legal one.
local function resolve_point(n, t)
	if not t or not t.x then return nil end
	local key
	if t.kind == "anchor" and t.pi and t.ni then
		key = n .. ":" .. t.pi .. ":" .. t.ni
		local c = resolve_cache[key]
		if c ~= nil then return c or nil end
	end
	local result = false
	if t.pi and t.ni then
		for _, off in ipairs(t.kind == "front" and FRONT_OFFSETS or SNAP_OFFSETS) do
			local x, y = node_xy(t.pi, t.ni + off)
			if x and power_point_ok(n, x, y) then result = { x = x, y = y }; break end
		end
	end
	if not result and power_point_ok(n, t.x, t.y) then result = { x = t.x, y = t.y } end
	if key then resolve_cache[key] = result end
	return result or nil
end

local function power_button(n)
	local gg = game.game_gui
	return gg and gg["power_" .. n]
end

local function powers_info(targets)
	local out = {}
	for n = 1, 3 do
		local b = power_button(n)
		local o = { n = n, present = b ~= nil, ready = false, cooldown_left = 0, ok = {} }
		if ENV.planning then o.points={} end
		if b then
			local dis = safe(b.is_disabled, b)
			o.ready = (dis == false)
			o.mode = b.mode
			if ENV.candidates then o.cooldown_total=b.cooldown_time or 0 end
			if b.mode == "cooldown" and b.cooldown_view then
				o.cooldown_left = math.max(0, (b.cooldown_time or 0) - ((store().ts or 0) - (b.cooldown_view.start_ts or 0)))
			end
			if n == 3 and not hero_alive(hero_entity()) then o.ready = false end
		end
		for i, t in ipairs(targets) do
			local point=o.ready and safe(resolve_point,n,t) or nil
			o.ok[i] = point~=nil and point~=false
			if ENV.planning then o.points[i]=point or false end
		end
		out[n] = o
	end
	return out
end

local function hero_point_basic_ok(e, x, y)
	if not x then return false end
	local nr, ng = e.nav_rally, e.nav_grid
	if not nr or not ng then return false end
	return ((not nr.requires_node_nearby or P:valid_node_nearby(x, y, nil, NF_RALLY))
		and GR:cell_is_only(x, y, ng.valid_terrains_dest)) and true or false
end

local function hero_info(targets)
	local e = hero_entity()
	if not e then return nil end
	local un = e.hero and e.hero.skills and e.hero.skills.ultimate
	local nr = e.nav_rally
	local o = {
		id = e.id, name = e.template_name, x = e.pos.x, y = e.pos.y,
		hp = e.health.hp, hp_max = e.health.hp_max, dead = e.health.dead and true or false,
		level = e.hero and e.hero.level, xp = e.hero and e.hero.xp, ult_level = un and un.level or 0,
		rally_x = nr and nr.pos and nr.pos.x, rally_y = nr and nr.pos and nr.pos.y, move_ok = {},
	}
	for i, t in ipairs(targets) do
		o.move_ok[i] = (not o.dead) and (safe(hero_point_basic_ok, e, t.x, t.y) and true or false)
	end
	return o
end

-- Barracks rally: candidate path points inside the tower's rally ellipse, picks = {upstream, middle, downstream}
local function barrack_candidates(e)
	ensure_anchors()
	local key = tostring(e.id) .. ":" .. tostring(e.template_name)
	local c = rally_cache[key]
	if c then return c end
	c = {}
	local b = e.barrack
	local off = e.tower.range_offset or { x = 0, y = 0 }
	local rc = V.v(e.pos.x + off.x, e.pos.y + off.y)
	for _, d in ipairs(dense) do
		local okp = safe(function()
			return U.is_inside_ellipse(V.v(d.x, d.y), rc, b.rally_range)
				and (b.rally_anywhere or (P:valid_node_nearby(d.x, d.y, nil, NF_RALLY) and GR:cell_is_only(d.x, d.y, b.rally_terrains)))
		end)
		if okp then c[#c + 1] = d end
	end
	rally_cache[key] = c
	return c
end

local function rally_picks(e)
	local c = barrack_candidates(e)
	local n = #c
	if n == 0 then return { false, false, false } end
	if ENV.tactical then
		-- Score the actual blocking point, not the barracks tower's position.
		local ranked = {}
		for _, p in ipairs(c) do
			local support = 0
			for _, tower in pairs(store().entities) do
				if tower.tower and not tower.barrack and tower.attacks and tower.attacks.range then
					local off = tower.tower.range_offset or {x=0,y=0}
					if U.is_inside_ellipse(V.v(p.x,p.y), V.v(tower.pos.x+off.x,tower.pos.y+off.y), tower.attacks.range) then
						support = support + 1
					end
				end
			end
			ranked[#ranked+1] = {point=p, support=support}
		end
		table.sort(ranked, function(a,b)
			if a.support ~= b.support then return a.support > b.support end
			return a.point.prog > b.point.prog
		end)
		return {ranked[1].point, false, false}
	end
	local picks = { c[1], c[math.ceil(n / 2)], c[n] }
	for k = 2, 3 do
		if picks[k].x == picks[k - 1].x and picks[k].y == picks[k - 1].y then picks[k] = false end
	end
	return picks
end


-- Static quality of a tower spot: which part of the enemy path does a tower built here cover?
-- c = fraction of path samples inside the tower's range, p = mean progress of covered samples (0 = spawn, 1 = exit),
-- pm = furthest covered progress, r = range used. Independent of where enemies currently are.
local function kind_range(kind)
	local tpl = KINDS[kind]
	local nt = tpl and E:get_template(tpl)
	if nt and nt.build_name then nt = E:get_template(nt.build_name) end
	if not nt then return nil end
	if nt.attacks and nt.attacks.range then return nt.attacks.range end
	if nt.barrack and nt.barrack.rally_range then return nt.barrack.rally_range end
	return nil
end

local function spot_cover(sp)
	local c = cover_cache[sp.hid]
	if c then return c end
	ensure_anchors()
	c = {}
	local ctr = V.v(sp.x, sp.y)
	for kind in pairs(KINDS) do
		local r = safe(kind_range, kind)
		local n, sum, pmax = 0, 0, 0
		local paths = {}
		if r and #dense > 0 then
			for _, d in ipairs(dense) do
				local inside = safe(U.is_inside_ellipse, V.v(d.x, d.y), ctr, r)
				if inside then
					n = n + 1; sum = sum + d.prog
					paths[tostring(d.pi)] = (paths[tostring(d.pi)] or 0) + 1
					if d.prog > pmax then pmax = d.prog end
				end
			end
		end
		c[kind] = { c = (#dense > 0) and (n / #dense) or 0, p = (n > 0) and (sum / n) or 0, pm = pmax, r = r, paths=paths }
	end
	cover_cache[sp.hid] = c
	return c
end

local defense_cache,defense_cache_dense={},nil
local function defense_paths(e)
	ensure_anchors()
	local off=e.tower.range_offset or {x=0,y=0}
	local range=e.attacks and e.attacks.range or e.barrack and e.barrack.rally_range or 0
	local center=V.v(e.pos.x+off.x,e.pos.y+off.y)
	if defense_cache_dense~=dense then defense_cache={};defense_cache_dense=dense end
	local key=tostring(center.x)..':'..tostring(center.y)..':'..tostring(range)
	if defense_cache[key] then return defense_cache[key] end
	local out={}
	for _,d in ipairs(dense) do
		if U.is_inside_ellipse(V.v(d.x,d.y),center,range) then
			local key=tostring(d.pi)
			out[key]=out[key] or {0,0,0}
			local bin=math.min(3,math.floor(d.prog*3)+1)
			out[key][bin]=out[key][bin]+1
		end
	end
	defense_cache[key]=out
	return out
end

local length_cache,length_cache_dense={},nil
local function defense_lengths(e)
	ensure_anchors()
	if length_cache_dense~=dense then length_cache={};length_cache_dense=dense end
	local off=e.tower.range_offset or {x=0,y=0}
	local center=V.v(e.pos.x+off.x,e.pos.y+off.y)
	local range=e.attacks and e.attacks.range or e.barrack and e.barrack.rally_range or 0
	local key=tostring(center.x)..':'..tostring(center.y)..':'..tostring(range)
	if length_cache[key] then return length_cache[key] end
	local out,previous={},{}
	for _,d in ipairs(dense) do
		local prev=previous[d.pi]
		local inside=U.is_inside_ellipse(V.v(d.x,d.y),center,range)
		if prev then
			local factor=(inside and 0.5 or 0)+(prev.inside and 0.5 or 0)
			local distance=math.sqrt((d.x-prev.x)^2+(d.y-prev.y)^2)*factor
			local id=tostring(d.pi);out[id]=(out[id] or 0)+distance
		end
		previous[d.pi]={x=d.x,y=d.y,inside=inside}
	end
	length_cache[key]=out
	return out
end

local sample_cache,sample_cache_dense={},nil
local function defense_samples(e)
	ensure_anchors()
	if sample_cache_dense~=dense then sample_cache={};sample_cache_dense=dense end
	local off=e.tower.range_offset or {x=0,y=0}
	local center=V.v(e.pos.x+off.x,e.pos.y+off.y)
	local range=e.attacks and e.attacks.range or 0
	local key=tostring(center.x)..':'..tostring(center.y)..':'..tostring(range)
	if sample_cache[key] then return sample_cache[key] end
	local out={}
	for _,d in ipairs(dense) do
		if U.is_inside_ellipse(V.v(d.x,d.y),center,range) then
			local id=tostring(d.pi);out[id]=out[id] or {}
			out[id][#out[id]+1]=d.prog
		end
	end
	sample_cache[key]=out;return out
end

local route_geometry_cache,route_geometry_dense
local function route_geometry()
	ensure_anchors()
	if route_geometry_cache and route_geometry_dense==dense then return route_geometry_cache end
	local out={}
	for pi=1,16 do
		local good,active=pcall(P.is_path_active,P,pi)
		if good and active then
			local ok,a,b=pcall(function() return P:get_start_node(pi),P:get_end_node(pi) end)
			if ok and a and b and b>a then
				local row={segments={},partial=true,total_length=0,node_count=P:get_end_node(pi)-P:get_start_node(pi)+1}
				local previous
				local function append(ni)
					local x,y=node_xy(pi,ni)
					if not x then return end
					local progress=(ni-a)/(b-a)
					if previous then
						local length=math.sqrt((x-previous.x)^2+(y-previous.y)^2)
						row.segments[#row.segments+1]={start=previous.progress,['end']=progress,length=length}
						row.total_length=row.total_length+length
					end
					previous={x=x,y=y,progress=progress}
				end
				append(a)
				for ni=a+10,b-10,DENSE_STEP do append(ni) end
				append(b)
				out[tostring(pi)]=row
			end
		end
	end
	out['0']={segments={},partial=true,total_length=0}
	route_geometry_cache,route_geometry_dense=out,dense
	return out
end

local development_cache,development_cache_dense={},nil
local function development_for(name,sp)
    if ENV.light_observations then return nil end
	ensure_anchors()
	if development_cache_dense~=dense then development_cache={};development_cache_dense=dense end
	local key=tostring(name)..':'..sp.x..':'..sp.y
	if development_cache[key] then return development_cache[key] end
    local plans=Strategy.development(E,tower_menus,name,is_locked,price_of,ENV.route_planning)
	for _,plan in ipairs(plans) do
		if plan.combat then
			local nt=E:get_template(plan.tpl)
			local placed={tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack}
			plan.target_paths=defense_paths(placed)
			plan.target_lengths=defense_lengths(placed)
			plan.target_samples=defense_samples(placed)
		end
	end
	development_cache[key]=plans
	return plans
end

function ENV.special_forecast(index)
	return jenc(Strategy.special(E,store(),index))
end

local function forecast(offset)
	local db = require("wave_db").db
	assert(db and db.groups, "Tactical observation requires loaded wave groups")
	local index = (store().wave_group_number or 0) + (offset or 1)
	local group = db.groups[index]
	local out = {index=index, available=group ~= nil, spawns={}}
	for _, wave in pairs(group and group.waves or {}) do
		local spawn_cursor=(wave.delay or 0)/FPS
		for _, spawn in ipairs(wave.spawns or {}) do
			local e = E:get_template(spawn.creep)
			assert(e and e.health, "Missing enemy template for wave forecast: " .. tostring(spawn.creep))
			local predicted_hp=e.health.hp_max or e.health.hp or 0
			if ENV.candidates then predicted_hp=Strategy.health_value(predicted_hp,store().level_difficulty) end
			out.spawns[#out.spawns+1] = {
				name=spawn.creep, count=spawn.max or 0, interval=spawn.interval or 0,
				pi=wave.path_index, armor=e.health.armor or 0, magic_armor=e.health.magic_armor or 0,
				flying=e.vis and F_FLYING and bit.band(e.vis.flags or 0,F_FLYING) ~= 0 or false,
				hp=predicted_hp,lives_cost=e.enemy and e.enemy.lives_cost,
				speed=ENV.planning and e.motion and e.motion.max_speed or nil,
				delay=ENV.strategy and (wave.delay or 0)/30 or nil,
				spawn_start_seconds=ENV.route_planning and spawn_cursor or nil,
				timing_known=ENV.route_planning and (spawn.min==nil or spawn.min==spawn.max) or false,
				interval_next_seconds=ENV.route_planning and (spawn.interval_next or 0)/FPS or nil,
				interval_seconds=ENV.strategy and (spawn.interval or 0)/30 or nil,
				mechanics=ENV.strategy and Strategy.mechanics(E,e) or nil}
			spawn_cursor=spawn_cursor+(spawn.max or 0)*(spawn.interval or 0)/FPS+(spawn.interval_next or 0)/FPS
		end
	end
	if ENV.strategy then
		out.special=Strategy.special(E,store(),index)
		for _,sp in ipairs(out.special.spawns) do out.spawns[#out.spawns+1]=sp end
		out.available=out.available or #out.special.spawns>0
	end
	return out
end

function ENV.obs()
	local s = store()
	local level = s.level or {}
	local ents = spot_entities()
	local sp_out = {}
	for i, sp in ipairs(spots) do
		local e = ents[i]
		local st = spot_state(e)
		local o = { hid = sp.hid, x = sp.x, y = sp.y, state = st, options = {}, cov = safe(spot_cover, sp) }
		if e then
			o.name = e.template_name
			o.kind = e.tower.type
			o.level = e.tower.level
			o.spent = e.tower.spent
			o.options = options_for(e, st)
			for _, op in ipairs(o.options) do
				op.ok = (not op.locked) and op.price ~= nil and (s.player_gold - pending_cost) >= op.price
				if ENV.candidates then
					local nt=E:get_template(op.tpl)
					if nt and nt.build_name then nt=E:get_template(nt.build_name) end
					if nt and nt.tower then
						if ENV.planning then
							op.combat=Strategy.combat(E,nt)
							op.development=development_for(nt.template_name or op.tpl,sp)
							if not ENV.light_observations then
								op.target_samples=defense_samples({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack})
								op.target_lengths=defense_lengths({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack})
							end
						end
						op.target_range=nt.attacks and nt.attacks.range or nt.barrack and nt.barrack.rally_range or 0
						op.target_paths=defense_paths({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack})
					end
				end
			end
			if st == "tower" then
				if ENV.planning then
					o.combat=Strategy.combat(E,e)
					if not ENV.light_observations then o.defense_samples=defense_samples(e) end
					o.development=development_for(e.template_name,sp)
					if not ENV.light_observations then o.defense_lengths=defense_lengths(e) end
				end
				if ENV.strategy then
					o.skills=Strategy.skills(e,tower_menus,s.player_gold-pending_cost)
					if ENV.planning then
						for _,sk in ipairs(o.skills) do sk.combat=Strategy.combat(E,e,sk.name,sk.level+1) end
					end
					o.defense_paths=defense_paths(e)
				end
				o.sell = sell_value(e)
				o.blocked = e.tower.blocked and true or false
				if ENV.tactical then
					o.replace_options = {}
					for kind,tpl in pairs(KINDS) do
						local option={kind=kind,price=price_of(tpl),locked=is_locked(tpl)}
						if ENV.candidates then
							local nt=E:get_template(tpl)
							if nt and nt.build_name then nt=E:get_template(nt.build_name) end
							if nt then
								if ENV.planning then
									option.combat=Strategy.combat(E,nt)
									if not ENV.light_observations then option.target_samples=defense_samples({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack}) end
									option.development=development_for(nt.template_name or tpl,sp)
									if not ENV.light_observations then option.target_lengths=defense_lengths({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack}) end
								end
								option.target_range=nt.attacks and nt.attacks.range or nt.barrack and nt.barrack.rally_range or 0
								option.target_paths=defense_paths({tower=nt.tower,pos=sp,attacks=nt.attacks,barrack=nt.barrack})
							end
						end
						o.replace_options[#o.replace_options+1]=option
					end
				end
				o.attack_range = e.attacks and e.attacks.range
				local off = e.tower.range_offset or {x=0,y=0}
				o.attack_x, o.attack_y = e.pos.x+off.x, e.pos.y+off.y
				if e.barrack and e.tower.range_offset then
					local picks = safe(rally_picks, e) or { false, false, false }
					o.rally_ok = { picks[1] and true or false, picks[2] and true or false, picks[3] and true or false }
					if ENV.tactical then o.rally_points = picks end
					if e.barrack.rally_pos then o.rally_x, o.rally_y = e.barrack.rally_pos.x, e.barrack.rally_pos.y end
				end
			end
		end
		sp_out[#sp_out + 1] = o
	end
	local es = enemies_list()
	local tg = targets_list(es)
	local active_spawners = 0
	for _ in pairs(s.waves_active or {}) do active_spawners = active_spawners + 1 end
	if ENV.candidates and not ENV.light_observations and not combat_catalog_cache then combat_catalog_cache=Strategy.catalog(E) end
	return {
		tick = s.tick, gold = s.player_gold, lives = s.lives,
		tower_damage = ENV.damage_tracker and ENV.damage_tracker:status() or nil,
		skill_mechanisms_version=ENV.skill_mechanisms and 1 or nil,
		support_mechanisms_version=ENV.support_mechanisms and 1 or nil,
		light_observations_version=ENV.light_observations and 1 or nil,
		level_idx = s.level_idx, level_mode = s.level_mode, difficulty = s.level_difficulty,
		locked_hero = level.locked_hero or false, locked_towers = level.locked_towers or {},
		max_upgrade_level = level.max_upgrade_level,
		wave = s.wave_group_number, wave_total = s.wave_group_total,
		forecast = ENV.tactical and forecast() or nil,
		forecast_second = ENV.candidates and forecast(2) or nil,
		strategy_version = ENV.strategy and 1 or nil,
		planning_version = ENV.planning and 1 or nil,
		route_geometry = ENV.route_planning and route_geometry() or nil,
		combat_catalog = ENV.candidates and not ENV.light_observations and combat_catalog_cache or nil,
		rain_profile = ENV.planning and Strategy.rain(E,s) or nil,
		special_current = ENV.strategy and Strategy.special(E,s,s.wave_group_number or 0) or nil,
		waves_finished = s.waves_finished and true or false,
		level_run_complete = s.level and s.level.run_complete and true or false,
		enemies_alive = #es, active_spawners = active_spawners,
		wave_spawn_pending = s.wave_spawn_thread ~= nil,
		can_call_wave = can_call_wave(), early_reward = s.early_wave_reward,
		next_wave_left = wave_left(), next_wave_total = wave_total_est(), wave_group_idx = W.group and W.group.group_idx,
		wave_check = { early_total = W.early_total, auto_gap = W.auto_gap, early_remaining = W.last_remaining },
		outcome = outcome(), spots = sp_out, enemies = es,
		replacement_pending = replacement ~= nil, replacement_result = replacement_result,
		targets = tg, powers = powers_info(tg), hero = hero_info(tg),
	}
end

function ENV.obs_json() return jenc(ENV.obs()) end

function ENV.info()
	local s = store()
	ensure_spots()
	ensure_anchors()
	return jenc({ version = ENV.VERSION, tick_length = TICK_LENGTH, n_spots = #spots,
		rng_isolation_version = AI and AI.rng_isolation_version or 0,
		n_anchors = #anchors, n_dense = #dense, has_hero = hero_entity() ~= nil, n_targets = N_TARGETS,
		level_idx = s.level_idx, level_mode = s.level_mode, difficulty = s.level_difficulty,
		restart_count = s.restart_count or 0 })
end

-------------------------------------------------------------------------------
-- Actions (the game does NOT validate gold itself, so we do it here)
-------------------------------------------------------------------------------
local function fail(reason) return { ok = false, reason = reason } end

local function spot_index(hid)
	ensure_spots()
	for i, sp in ipairs(spots) do
		if sp.hid == hid then return i end
	end
end

local function do_upgrade(hid, tpl, expect_state)
	local i = spot_index(hid)
	if not i then return fail("no_such_spot") end
	local e = spot_entities()[i]
	local st = spot_state(e)
	if st ~= expect_state then return fail("bad_state:" .. st) end
	if e.tower.upgrade_to or e.tower.sell then return fail("busy") end
	local found
	for _, op in ipairs(options_for(e, st)) do
		if op.tpl == tpl then found = op break end
	end
	if not found then return fail("not_an_option") end
	if found.locked then return fail("locked") end
	if not found.price then return fail("no_price") end
	if store().player_gold - pending_cost < found.price then return fail("no_gold") end
	e.tower.upgrade_to = tpl
	pending_cost = pending_cost + found.price
	return { ok = true, cost = found.price }
end

function ENV.build(hid, kind)
	local tpl = KINDS[kind]
	if not tpl then return fail("bad_kind") end
	return do_upgrade(hid, tpl, "empty")
end

function ENV.upgrade(hid, tpl) return do_upgrade(hid, tpl, "tower") end

function ENV.tower_skill(hid,name)
	if not ENV.strategy or outcome() or replacement then return fail('unavailable') end
	local i=spot_index(hid)
	return Strategy.buy_skill(store(),i and spot_entities()[i],name,tower_menus,pending_cost,signal)
end

function ENV.sell(hid)
	local i = spot_index(hid)
	if not i then return fail("no_such_spot") end
	local e = spot_entities()[i]
	local st = spot_state(e)
	if st ~= "tower" then return fail("bad_state:" .. st) end
	if e.tower.upgrade_to or e.tower.sell then return fail("busy") end
	e.tower.sell = true
	return { ok = true, refund = sell_value(e) }
end

function ENV.replace(hid, kind)
	if not ENV.tactical or outcome() then return fail("unavailable") end
	if replacement then return fail("replacement_pending") end
	local i = spot_index(hid)
	local e = i and spot_entities()[i]
	local tpl = KINDS[kind]
	local price = tpl and price_of(tpl)
	if not e or spot_state(e) ~= "tower" or e.tower.blocked then return fail("bad_source") end
	if not price or is_locked(tpl) then return fail("locked_replacement") end
	if store().player_gold + sell_value(e) < price then return fail("not_enough_gold") end
	local sold = ENV.sell(hid)
	if not sold.ok then return sold end
	replacement = {hid=hid,kind=kind,tpl=tpl,start_tick=store().tick,loss=math.max(0,(e.tower.spent or 0)-sold.refund)}
	replacement_result = nil
	return {ok=true,pending=true}
end

local function advance_replacement()
	local p = replacement
	if not p then return end
	local i = spot_index(p.hid)
	local e = i and spot_entities()[i]
	if e and spot_state(e) == "empty" then
		local result = ENV.build(p.hid,p.kind)
		replacement_result = {ok=result.ok,reason=result.reason,hid=p.hid,kind=p.kind,loss=p.loss,started_tick=p.start_tick}
		replacement = nil
	elseif store().tick - p.start_tick > 120 then
		replacement_result = {ok=false,reason="replacement_timeout",hid=p.hid,kind=p.kind}
		replacement = nil
	end
end

function ENV.call_wave()
	if outcome() then return fail("game_over") end
	if not can_call_wave() then return fail("not_ready") end
	store().send_next_wave = true
	W.sent_by_us = true
	return { ok = true }
end

-------------------------------------------------------------------------------
-- Manual stepping: freeze the game's own real-time updates, tick on demand.
-------------------------------------------------------------------------------
local frozen_sim, frozen_raw

function ENV.manual(on)
	local sim = game.simulation
	if on then
		if not frozen_sim then
			frozen_sim, frozen_raw = sim, rawget(sim, "update")
			sim.update = function() end
		end
	elseif frozen_sim then
		frozen_sim.update = frozen_raw
		frozen_sim, frozen_raw = nil, nil
	end
	return on and true or false
end

local unblock_click_ts = {}
local function click_tower_locks(s)
	if not ENV.tactical then return 0 end
	local clicks, boss_clicks, live = 0, 0, {}
	local jt_prompt = false
	for _, e in pairs(s.entities) do
		if e.template_name == "decal_jt_tap" and not e.pending_removal then jt_prompt = true end
	end
	for id, e in pairs(s.entities) do
		-- Only these two known clickable modifiers; never unlock cutscenes,
		-- map obstacles, paid tower holders or non-clickable timed debuffs.
		local boss_finish = e.template_name == "eb_jt" and jt_prompt
			and e.health and e.health.ignore_damage and not e.health.dead
		if e.template_name == "mod_jt_tower" or e.template_name == "mod_veznan_tower" or boss_finish then
			live[id] = true
			local ts = s.ts or 0
			if not e.pending_removal and e.ui and e.ui.can_click and not e.ui.clicked
				and ts >= (unblock_click_ts[id] or -1) then
				e.ui.clicked = true -- same signal as the game's normal mouse handler
				unblock_click_ts[id] = ts + 0.2
				if boss_finish then boss_clicks = boss_clicks + 1 else clicks = clicks + 1 end
			end
		end
	end
	for id in pairs(unblock_click_ts) do
		if not live[id] then unblock_click_ts[id] = nil end
	end
	return clicks, boss_clicks
end

function ENV.step(n)
	local s, sim = store(), game.simulation
	local g0, l0 = s.player_gold, s.lives
	local seen, counted, pre, kills, kill_gold = {}, {}, {}, 0, 0
	for _, e in pairs(s.entities) do
		if e.enemy and e.health then pre[e.id] = true end
	end

	local function scan()
		for _, e in pairs(s.entities) do
			if e.enemy and e.health then
				if e.health.dead or e.health.hp <= 0 then
					-- killed during this step: either seen alive earlier in the step,
					-- or it did not exist when the step began (born and died in between scans)
					if not counted[e.id] and (seen[e.id] or not pre[e.id]) then
						counted[e.id] = true
						kills = kills + 1
						kill_gold = kill_gold + (seen[e.id] or e.enemy.gold or 0)
					end
				else
					seen[e.id] = e.enemy.gold or 0
				end
			end
		end
	end

	scan()
	local ran, unblock_clicks, boss_finish_clicks = 0, 0, 0
	for i = 1, n do
		if outcome() then break end
		s.dt = TICK_LENGTH
		s.ts = (s.ts or 0) + TICK_LENGTH
		local lock_clicks, finish_clicks = click_tower_locks(s)
		unblock_clicks = unblock_clicks + lock_clicks
		boss_finish_clicks = boss_finish_clicks + (finish_clicks or 0)
		advance_replacement()
		sim:do_tick()
		ran = i
		if ENV.auto_wave and W.ready and not W.sent_by_us then
			local left = wave_left()
			if left and left <= ENV.wave_margin and can_call_wave() then
				s.send_next_wave = true
				W.sent_by_us = true
			end
		end
		if i % SCAN_EVERY == 0 then scan() end
		if outcome() then break end
	end
	scan()
	pending_cost = 0
	local result=ENV.route_tracker:drain(l0,s.lives)
	result.ticks=ran;result.d_gold=s.player_gold-g0;result.d_lives=s.lives-l0
	result.kills=kills;result.kill_gold=kill_gold;result.unblock_clicks=unblock_clicks;result.boss_finish_clicks=boss_finish_clicks
	result.route_cohort_status=ENV.route_tracker:status()
	return result
end


function ENV.power(n, ti)
	if outcome() then return fail("game_over") end
	local b = power_button(n)
	if not b then return fail("no_power") end
	if safe(b.is_disabled, b) ~= false then return fail("not_ready") end
	if n == 3 and not hero_alive(hero_entity()) then return fail("hero_dead") end
	local t = targets_list(enemies_list())[ti]
	if not t or not t.x then return fail("no_target") end
	local pt = safe(resolve_point, n, t)
	if not pt then return fail("invalid_point") end
	b:fire(pt.x, pt.y)
	return { ok = true, x = pt.x, y = pt.y }
end

function ENV.hero_move(ti)
	if outcome() then return fail("game_over") end
	local e = hero_entity()
	if not e then return fail("no_hero") end
	if not hero_alive(e) then return fail("hero_dead") end
	local t = targets_list(enemies_list())[ti]
	if not t or not t.x then return fail("no_target") end
	local x, y = t.x, t.y
	if not safe(hero_point_basic_ok, e, x, y) then return fail("invalid_point") end
	local ng, nr = e.nav_grid, e.nav_rally
	local wp
	if not ng.ignore_waypoints then
		wp = safe(function() return GR:find_waypoints(e.pos, nr.pos, V.v(x, y), ng.valid_terrains) end)
	end
	local tele = e.teleport and not e.teleport.disabled and e.teleport.min_distance < V.dist(x, y, e.pos.x, e.pos.y)
	if not (tele or ng.ignore_waypoints or wp) then return fail("unreachable") end
	if not ng.ignore_waypoints then ng.waypoints = wp end
	nr.new = true
	nr.pos = V.v(x, y)
	nr.center = V.v(x, y)
	return { ok = true }
end

function ENV.rally(hid, k)
	local i = spot_index(hid)
	if not i then return fail("no_such_spot") end
	local e = spot_entities()[i]
	if spot_state(e) ~= "tower" then return fail("bad_state") end
	if not (e.barrack and e.tower.range_offset) then return fail("not_barracks") end
	local p = (safe(rally_picks, e) or {})[k]
	if not p then return fail("no_point") end
	e.barrack.rally_pos = V.v(p.x, p.y)
	e.barrack.rally_new = true
	return { ok = true }
end

-- Diagnostics for landing-point legality: shows which predicate fails (run in a real game and read the output).
function ENV.diag_power()
	ensure_anchors()
	local function b(f, ...)
		local ok, r = pcall(f, ...)
		if not ok then return "error" end
		return r and true or false
	end
	local function probe(x, y)
		return {
			x = x, y = y,
			cliff = b(GR.cell_is, GR, x, y, TERRAIN_CLIFF), faerie = b(GR.cell_is, GR, x, y, TERRAIN_FAERIE),
			water = b(GR.cell_is, GR, x, y, TERRAIN_WATER),
			nf_p1_r = b(P.valid_node_nearby, P, x, y, 1.4285714285714286, NF_POWER_1),
			nf_p1_def = b(P.valid_node_nearby, P, x, y, nil, NF_POWER_1),
			nf_rally = b(P.valid_node_nearby, P, x, y, nil, NF_RALLY),
			p1_ok = b(power_point_ok, 1, x, y), p2_ok = b(power_point_ok, 2, x, y),
		}
	end
	local out = { dense = {}, enemies = {}, snapped = {}, lead_front = ENV.lead_front, lead_cluster = ENV.lead_cluster }
	local n = math.min(#dense, 10)
	for k = 1, n do
		local d = dense[math.floor((k - 1) * #dense / math.max(n, 1)) + 1]
		local o = probe(d.x, d.y); o.pi, o.ni = d.pi, d.ni
		out.dense[#out.dense + 1] = o
	end
	local es = enemies_list()
	for k = 1, math.min(#es, 5) do
		local e = es[k]
		local o = probe(e.x, e.y); o.pi, o.spi, o.ni = e.pi, e.spi, e.ni
		out.enemies[#out.enemies + 1] = o
		local sx, sy = node_xy(e.pi, e.ni)
		if sx then
			local so = probe(sx, sy); so.pi, so.ni = e.pi, e.ni
			out.snapped[#out.snapped + 1] = so
		end
	end
	return jenc(out)
end


-------------------------------------------------------------------------------
-- v4: out-of-level upgrades (stars). In-memory only: the save file is never written.
-- Same recipe the map screen uses: E:load() (pristine templates), set_levels, patch_templates.
-------------------------------------------------------------------------------
local UP_CLASSES = { "archers", "barracks", "mages", "engineers", "rain", "reinforcements" }

local function saved_levels()
	if not storage then return nil end
	local slot = safe(function() return storage:load_slot() end)
	if not (slot and slot.upgrades) then return nil end
	local out = {}
	for _, c in ipairs(UP_CLASSES) do out[c] = slot.upgrades[c] or 0 end
	return out
end

function ENV.meta_info()
	if not (UPGR and DI) then return jenc({ available = false }) end
	local costs = {}
	for _, c in ipairs(UP_CLASSES) do
		local arr = {}
		for _, u in pairs(UPGR.list) do
			if u.class == c then arr[u.level] = u.price end
		end
		costs[c] = arr
	end
	local cur = {}
	for _, c in ipairs(UP_CLASSES) do cur[c] = UPGR.levels[c] or 0 end
	local stars
	if storage then
		stars = safe(function() return U.count_stars(storage:load_slot()) end)
	end
	return jenc({ available = true, classes = UP_CLASSES, costs = costs, levels = cur, saved = saved_levels(),
		stars = stars, max_level = 5, difficulty = store().level_difficulty })
end

function ENV.apply_upgrades(levels)
	combat_catalog_cache=nil
	if not (UPGR and DI) then return "unavailable" end
	local lv = {}
	for _, c in ipairs(UP_CLASSES) do
		lv[c] = math.max(0, math.min(5, math.floor(tonumber(levels and levels[c]) or 0)))
	end
	E:load()
	UPGR:set_levels(lv)
	local di = store().level_difficulty
	if di then DI:set_level(di) end
	UPGR:patch_templates(5)
	DI:patch_templates()
	rally_cache, resolve_cache, cover_cache = {}, {}, {}
	return "ok"
end

function ENV.restore_upgrades()
	local sv = saved_levels()
	if not sv then return "no_saved_levels" end
	return ENV.apply_upgrades(sv)
end

-- Cheap observable that changes with the archers / barracks upgrades (used to verify they take effect).
function ENV.probe_upgrade()
	local a = E:get_template("tower_archer_1")
	local b = E:get_template("soldier_militia")
	local l1 = a and a.attacks and a.attacks.list and a.attacks.list[1]
	return jenc({ archer_range = a and a.attacks and a.attacks.range, archer_list_range = l1 and l1.range,
		archer_refund = a and a.tower and a.tower.refund_factor,
		militia_hp = b and b.health and b.health.hp_max, levels = (UPGR and UPGR.levels) })
end

function ENV.configure(t)
	combat_catalog_cache=nil
	development_cache={}
	for _, k in ipairs({ "auto_wave", "wave_margin", "lead_front", "lead_cluster", "tactical", "strategy", "candidates", "planning", "route_planning", "tower_damage", "skill_mechanisms", "light_observations", "support_mechanisms" }) do
		if t and t[k] ~= nil then ENV[k] = t[k] end
	end
	if ENV.tower_damage and not ENV.damage_tracker then ENV.damage_tracker=DamageAudit.install(game,Strategy) end
	if not ENV.tower_damage and ENV.damage_tracker then ENV.damage_tracker:uninstall();ENV.damage_tracker=nil end
	return jenc({ auto_wave = ENV.auto_wave, wave_margin = ENV.wave_margin,
		lead_front = ENV.lead_front, lead_cluster = ENV.lead_cluster })
end

function ENV.reset_damage_audit()
	if ENV.damage_tracker then ENV.damage_tracker:uninstall() end
	ENV.tower_damage=true;ENV.damage_tracker=DamageAudit.install(game,Strategy)
	return jenc(true)
end

-- actions: list like {{"build",3,"archer"},{"upgrade",3,"tower_archer_2"},{"sell",3},{"wave"}}
function ENV.turn(actions, n)
	local results = {}
	for i, a in ipairs(actions or {}) do
		local k = a[1]
		local r
		if k == "build" then r = ENV.build(a[2], a[3])
		elseif k == "upgrade" then r = ENV.upgrade(a[2], a[3])
		elseif k == "tower_skill" then r = ENV.tower_skill(a[2],a[3])
		elseif k == "sell" then r = ENV.sell(a[2])
		elseif k == "replace" then r = ENV.replace(a[2],a[3])
		elseif k == "wave" then r = ENV.call_wave()
		elseif k == "power" then r = ENV.power(a[2], a[3])
		elseif k == "hero_move" then r = ENV.hero_move(a[2])
		elseif k == "rally" then r = ENV.rally(a[2], a[3])
		elseif k == "noop" then r = { ok = true }
		else r = fail("bad_action") end
		results[i] = r
	end
	local st = ENV.step(n or 1)
	return jenc({ results = results, step = st, obs = ENV.obs() })
end

-- game:restart() reinitializes the GUI without destroying the previous instance.
-- Dispose its timers and signal handlers BEFORE clearing the terminal result;
-- otherwise the delayed victory view can read the next episode's nil outcome.
-- Hook the restart itself so both ENV.reset and the GUI Restart button are safe.
local restart_raw = rawget(game, "restart")
local restart_original = game.restart
local function restart_clean(self, ...)
	self.game_gui:destroy()
	self.store.game_outcome = nil
	self.store.paused = false
	return restart_original(self, ...)
end
game.restart = restart_clean

-- Uses the normal restart entry point, preserving game-restart notifications.
function ENV.reset()
	ENV.route_tracker:uninstall()
	if ENV.damage_tracker then ENV.damage_tracker:uninstall();ENV.damage_tracker=nil end
	combat_catalog_cache=nil
	replacement, replacement_result = nil, nil
	unblock_click_ts = {}
	spots = nil
	anchors = nil
	rally_cache = {}
	resolve_cache = {}
	pending_cost = 0
	game.game_gui:restart_game()
	ENV.route_tracker=RouteEvents.install(game,signal)
	if ENV.tower_damage then ENV.damage_tracker=DamageAudit.install(game,Strategy) end
	return "restart requested"
end

function ENV.uninstall()
	if ENV.route_tracker then ENV.route_tracker:uninstall() end
	if ENV.damage_tracker then ENV.damage_tracker:uninstall();ENV.damage_tracker=nil end
	ENV.manual(false)
	if game.restart == restart_clean then game.restart = restart_raw end
	signal.remove("game-victory", h_win)
	signal.remove("game-defeat", h_lose)
	signal.remove("game-start", h_start)
	signal.remove("next-wave-ready", h_wready)
	signal.remove("early-wave-called", h_wearly)
	signal.remove("next-wave-sent", h_wsent)
end

return "kr_env loaded"
