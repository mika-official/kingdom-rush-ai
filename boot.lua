-- boot.lua : 由 inject.js 注入到游戏 Lua 环境中执行 (LuaJIT / Lua 5.1 语法)
-- 作用: 1) 在本机 127.0.0.1:9001 开一个"发一行 Lua、回一行结果"的服务
--       2) 提供建塔测试 T.build_test()
-- 服务挂在 director.update 上，切换关卡期间仍能响应。

assert(not (rawget(_G, "AI") and AI.training), "Training storage is active; restart the game process before re-injecting")
if rawget(_G, "AI") and AI.uninstall then pcall(AI.uninstall) end  -- 允许重复注入

AI = {}
AI.VERSION = 2
T = {}

local socket = require("socket")
local E = require("entity_db")

local PORT = 9001

-------------------------------------------------------------------------------
-- 日志 / 序列化
-------------------------------------------------------------------------------
local logbuf = {}
local function log(fmt, ...)
	local ok, s = pcall(string.format, fmt, ...)
	if not ok then s = tostring(fmt) end
	logbuf[#logbuf + 1] = s
	print("[AI] " .. s)
end

local function dump(v, d)
	d = d or 0
	if type(v) == "table" then
		if d >= 3 then return "{...}" end
		local parts = {}
		for k, x in pairs(v) do parts[#parts + 1] = tostring(k) .. "=" .. dump(x, d + 1) end
		table.sort(parts)
		return "{" .. table.concat(parts, ", ") .. "}"
	end
	return tostring(v)
end

-------------------------------------------------------------------------------
-- 游戏状态辅助
-------------------------------------------------------------------------------
local function store() return game and game.store end

-- 空塔位: tower.type == "holder" (来自 tower_holder 模板)。被封锁的塔位 type 不同, 会被排除。
local function holders()
	local out = {}
	for _, e in pairs(store().entities) do
		if e.tower and e.tower.type == "holder" and e.template_name
			and e.template_name:find("^tower_holder") then
			out[#out + 1] = e
		end
	end
	table.sort(out, function(a, b)
		return tonumber(a.tower.holder_id) < tonumber(b.tower.holder_id)
	end)
	return out
end

-- 与 GUI (TowerMenu:update) 相同的取价方式: 先跳到 build_name 指向的模板
local function price_of(tpl)
	local nt = E:get_template(tpl)
	if nt.build_name then nt = E:get_template(nt.build_name) end
	return nt.tower.price
end

local function is_locked(tpl)
	local lt = store().level and store().level.locked_towers
	if not lt then return false end
	for _, n in ipairs(lt) do if n == tpl then return true end end
	return false
end

-- 塔位被替换成新实体后 id 可能变, 所以按坐标找
local function tower_at(pos)
	local best, bd
	for _, e in pairs(store().entities) do
		if e.tower and e.pos then
			local d = math.abs(e.pos.x - pos.x) + math.abs(e.pos.y - pos.y)
			if d < 15 and (not bd or d < bd) then best, bd = e, d end
		end
	end
	return best
end

-------------------------------------------------------------------------------
-- 协程调度: 测试逻辑写成顺序代码, 每帧在 game.update 之后推进一步
-------------------------------------------------------------------------------
local co
local function run_async(fn)
	co = coroutine.create(function()
		local ok, err = pcall(fn)
		if not ok then log("测试出错: %s", tostring(err)) end
		log("=== 测试结束, 用 T.report() 查看完整日志 ===")
	end)
end

-------------------------------------------------------------------------------
-- 一次建塔尝试: 写 tower.upgrade_to, 然后逐帧观察
-------------------------------------------------------------------------------
local function attempt(h, tpl, label)
	local s = store()
	local pos = { x = h.pos.x, y = h.pos.y }
	local gold0 = s.player_gold
	local price = price_of(tpl)

	log("[%s] holder_id=%s pos=(%d,%d) 模板=%s 价格=%s 金币=%d",
		label, tostring(h.tower.holder_id), pos.x, pos.y, tpl, tostring(price), gold0)

	h.tower.upgrade_to = tpl

	local last_name, last_gold, last_up = h.template_name, gold0, tpl
	local built, final_name = false, h.template_name
	local quiet = 0   -- 连续多少帧没有任何变化

	for f = 1, 600 do
		coroutine.yield()
		local e = tower_at(pos)
		local name = e and e.template_name or "<无实体>"
		local up = e and e.tower and e.tower.upgrade_to
		local gold = s.player_gold

		if name ~= last_name or gold ~= last_gold or up ~= last_up then
			log("  帧%d tick=%s 实体=%s 金币=%d upgrade_to=%s",
				f, tostring(s.tick), name, gold, tostring(up))
			last_name, last_gold, last_up = name, gold, up
			quiet = 0
		else
			quiet = quiet + 1
		end

		-- 塔位一直是空塔位(没进入建造流程)且 2 秒左右毫无变化 -> 视为被拒绝, 不必等满
		if quiet >= 120 and name:find("^tower_holder") then
			log("  %d 帧内无任何变化, 判定为未被受理", quiet)
			break
		end

		if e and not name:find("^tower_holder") and not name:find("^tower_build") then
			built, final_name = true, name
			for _ = 1, 5 do coroutine.yield() end   -- 多等几帧, 让扣钱结算完
			break
		end
	end

	local result = { built = built, name = final_name, gold_before = gold0,
		gold_after = s.player_gold, price = price }
	log("[%s] 结果: 建成=%s 实体=%s 金币 %d -> %d (变化 %d)", label, tostring(built),
		final_name, gold0, s.player_gold, s.player_gold - gold0)
	return result
end

-------------------------------------------------------------------------------
-- 建塔测试: A=金币充足, B=金币不足
-------------------------------------------------------------------------------
function T.build_test(tpl)
	tpl = tpl or "tower_build_archer"
	if co and coroutine.status(co) ~= "dead" then return "已有测试在运行" end
	logbuf = {}

	run_async(function()
		local s = store()
		local hs = holders()
		log("paused=%s tick=%s wave=%s 金币=%s 生命=%s 空塔位=%d",
			tostring(s.paused), tostring(s.tick), tostring(s.wave_group_number),
			tostring(s.player_gold), tostring(s.lives), #hs)

		if #hs < 2 then log("需要至少 2 个空塔位, 中止"); return end
		if is_locked(tpl) then log("警告: 本关锁定了 %s, 结果可能不代表通常情况", tpl) end

		local price = price_of(tpl)
		local orig_gold = s.player_gold

		-- A: 金币充足
		if s.player_gold < price + 50 then
			s.player_gold = price + 100
			log("A 前把金币临时设为 %d", s.player_gold)
		end
		local a = attempt(hs[1], tpl, "A 金币充足")

		-- B: 金币不足 (价格 - 60, 至少为 0)
		local low = math.max(0, price - 60)
		s.player_gold = low
		log("B 前把金币设为 %d (价格 %d)", low, price)
		local b = attempt(hs[2], tpl, "B 金币不足")

		-- 判读
		log("---------- 判读 ----------")
		if a.built and a.gold_after == a.gold_before - price then
			log("A: 建塔成功且按价格扣钱(-%d), 扣钱由游戏内部完成", price)
		elseif a.built then
			log("A: 建成, 但金币变化=%d, 与价格 %d 不符, 需要进一步排查", a.gold_after - a.gold_before, price)
		else
			log("A: 没建成! 检查 paused、塔位是否可用、upgrade_to 是否被重置")
		end

		if not b.built and b.gold_after == b.gold_before then
			log("B: 金币不足时被拒绝 -> 游戏内部有校验, 可直接依赖")
		elseif b.built and b.gold_after < 0 then
			log("B: 金币不足仍建成且金币变负(%d) -> 内部无校验, 必须在 AI 侧自己判断", b.gold_after)
		elseif b.built and b.gold_after == b.gold_before then
			log("B: 金币不足仍建成且没扣钱 -> 内部无校验, 必须自己判断")
		else
			log("B: 结果不典型 built=%s 金币 %d->%d", tostring(b.built), b.gold_before, b.gold_after)
		end

		s.player_gold = orig_gold
		log("已把金币恢复为测试前的 %d", orig_gold)
	end)
	return "测试已启动, 稍后用 T.report() 查看"
end

-------------------------------------------------------------------------------
-- 手动调试用
-------------------------------------------------------------------------------
function T.report() return table.concat(logbuf, "\n") end
function T.clear() logbuf = {}; return "ok" end

function T.status()
	local s = store()
	return { gold = s.player_gold, lives = s.lives, wave = s.wave_group_number,
		wave_total = s.wave_group_total, tick = s.tick, paused = s.paused,
		holders = #holders() }
end

function T.holders()
	local out = {}
	for _, e in ipairs(holders()) do
		out[#out + 1] = string.format("holder_id=%s id=%s %s (%d,%d)",
			tostring(e.tower.holder_id), tostring(e.id), e.template_name, e.pos.x, e.pos.y)
	end
	return table.concat(out, "\n")
end

function T.towers()
	local out = {}
	for _, e in pairs(store().entities) do
		if e.tower and e.tower.type ~= "holder" and e.template_name
			and not e.template_name:find("^tower_holder") then
			out[#out + 1] = string.format("id=%s %s type=%s level=%s spent=%s (%d,%d)",
				tostring(e.id), e.template_name, tostring(e.tower.type), tostring(e.tower.level),
				tostring(e.tower.spent), e.pos.x, e.pos.y)
		end
	end
	return table.concat(out, "\n")
end

-- 手动建塔(不检查金币, 用于对比); 想要带校验的版本自己加 price 判断
function T.build(holder_id, tpl)
	for _, e in ipairs(holders()) do
		if tostring(e.tower.holder_id) == tostring(holder_id) then
			e.tower.upgrade_to = tpl or "tower_build_archer"
			return "已写入 upgrade_to -> " .. (tpl or "tower_build_archer")
		end
	end
	return "找不到空塔位 " .. tostring(holder_id)
end

function T.sell(entity_id)
	local e = store().entities[entity_id]
	if not (e and e.tower) then return "找不到塔实体 " .. tostring(entity_id) end
	e.tower.sell = true
	return "已写入 sell=true"
end

-------------------------------------------------------------------------------
-- TCP 服务: 每行一段 Lua, 先当表达式执行, 失败再当语句
-------------------------------------------------------------------------------
local srv = assert(socket.bind("127.0.0.1", PORT))
srv:settimeout(0)
local cli, buf = nil, ""
local outbound, outbound_pos = "", 1

local function disconnect()
	if cli then pcall(function() cli:close() end) end
	cli, buf, outbound, outbound_pos = nil, "", "", 1
end

local function flush_reply()
	if #outbound == 0 then return true end
	-- LuaSocket reports the last byte index, including partial timeout sends.
	local sent, err, partial = cli:send(outbound, outbound_pos)
	outbound_pos = (sent or partial or (outbound_pos - 1)) + 1
	if outbound_pos > #outbound then outbound, outbound_pos = "", 1; return true end
	if err and err ~= "timeout" then disconnect() end
	return false
end

local function exec(line)
	local f, err = loadstring("return " .. line)
	if not f then f, err = loadstring(line) end
	if not f then return "语法错误: " .. tostring(err) end
	local ok, r = pcall(f)
	if not ok then return "运行错误: " .. tostring(r) end
	local out = dump(r)
	if #out > 2000000 then out = out:sub(1, 2000000) .. "...(截断)" end
	return out
end

local function pump()
	if not cli then
		cli = srv:accept()
		if cli then cli:settimeout(0); buf = "" end
	end
	if not cli then return end
	if not flush_reply() then return end

	local data, err, part = cli:receive("*a")
	data = data or part
	if data and #data > 0 then buf = buf .. data end

	local line, rest = buf:match("^(.-)\r?\n(.*)$")
	while line do
		buf = rest
		local reply = exec(line):gsub("\n", "\\n")
		outbound, outbound_pos = reply .. "\n", 1
		if not flush_reply() then return end
		line, rest = buf:match("^(.-)\r?\n(.*)$")
	end

	if err == "closed" then
		disconnect()
	end
end

-------------------------------------------------------------------------------
-- 挂到 game.update 上
-------------------------------------------------------------------------------
local update_host = rawget(_G, "director") or package.loaded.director or game
local orig_update = update_host.update
AI.persistent = update_host ~= game
local hooked_update

function AI.uninstall()
	if update_host.update == hooked_update then update_host.update = orig_update end
	pcall(function() srv:close() end)
	if cli then pcall(function() cli:close() end) end
	co = nil
end

hooked_update = function(self, dt)
	local ok, err = pcall(pump)
	if not ok then print("[AI] pump 出错: " .. tostring(err)) end

	local r = orig_update(self, dt)

	if co and coroutine.status(co) ~= "dead" then
		local ok2, err2 = coroutine.resume(co)
		if not ok2 then log("协程出错: %s", tostring(err2)) end
	end
	return r
end
update_host.update = hooked_update

log("boot.lua 已加载, 监听 127.0.0.1:%d", PORT)
