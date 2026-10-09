-- v4 helpers. Loaded alongside kr_env.lua, never installed into the game files.
local S = {}
local bit = require('bit')

function S.health_value(value,difficulty)
    if type(value)=='table' then value=value[difficulty or 2] end
    return type(value)=='number' and value or nil
end

-- Planning observations are read-only. Script-specific effects remain unknown
-- unless their template fields are sufficient to describe them.
local attack_powers={arcane_wizard={[2]='disintegrate',[3]='teleport'},
    sorcerer={[2]='polymorph'},bfg={[2]='missile',[3]='cluster'},
    tesla={[2]='overcharge'},ranger={[2]='thorn'},musketeer={[2]='sniper',[3]='shrapnel'}}
local function numeric(v) return type(v)=='number' and v or nil end
local function has(v,flag) return flag and bit.band(v or 0,flag)~=0 or false end
local function mechanisms(E,e,level)
    local out={};local list=e.attacks and e.attacks.list or {};local base=list[1] or {}
    local function add(name,kind,values)
        local r=values or {};r.name=name;r.kind=kind;r.level=level(name);r.known=true;r.partial=true
        out[#out+1]=r;return r
    end
    for name,p in pairs(e.powers or {}) do
        if name=='poison' then
            local mod=E:get_template(p.mod);local d=mod and mod.dps or {};local m=mod and mod.modifier or {}
            add(name,'dot',{damage_per_second=((d.damage_min or 0)+(d.damage_max or 0))/2+(d.damage_inc or 0)*level(name),
                tick_interval=d.damage_every or 1,duration=m.duration or 0,cooldown=base.cooldown or 1,
                flying=true,refreshes=true,channel='true'})
        elseif name=='thorn' then
            local at=E:get_template(p.aura);local a=at and at.aura or {};local mt=a.mod and E:get_template(a.mod);local m=mt and mt.modifier or {}
            add(name,'root',{duration=(m.duration or 0)+(m.duration_inc or 0)*level(name),
                damage_per_second=mt and ((mt.damage_min or 0)+(mt.damage_max or 0))/2/(mt.damage_every or 1) or 0,
                radius=a.radius or 0,max_targets=(a.max_count or 0)+(a.max_count_inc or 0)*level(name),
                cooldown=a.cooldown or 9,min_targets=a.min_count or 1,flying=false,channel='physical',boss_immune=true})
        elseif name=='sniper' then
            local a=list[2] or {}
            add(name,'execute',{chance=(p.instakill_chance_inc or 0)*level(name),
                hp_fraction=(p.damage_factor_inc or 0)*level(name),cooldown=a.cooldown or 14,
                range=a.range or e.attacks.range,flying=not has(a.vis_bans,F_FLYING),boss_immune=true})
        elseif name=='shrapnel' then
            local a;for _,v in ipairs(list) do if v.power_name=='shrapnel' then a=v end end;a=a or {}
            local b=a.bullet and E:get_template(a.bullet);local d=b and b.bullet or {}
            add(name,'splash',{damage=((d.damage_min or 0)+(d.damage_max or 0))/2+
                ((d.damage_min_inc or 0)+(d.damage_max_inc or 0))/2*level(name),
                projectiles=a.loops or 1,radius=d.damage_radius or 0,cooldown=a.cooldown or 9,
                range=a.range,flying=false,channel='physical'})
        elseif name=='disintegrate' then
            add(name,'execute',{chance=1,cooldown=(p.cooldown_base or 22)+(p.cooldown_inc or -2)*level(name),
                flying=true,boss_immune=true})
        elseif name=='teleport' then
            local a=list[3] or {};local at=a.aura and E:get_template(a.aura);local au=at and at.aura or {}
            local mt=au.mod and E:get_template(au.mod) or {}
            add(name,'rewind',{cooldown=a.cooldown or 10,radius=au.radius or 0,
                max_targets=(p.max_count_base or 3)+(p.max_count_inc or 1)*level(name),
                rewind_nodes=(math.abs(mt.nodes_offset_min or 0)+math.abs(mt.nodes_offset_max or 0))/2+math.abs(mt.nodes_offset_inc or 0)*level(name),
                flying=true,boss_immune=true})
        elseif name=='elemental' and ENV and ENV.support_mechanisms then
            local soldier=E:get_template(e.barrack and e.barrack.soldier_type or 'soldier_elemental')
            local h=soldier and soldier.health or {};local regen=soldier and soldier.regen or {}
            local attack=soldier and soldier.melee and soldier.melee.attacks and soldier.melee.attacks[1] or {}
            local n=level(name)
            local r=add(name,'soldier',{hp_max=(h.hp_max or 0)+(h.hp_inc or 0)*n,
                armor=(h.armor or 0)+(h.armor_inc or 0)*n,
                damage_min=(attack.damage_min or 0)+(attack.damage_inc or 0)*n,
                damage_max=(attack.damage_max or 0)+(attack.damage_inc or 0)*n,
                cooldown=attack.cooldown or 0,radius=attack.damage_radius or 0,max_targets=attack.count or 1,
                respawn_seconds=h.dead_lifetime or 0,regen_amount=regen.health or 0,
                regen_interval=regen.cooldown or 0,regen_delay=regen.last_hit_standoff_time or 0,
                block_count=n>0 and 1 or 0,flying=false})
            r.known=soldier~=nil and h.hp_max~=nil and attack.damage_min~=nil and attack.cooldown~=nil
            -- Template capacity is inherited from barracks. Sorcerer's script
            -- summons one elemental only, and none before the first purchase.
        elseif name=='polymorph' then
            local mod=E:get_template('mod_polymorph_sorcerer');local pm=mod and mod.polymorph or {}
            add(name,'transform',{cooldown=(p.cooldown_base or 22)+(p.cooldown_inc or -2)*level(name),
                health_factor=pm.transfer_health_factor or .5,speed_factor=pm.transfer_speed_factor or 1.5,
                flying=true,boss_immune=true})
        elseif name=='bolt' then
            local b=base.bullet and E:get_template(base.bullet) or {}
            add(name,'chain',{max_targets=1+((b.bounces_lvl or {})[level(name)] or 0),
                base_targets=1+((b.bounces_lvl or {})[0] or 0),cooldown=base.cooldown or 2.2,
                radius=b.bounce_range or 80,flying=true})
        elseif name=='overcharge' then
            local at=E:get_template('aura_tesla_overcharge');local a=at and at.aura or {}
            add(name,'splash',{damage=((a.damage_min or 0)+(a.damage_max or 0))/2+(a.damage_inc or 0)*level(name),
                projectiles=1,radius=a.radius or 0,
                cooldown=base.cooldown or 2.2,flying=true,channel='true',triggered=true})
        elseif name=='missile' then
            local a=list[2] or {};local bt=a.bullet and E:get_template(a.bullet);local b=bt and bt.bullet or {}
            add(name,'splash',{damage=((b.damage_min or 0)+(b.damage_max or 0))/2+(p.damage_inc or 0)*level(name),
                projectiles=1,radius=b.damage_radius or 0,cooldown=a.cooldown_mixed or a.cooldown or 14.1,
                air_cooldown=a.cooldown_flying or 6.5,range=(a.range_base or e.attacks.range)*(1+(p.range_inc_factor or 0)*level(name)),
                flying=true,channel='physical'})
        elseif name=='cluster' then
            local a=list[3] or {};local bt=a.bullet and E:get_template(a.bullet);local b=bt and bt.bullet or {}
            local ft=b.fragment_name and E:get_template(b.fragment_name);local f=ft and ft.bullet or {}
            add(name,'splash',{damage=((f.damage_min or 0)+(f.damage_max or 0))/2,
                projectiles=(p.fragment_count_base or 1)+(p.fragment_count_inc or 2)*level(name),
                radius=f.damage_radius or 0,cooldown=a.cooldown or 17,range=a.range,flying=false,channel='physical',spread=true})
        else
            local r=add(name,'soldier',{cooldown=0});r.known=false
        end
    end
    if e.tower and e.tower.type=='sorcerer' then
        local mt=E:get_template('mod_sorcerer_curse_dps');local d=mt and mt.dps or {};local mm=mt and mt.modifier or {}
        local at=E:get_template('mod_sorcerer_curse_armor');local ab=at and at.armor_buff or {}
        local r=add('curse','armor_debuff',{damage_per_second=((d.damage_min or 0)+(d.damage_max or 0))/2/(d.damage_every or 1),
            duration=mm.duration or 0,armor_reduction=math.abs(ab.factor or 0),cooldown=base.cooldown or 1.5,
            flying=true,refreshes=true,tick_interval=d.damage_every or 1});r.level=1
    end
    return out
end
function S.combat(E,e,skill_name,skill_level)
    if not e then return {known=false,attacks={}} end
    local kind=e.tower and e.tower.type
    local detailed=ENV and ENV.skill_mechanisms
    local out={known=false,attacks={},range=e.attacks and e.attacks.range or 0,
        block=e.barrack and (e.barrack.max_soldiers or 3) or 0,skills={},template=e.template_name}
    local function level(name)
        if name==skill_name then return skill_level end
        return e.powers and e.powers[name] and e.powers[name].level or 0
    end
    for name,p in pairs(e.powers or {}) do
        out.skills[#out.skills+1]={name=name,level=level(name),known=kind=='tesla' and name=='bolt'}
    end
    for i,a in ipairs(e.attacks and e.attacks.list or {}) do
        local power=detailed and a.power_name or (attack_powers[kind] or {})[i]
        if detailed and kind=='musketeer' then power=a.power_name end
        local enabled=i==1 or (power and level(power)>0) or false
        local b=a.bullet and E:get_template(a.bullet)
        local d=b and b.bullet or {}
        local aura=a.aura and E:get_template(a.aura)
        aura=aura and aura.aura
        local lo=numeric(d.damage_min) or numeric(a.damage_min)
        local hi=numeric(d.damage_max) or numeric(a.damage_max)
        if detailed and kind=='arcane_wizard' and i==1 then
            local mod=E:get_template('mod_ray_arcane');local md=mod and mod.dps or {}
            lo=numeric(md.damage_min);hi=numeric(md.damage_max)
        end
        local radius=numeric(d.damage_radius) or 0
        local targets=1
        local damage_type=d.damage_type or a.damage_type or 0
        if detailed and kind=='arcane_wizard' and i==1 then
            local mt=E:get_template('mod_ray_arcane');damage_type=mt and mt.dps and mt.dps.damage_type or damage_type
        end
        if kind=='tesla' and i==1 and b then
            lo=numeric(b.bounce_damage_min);hi=numeric(b.bounce_damage_max)
            targets=1+((b.bounces_lvl or {})[level('bolt')] or 0)
            radius=b.bounce_range or radius
            -- Tesla's electrical hit is script-distributed; retain its identity.
            damage_type=DAMAGE_ELECTRICAL or damage_type
        elseif aura then
            radius=aura.radius or 0
            lo=numeric(aura.damage_min);hi=numeric(aura.damage_max)
            if kind=='tesla' and power=='overcharge' then
                local inc=(aura.damage_inc or 0)*level(power)
                lo=(lo or 0)+inc;hi=(hi or 0)+inc
            end
            damage_type=aura.damage_type or 0
        end
        local channel=has(damage_type,DAMAGE_TRUE) and 'true' or
            has(damage_type,DAMAGE_MAGICAL) and 'magic' or 'physical'
        local cooldown=numeric(a.cooldown) or numeric(e.attacks and e.attacks.min_cooldown)
        if aura and not cooldown then cooldown=numeric(aura.cycle_time) end
        local known=lo~=nil and hi~=nil and cooldown~=nil and cooldown>0
        local control=power=='teleport' or power=='polymorph' or power=='thorn'
        local r={name=a.bullet or a.aura,power=power,level=power and level(power),enabled=enabled,
            known=known,damage_min=lo,damage_max=hi,interval=cooldown,channel=channel,
            radius=radius,max_targets=targets,chain=kind=='tesla' and i==1,
            chain_factor=b and b.bounce_damage_factor,
            flying=not has(a.vis_bans or 0,F_FLYING),control=control,
            conditional=i>1,shoot_delay=a.shoot_time or 0}
        if detailed and kind=='musketeer' and power=='sniper' then
            -- Two mutually exclusive outcomes of one cast, not two extra guns.
            r.effect='execute';r.known=false
        end
        if power=='polymorph' then
            local mod=d.mod and E:get_template(d.mod)
            local pm=mod and mod.polymorph
            if pm then
                r.effect='polymorph';r.effect_known=true
                r.health_factor=pm.transfer_health_factor
                r.speed_factor=pm.transfer_speed_factor
                local p=e.powers and e.powers.polymorph
                if p then r.interval=(p.cooldown_base or cooldown)+(p.cooldown_inc or 0)*level(power) end
            end
        end
        if kind=='tesla' then r.nonstack_group='tesla' end
        out.attacks[#out.attacks+1]=r
        if enabled and known then out.known=true end
        if power then
            for _,s in ipairs(out.skills) do if s.name==power then s.known=known or control end end
        end
    end
    if detailed then out.mechanisms=mechanisms(E,e,level) end
    if ENV and ENV.support_mechanisms and kind=='sorcerer' then out.block=level('elemental')>0 and 1 or 0 end
    return out
end

-- Read-only grade/branch lookup from the currently patched native templates.
-- Four-level branches have independent rows; no grade-based DPS multiplier.
function S.catalog(E)
    local out={}
    for _,kind in ipairs({'archer','mage','engineer','barrack'}) do
        for grade=1,3 do
            local name='tower_'..kind..'_'..grade
            local t=E:get_template(name)
            if t then out[name]=S.combat(E,t) end
        end
    end
    for _,name in ipairs({'tower_ranger','tower_musketeer','tower_arcane_wizard','tower_sorcerer',
                         'tower_paladin','tower_barbarian','tower_bfg','tower_tesla'}) do
        local t=E:get_template(name)
        if t then out[name]=S.combat(E,t) end
    end
    return out
end

function S.development(E,menus,root,locked,price,all_branches)
    local out={}
    local function walk(name,cost,grade,depth)
        if depth>3 then return end
        local t=E:get_template(name)
        if not t or not t.tower then return end
        if depth>0 or t.tower.type=='tesla' or t.tower.type=='sorcerer' or (all_branches and grade>=4) then
            local plan={tpl=name,cost=cost,grade=grade}
            if t.tower.type=='tesla' or t.tower.type=='sorcerer' or (all_branches and grade>=4) then
                plan.combat=S.combat(E,t);plan.core=true
                if depth>0 then out[#out+1]=plan end
                local skills=all_branches and t.powers or {[t.tower.type=='tesla' and 'overcharge' or 'polymorph']=true}
                for skill in pairs(skills or {}) do
                    local p=t.powers and t.powers[skill]
                    if p and type(p.price_base)=='number' then
                        out[#out+1]={tpl=name,cost=cost+p.price_base,grade=grade,core=true,
                            skill=skill,combat=S.combat(E,t,skill,1)}
                    end
                end
            else out[#out+1]=plan end
        end
        local menu=(menus[t.tower.type] or {})[t.tower.level] or {}
        for _,item in pairs(menu) do
            if item.action=='tw_upgrade' and not locked(item.action_arg) then
                local p=price(item.action_arg)
                if type(p)=='number' then walk(item.action_arg,cost+p,math.min(4,grade+1),depth+1) end
            end
        end
    end
    local t=E:get_template(root)
    if t and t.build_name then root=t.build_name end
    local t0=E:get_template(root)
    local grade=t0 and t0.tower and t0.tower.level or 1
    if t0 and t0.tower and (attack_powers[t0.tower.type] or t0.tower.type=='paladin' or t0.tower.type=='barbarian') then grade=4 end
    walk(root,0,grade,0)
    return out
end

function S.rain(E,s)
    local c=E:get_template('power_fireball_control') or {}
    local b=E:get_template('power_fireball') or {}
    local d=b.bullet or {}
    return {known=type(d.damage_radius)=='number',radius=d.damage_radius or 60,
        spread=c.max_spread or 20,count=c.fireball_count or 3,
        damage_min=d.damage_min,damage_max=d.damage_max,
        burst_interval=0.33,max_speed=d.max_speed,delay_known=false,
        start_y=s and s.visible_coords and s.visible_coords.top or REF_H or 768,
        cataclysm_count=c.cataclysm_count or 0,scorch=b.scorch_earth and true or false}
end

function S.skills(e, menus, gold)
    local out = {}
    local menu = menus[e.tower.type]
    for _, item in pairs(menu and menu[e.tower.level] or {}) do
        local p = e.powers and e.powers[item.action_arg]
        if item.action == 'upgrade_power' and p then
            local price = p.level == 0 and p.price_base or p.price_inc
            out[#out+1] = {name=item.action_arg, level=p.level, max_level=p.max_level,
                price=price, eligible=not e.tower.blocked and not e.tower.sell and not e.tower.upgrade_to
                    and p.level < p.max_level and type(price)=='number',
                ok=not e.tower.blocked and not e.tower.sell and not e.tower.upgrade_to
                    and p.level < p.max_level and type(price)=='number' and gold>=price}
        end
    end
    table.sort(out,function(a,b) return a.name<b.name end)
    return out
end

function S.buy_skill(s, e, name, menus, pending, signal)
    if not e or not e.tower then return {ok=false,why='no_tower'} end
    for _, op in ipairs(S.skills(e,menus,s.player_gold-pending)) do
        if op.name==name then
            if not op.ok then return {ok=false,why='skill_unavailable'} end
            local p=e.powers[name]
            -- Same state transition and price calculation as the native menu.
            p.level=p.level+1
            p.changed=true
            s.player_gold=s.player_gold-op.price
            e.tower.spent=(e.tower.spent or 0)+op.price
            signal.emit('tower-power-upgraded',e,p)
            return {ok=true,cost=op.price,skill=name,level=p.level,hid=e.tower.holder_id}
        end
    end
    return {ok=false,why='unknown_skill'}
end

function S.mechanics(E,e)
    local target=e.moon and e.moon.transform_name
    local transformed=target and E:get_template(target)
    local regen=e.regen~=nil
    local aura_regen=false
    local aura_parameters
    for _, a in pairs(e.auras and e.auras.list or {}) do
        if a.name and a.name:find('regen') then
            regen=true;aura_regen=true
            local t=E:get_template(a.name)
            if t and t.regen and a.name:match('^aura_troll') then aura_parameters=t.regen end
        end
    end
    -- Fields are template observations, not proof of the script's exact
    -- activation rule. Unrecognized/aura regeneration remains unknown.
    local r=type(e.regen)=='table' and e.regen or {}
    if aura_parameters then
        r={health=r.health or aura_parameters.health,cooldown=r.cooldown or aura_parameters.cooldown}
    end
    local heal=numeric(r.health) or numeric(r.hp) or numeric(r.heal_amount)
    local interval=numeric(r.cooldown) or numeric(r.interval)
    local regen_known=not regen or ((not aura_regen or aura_parameters~=nil) and heal~=nil and interval~=nil and interval>0)
    return {transform=transformed~=nil, transform_name=target,
        transform_threshold=e.lycan_trigger_factor or 0,
        transform_magic=transformed and transformed.health and transformed.health.magic_armor or 0,
        transform_hp=transformed and transformed.health and transformed.health.hp_max or 0,
        regen=regen,regen_known=regen_known,regen_amount=heal,regen_interval=interval,
        regen_rate=regen_known and regen and heal/interval or nil,
        regen_rule_known=not regen,death_spawn=e.death_spawns~=nil,
        -- These two script mechanisms were observed creating native entities.
        -- Unknown timed actions are not assumed to be ordinary wave births.
        summon=e.spawner~=nil or e.spawner_waves~=nil or e.necromancer~=nil or e.template_name=='enemy_necromancer',
        replication=e.cloning~=nil or e.clone~=nil or e.template_name=='enemy_demon_legion',
        immune=e.health and e.health.ignore_damage and true or false,
        boss=e.template_name and e.template_name:sub(1,3)=='eb_' or false,
        polymorph_immune=has(e.vis and e.vis.bans,F_POLYMORPH),
        poison_immune=has(e.vis and e.vis.bans,F_POISON),
        instakill_immune=has(e.vis and e.vis.bans,F_INSTAKILL),
        root_immune=has(e.vis and e.vis.bans,F_STUN),
        teleport_immune=has(e.vis and e.vis.bans,F_TELEPORT),
        sheep=e.template_name and e.template_name:find('enemy_sheep',1,true)==1 or false}
end

-- Runtime mega_spawner fields and tuple layout verified against scripts.lua
-- lines 5785-6004 (bytecode): delay, jitter, group, subpath, quantity,
-- force_all, sequence, interval_min, interval_max, template, custom_data.
-- Random points are expected counts, not a prediction of the RNG outcome.
function S.special(E,s,index)
    local out={spawns={},sources=0,unknown=0}
    local function emit(owner,w,offset,depth)
        if depth>4 then out.unknown=out.unknown+1; return end
        local name=w[10]
        if name=='PACK' then
            local pack=owner.spawner_packs and owner.spawner_packs[w[11] and w[11].spawnPackId]
            if not pack then out.unknown=out.unknown+1; return end
            for _, child in pairs(pack) do emit(owner,child,offset+(w[1] or 0),depth+1) end
            return
        end
        local template=type(name)=='string' and E:get_template(name)
        if not template or not template.health then out.unknown=out.unknown+1; return end
        local ids=(owner.spawner_groups or {})[w[3]] or {w[3]}
        local count=0
        for _ in pairs(ids) do count=count+1 end
        for _, id in pairs(ids) do
            local point=(owner.spawner_points or {})[id]
            if point then
                local pos=point.to or point.from or {}
                local qty=(w[5] or 0)*(w[6] and 1 or 1/math.max(1,count))
                if qty>0 then
                    out.spawns[#out.spawns+1]={name=name,count=qty,pi=point.path,
                        x=pos.x,y=pos.y,entry_x=point.from and point.from.x,entry_y=point.from and point.from.y,
                        delay=offset+(w[1] or 0),delay_jitter=w[2] or 0,
                        interval_seconds=((w[8] or 0)+(w[9] or 0))/2,
                        interval=30*((w[8] or 0)+(w[9] or 0))/2,
                        special=true,expected=true,armor=template.health.armor or 0,
                        speed=template.motion and template.motion.max_speed,
                        magic_armor=template.health.magic_armor or 0,hp=S.health_value(template.health.hp_max,s.level_difficulty),
                        lives_cost=template.enemy and template.enemy.lives_cost,
                        flying=template.vis and F_FLYING and bit.band(template.vis.flags or 0,F_FLYING)~=0 or false,
                        mechanics=S.mechanics(E,template)}
                end
            else out.unknown=out.unknown+1 end
        end
    end
    for _,owner in pairs(s.entities) do
        if owner.spawner_waves and not owner.interrupt then
            out.sources=out.sources+1
            -- Manual sequences are independent of the campaign wave counter.
            local wave=owner.manual_wave or index
            for _,w in pairs(owner.spawner_waves[wave] or {}) do emit(owner,w,0,0) end
        end
    end
    return out
end

return S
