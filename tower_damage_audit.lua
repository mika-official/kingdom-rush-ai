-- Optional diagnostics at the native damage boundary. No damage/RNG changes.
local M={}
local bit=require('bit')
local function pack(...) return {n=select('#',...),...} end
function M.install(game,Strategy)
    local E=require('entity_db');local U=require('utils')
    local audit={rows={},roots={},entities={},seen={},killed={},other={},unknown_sources={},store=game.store,events={},in_damage=false,transforms={},control_marks={}}
    local function tower_row(e)
        if not e or not e.tower or e.tower.type=='holder' or (e.template_name or ''):find('^tower_build_') then return nil end
        local skills={};for name,p in pairs(e.powers or {}) do skills[#skills+1]=name..':'..tostring(p.level or 0) end;table.sort(skills)
        local key=tostring(e.id)..':'..tostring(e.template_name)..':'..table.concat(skills,',')
        if not audit.rows[key] then
            audit.rows[key]={key=key,entity_id=e.id,holder_id=e.tower.holder_id,template=e.template_name,
                kind=e.tower.type,level=e.tower.level,skills=table.concat(skills,','),combat=Strategy.combat(E,e),
                seconds=0,actual_damage=0,native_damage=0,raw_damage=0,hits=0,kills=0,killed_hp_max=0,
                by_effect={},effect_creations={},execute_hp_removed=0,execute_by_effect={},effect_targets={},
                control_seconds=0,rewind_nodes=0,transform_count=0,transform_hp_relief=0}
        end
        return key
    end
    local function resolve(id,depth)
        if not id or (depth or 0)>8 then return nil end
        if audit.roots[id] then return audit.roots[id] end
        local e=audit.entities[id] or audit.store.entities[id]
        if not e then return nil end
        if e.enemy then return nil end
        local key=tower_row(e)
        if not key then
            local parents={e.bullet and e.bullet.source_id,e.modifier and e.modifier.source_id,
                e.aura and e.aura.source_id,e.soldier and e.soldier.tower_id,e.soldier and e.soldier.owner_id,
                e.owner_id,e.source_id,e.spawner_id}
            -- pairs tolerates holes in the optional parent list.
            for _,parent in pairs(parents) do if parent~=id then key=resolve(parent,(depth or 0)+1);if key then break end end end
        end
        if key then audit.roots[id]=key end
        return key
    end
    audit.observer={on_insert=function(self,e,s)
        if s==audit.store then
            audit.entities[e.id]=e;local root=resolve(e.id)
            local transform=audit.transforms[e.id]
            if transform and transform.root and audit.rows[transform.root] and e.health then
                local row=audit.rows[transform.root];row.transform_count=row.transform_count+1
                row.transform_hp_relief=row.transform_hp_relief+math.max(0,transform.hp-(e.health.hp or 0))
                audit.transforms[e.id]=nil
            end
            if root and (e.template_name=='mod_thorn' or e.template_name=='mod_teleport_arcane') then
                local target_id=e.modifier and e.modifier.target_id or e.target_id
                local target=s.entities[target_id] or audit.entities[target_id]
                audit.control_marks[e.id]={root=root,target_id=target_id,template=e.template_name,
                    before_node=target and target.nav_path and target.nav_path.ni,recorded=0}
            end
        end
        return true
    end}
    function audit:capture_creation(e,parent)
        self.entities[e.id]=e
        if e.enemy and parent and parent.template_name=='mod_polymorph_sorcerer' then
            local target_id=parent.modifier and parent.modifier.target_id or parent.target_id
            local target=self.store.entities[target_id] or self.entities[target_id]
            if target and target.health then self.transforms[e.id]={root=resolve(parent.id),hp=target.health.hp or 0} end
        end
        if parent and not e.enemy then
            self.entities[parent.id]=parent
            local key=parent.tower and tower_row(parent) or resolve(parent.id)
            if key then
                self.roots[e.id]=key
                local row=self.rows[key];local name=e.template_name or 'unknown'
                row.effect_creations[name]=(row.effect_creations[name] or 0)+1
            end
        end
    end
    game.simulation.systems_on_insert[#game.simulation.systems_on_insert+1]=audit.observer
    for _,e in pairs(audit.store.entities) do audit.entities[e.id]=e end
    local original_predict=U.predict_damage
    audit.predict=function(target,d,...)
        if audit.in_damage and target and target.enemy and target.health and d and not audit.seen[d.id] then
            audit.pending[#audit.pending+1]={d=d,target=target,hp=math.max(0,target.health.hp or 0),root=resolve(d.source_id)}
        end
        return original_predict(target,d,...)
    end
    U.predict_damage=audit.predict
    local system,original_update
    for _,sys in ipairs(game.simulation.systems_on_update) do
        if type(sys.on_update)=='function' then
            local raw=string.dump(sys.on_update)
            if raw:find('predict_damage',1,true) and raw:find('damage_result',1,true) and raw:find('damage_queue',1,true) then
                assert(not system,'Ambiguous native damage system');system=sys;original_update=sys.on_update
            end
        end
    end
    assert(system,'Native damage system not found')
    audit.update=function(...)
        local s=audit.store
        local queued={}
        for _,d in ipairs(s.damage_queue or {}) do
            local target=s.entities[d.target_id]
            if target and target.enemy and target.health and (target.health.hp or 0)>0 then
                queued[#queued+1]={d=d,target=target,hp=target.health.hp,root=resolve(d.source_id)}
            end
        end
        if audit.last_tick~=s.tick then
            audit.last_tick=s.tick
            local controlled={}
            for _,e in pairs(s.entities) do if e.tower and e.tower.type~='holder' then
                -- Refresh tower binding after upgrade; existing projectiles keep their launch binding.
                local key=tower_row(e);audit.roots[e.id]=key
                if key then audit.rows[key].seconds=audit.rows[key].seconds+(TICK_LENGTH or 1/60) end
            end end
            for id,mark in pairs(audit.control_marks) do
                local target=s.entities[mark.target_id];local mod=s.entities[id]
                if mod then
                    if mark.template=='mod_thorn' then controlled[mark.root..':'..tostring(mark.target_id or id)]=mark.root end
                    if target and target.nav_path and mark.before_node then
                        local rewind=math.max(0,mark.before_node-target.nav_path.ni)
                        if mark.template=='mod_teleport_arcane' and rewind>mark.recorded then
                            audit.rows[mark.root].rewind_nodes=audit.rows[mark.root].rewind_nodes+rewind-mark.recorded;mark.recorded=rewind
                        end
                    end
                else audit.control_marks[id]=nil end
            end
            for _,root in pairs(controlled) do audit.rows[root].control_seconds=audit.rows[root].control_seconds+(TICK_LENGTH or 1/60) end
        end
        audit.pending={};audit.in_damage=true
        local result=pack(pcall(original_update,...));audit.in_damage=false
        if not result[1] then error(result[2],0) end
        local credited={}
        for _,event in ipairs(audit.pending) do
            local d,target=event.d,event.target
            if not audit.seen[d.id] and type(d.damage_applied)=='number' then
                audit.seen[d.id]=true
                local applied=math.max(0,d.damage_applied);local actual=math.min(event.hp,applied)
                credited[target.id]=(credited[target.id] or 0)+actual
                local row=event.root and audit.rows[event.root]
                if not row then
                    local source=audit.entities[d.source_id] or s.entities[d.source_id]
                    local kind=source and (source.hero and 'hero' or source.enemy and 'enemy' or source.template_name) or 'unknown'
                    audit.other[kind]=(audit.other[kind] or 0)+actual
                    if source and not audit.unknown_sources[kind] then
                        local fields={};for k,v in pairs(source) do if type(v)~='function' and k~='render' and k~='main_script' then fields[k]=v end end
                        audit.unknown_sources[kind]=fields
                    end
                else
                    row.actual_damage=row.actual_damage+actual;row.native_damage=row.native_damage+applied
                    row.raw_damage=row.raw_damage+math.max(0,d.value or 0);row.hits=row.hits+1
                    local source=audit.entities[d.source_id] or s.entities[d.source_id]
                    local name=source and source.template_name or 'unknown'
                    row.by_effect[name]=(row.by_effect[name] or 0)+actual
                    row.effect_targets[name]=row.effect_targets[name] or {}
                    row.effect_targets[name][tostring(target.id)]=true
                    if event.hp>0 and applied>=event.hp and not audit.killed[target.id] then
                        audit.killed[target.id]=true;row.kills=row.kills+1;row.killed_hp_max=row.killed_hp_max+(target.health.hp_max or 0)
                    end
                end
            end
        end
        audit.seen_execute=audit.seen_execute or {}
        for _,event in ipairs(queued) do
            local d,target=event.d,event.target
            local killed=DR_KILL and bit.band(d.damage_result or 0,DR_KILL)~=0
            local execute=DAMAGE_INSTAKILL and bit.band(d.damage_type or 0,DAMAGE_INSTAKILL)~=0
            if execute and killed and not audit.seen_execute[d.id] then
                audit.seen_execute[d.id]=true
                local row=event.root and audit.rows[event.root]
                if row then
                    local removed=math.max(0,event.hp-(credited[target.id] or 0))
                    local source=audit.entities[d.source_id] or s.entities[d.source_id]
                    local name=source and source.template_name or 'unknown'
                    row.execute_hp_removed=row.execute_hp_removed+removed
                    row.execute_by_effect[name]=(row.execute_by_effect[name] or 0)+removed
                    if not audit.killed[target.id] then
                        audit.killed[target.id]=true;row.kills=row.kills+1;row.killed_hp_max=row.killed_hp_max+(target.health.hp_max or 0)
                    end
                end
            end
        end
        return unpack(result,2,result.n)
    end
    system.on_update=audit.update
    function audit:status()
        local rows={};for _,row in pairs(self.rows) do rows[#rows+1]=row end
        table.sort(rows,function(a,b) return a.key<b.key end)
        return {rows=rows,other_damage=self.other,unknown_sources=self.unknown_sources,method='native_damage_applied_clamped_to_pre_hit_hp'}
    end
    function audit:uninstall()
        if U.predict_damage==self.predict then U.predict_damage=original_predict end
        if system.on_update==self.update then system.on_update=original_update end
        local list=game.simulation.systems_on_insert
        for i=#list,1,-1 do if list[i]==self.observer then table.remove(list,i) end end
    end
    return audit
end
return M
