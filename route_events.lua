-- Native insertion and goal events. Diagnostics only; never changes combat.
local M={}
function M.install(game,signal,previous)
    local tracker={events={},birth_events={},births={},seen={},store=game.store,simulation=game.simulation}
    if previous and previous.store==game.store then
        tracker.births=previous.births;tracker.seen=previous.seen
    end
    local E=require('entity_db');local SU=require('script_utils');local scripts=require('scripts')
    local regular,labels={},{}
    for _,sys in ipairs(game.simulation.systems_on_update) do
        for _,key in ipairs({'init','on_update'}) do
            local f=sys[key]
            if type(f)=='function' then for i=1,30 do
                local name,value=debug.getupvalue(f,i);if not name then break end
                if name=='spawner' and type(value)=='function' then regular[value]=true end
            end end
        end
    end
    for name,methods in pairs(scripts) do if type(methods)=='table' then
        for method,f in pairs(methods) do if type(f)=='function' then labels[f]=tostring(name)..'.'..tostring(method) end end
    end end
    local original_create=E.create_entity
    tracker.create=function(self,name,...)
        local e=original_create(self,name,...)
        if e and ENV and ENV.damage_tracker then
            local parent
            for level=2,5 do
                if not debug.getinfo(level,'f') then break end
                for i=1,12 do
                    local key,value=debug.getlocal(level,i)
                    if not key then break end
                    if (key=='this' or key=='owner' or key=='source' or key=='bullet') and type(value)=='table' and value.id then parent=value;break end
                end
                if parent then break end
            end
            ENV.damage_tracker:capture_creation(e,parent)
        end
        if e and e.enemy then
            local caller=debug.getinfo(2,'f');local f=caller and caller.func
            local label=labels[f] or ''
            local context=tracker.creation_context
            if not context then
                for i=1,8 do
                    local key,value=debug.getlocal(2,i)
                    if not key then break end
                    if (key=='this' or key=='owner') and type(value)=='table' and value.enemy and value.id then context=value;break end
                end
            end
            local death_context=tracker.creation_context
            local kind=death_context and 'death_spawn' or regular[f] and 'regular' or
                (label:find('legion') or label:find('clone') or label:find('duplicat')) and 'replication' or
                (f==SU.y_spawner_spawn or label:find('spawner') or label:find('necromancer')) and 'summon' or
                (label:find('polymorph') or label:find('lycan')) and 'transformation' or 'unclassified'
            e._route_creation={kind=kind,parent_id=context and context.id or nil,source=label}
        end
        return e
    end
    E.create_entity=tracker.create
    local original_death_spawns=SU.do_death_spawns
    tracker.death_spawns=function(s,parent,...)
        local before=tracker.creation_context;tracker.creation_context=parent
        local ok,result=pcall(original_death_spawns,s,parent,...)
        tracker.creation_context=before
        if not ok then error(result,0) end
        return result
    end
    SU.do_death_spawns=tracker.death_spawns
    local function record_birth(e,s,verified)
        if not e.enemy then return end
        local n=e.nav_path or {}
        local row={id=e.id,wave=s.wave_group_number or 0,path=n.pi,
            born_tick=s.tick,verified=verified,name=e.template_name,
            kind=e._route_creation and e._route_creation.kind or 'unclassified',
            parent_id=e._route_creation and e._route_creation.parent_id,
            cost=e.enemy.lives_cost or 1}
        if not verified and e._route_birth then row=e._route_birth end
        tracker.births[e.id]=row
        e._route_birth=row
        e._route_spawn_wave=row.wave
        if verified then tracker.birth_events[#tracker.birth_events+1]=row end
    end
    tracker.observer={on_insert=function(self,e,s)
        if s==tracker.store then record_birth(e,s,true) end
        return true
    end}
    game.simulation.systems_on_insert[#game.simulation.systems_on_insert+1]=tracker.observer
    -- Existing entities predate attachment and are explicitly uncertain.
    for _,e in pairs(tracker.store.entities) do record_birth(e,tracker.store,false) end
    tracker.goal=function(e)
        if not e or not e.enemy or tracker.seen[e.id] then return end
        tracker.seen[e.id]=true
        local birth=tracker.births[e.id]
        local n=e.nav_path or {}
        tracker.events[#tracker.events+1]={id=e.id,name=e.template_name,
            path=(birth and birth.path) or n.pi or 0,spawn_path=birth and birth.path or 0,exit_path=n.pi or 0,
            wave=birth and birth.wave or 0,birth_verified=birth and birth.verified or false,
            cost=e.enemy.lives_cost or 1,tick=tracker.store.tick,
            lives_at_event=tracker.store.lives}
        tracker.events[#tracker.events].birth_kind=birth and birth.kind or 'unclassified'
    end
    signal.register('enemy-reached-goal',tracker.goal)
    function tracker:drain(lives_before,lives_after)
        local out={leaks_by_path={},leak_counts_by_path={},leak_cohorts={},
            leak_life_cohorts={},leaks_path_partial=false,goal_events=self.events,birth_events=self.birth_events}
        self.events={};self.birth_events={}
        local available=math.max(0,lives_before-lives_after)
        for _,e in ipairs(out.goal_events) do
            local path=tostring(e.path)
            local cost=math.max(0,e.cost)
            local actual=math.min(cost,available);available=available-actual
            e.actual_life=actual
            out.leaks_by_path[path]=(out.leaks_by_path[path] or 0)+actual
            out.leak_counts_by_path[path]=(out.leak_counts_by_path[path] or 0)+1
            local wave=tostring(e.wave)
            out.leak_cohorts[wave]=out.leak_cohorts[wave] or {}
            out.leak_life_cohorts[wave]=out.leak_life_cohorts[wave] or {}
            out.leak_cohorts[wave][path]=(out.leak_cohorts[wave][path] or 0)+1
            out.leak_life_cohorts[wave][path]=(out.leak_life_cohorts[wave][path] or 0)+actual
            out.leaks_path_partial=out.leaks_path_partial or e.path==0
        end
        if available>0 then
            out.leaks_by_path.unknown=available
            out.leaks_path_partial=true
        end
        return out
    end
    function tracker:status()
        local alive,pending,born={},{},{}
        for _,e in pairs(self.births) do
            local wave=tostring(e.wave);local path=tostring(e.path or 0)
            born[wave]=born[wave] or {};local row=born[wave][path]
            if not row then row={count=0,life_budget=0,kinds={},names={},unverified=0};born[wave][path]=row end
            row.count=row.count+1;row.life_budget=row.life_budget+(e.cost or 1)
            local kind=e.kind or 'unclassified';row.kinds[kind]=(row.kinds[kind] or 0)+1
            row.names[e.name]=(row.names[e.name] or 0)+1
            if not e.verified then row.unverified=row.unverified+1 end
        end
        local function add(e,target)
            if e.enemy and e.health and not e.health.dead and (e.health.hp or 0)>0 then
                local birth=self.births[e.id]
                local key=tostring(birth and birth.wave or -1)
                if birth and not birth.verified and birth.wave>0 then key='-1' end
                target[key]=(target[key] or 0)+1
            end
        end
        for _,e in pairs(self.store.entities) do add(e,alive) end
        for _,e in pairs(self.store.pending_inserts or {}) do add(e,pending) end
        local unverified=0
        local active_spawners=0
        for _ in pairs(self.store.waves_active or {}) do active_spawners=active_spawners+1 end
        for _,event in ipairs(self.events) do if not event.birth_verified then unverified=unverified+1 end end
        return {alive=alive,pending=pending,born=born,wave=self.store.wave_group_number,
            waves_finished=self.store.waves_finished or false,
            unverified_pending_goals=unverified,active_spawners=active_spawners}
    end
    function tracker:uninstall()
        if E.create_entity==self.create then E.create_entity=original_create end
        if SU.do_death_spawns==self.death_spawns then SU.do_death_spawns=original_death_spawns end
        signal.remove('enemy-reached-goal',self.goal)
        local list=self.simulation.systems_on_insert
        for i=#list,1,-1 do if list[i]==self.observer then table.remove(list,i) end end
    end
    return tracker
end
return M
