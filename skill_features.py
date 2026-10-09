"""Mechanism-specific candidate utilities; coarse, contextual, never rewards."""
import math
from combat_estimates import effective_attacks,attack_dps

SKILLS=('poison','thorn','sniper','shrapnel','disintegrate','teleport','polymorph','bolt','overcharge','curse',
        'missile','cluster','elemental','healing','shield','holystrike','twister','throwing','dual')
METRICS=('dot_damage','splash_damage','execute_hp','transform_hp','control_seconds','support_damage',
         'chain_damage','air_damage','armored_match','regen_match','eligible_count','partial')
HORIZONS=('current','next','second')
RAW_SCALES={'dot_dps':100,'dot_duration':10,'execute_chance':1,'execute_cooldown':30,
            'root_duration':10,'root_targets':10,'rewind_nodes':50,'splash_radius':250,
            'burst_damage':1000,'chain_targets':10,'transform_health':1,'transform_speed':3}
RAW_SCALES['armor_reduction']=1
DOT_SPREAD_EFFICIENCY=.7  # Native 40s: poison single=200 HP, four targets=615 HP.
FEATURE_NAMES=tuple(f'skill_{s}_{stage}' for s in SKILLS for stage in ('before','after'))
FEATURE_NAMES+=tuple(f'skill_raw_{m}_{stage}' for m in RAW_SCALES for stage in ('before','after'))
FEATURE_NAMES+=tuple(f'skill_{h}_{m}_{stage}' for h in HORIZONS for m in METRICS for stage in ('before','after','gain'))


def density(field,enemy,radius,future,flying):
    result=0.
    for z in field.future if future else field.enemies:
        if z.get('flying') and not flying:continue
        if (z.get('mechanics') or {}).get('immune'):continue
        if future:
            if z.get('pi')!=enemy.get('pi'):continue
            distance=abs(float(z.get('spawn_start_seconds') or 0)-float(enemy.get('spawn_start_seconds') or 0))*max(1,float(enemy.get('speed') or 30))
            count=float(z.get('count') or 0);interval=float(z.get('interval_seconds') or 0)
            if distance<=radius*2:
                result+=count if interval<=0 else min(count,1+2*radius/max(1,float(z.get('speed') or 30))/interval)
        elif math.hypot(float(z.get('x') or 0)-float(enemy.get('x') or 0),float(z.get('y') or 0)-float(enemy.get('y') or 0))<=radius:
            result+=1
    return max(1,min(12,result))


def utility(field,tower,defense,future):
    out=dict.fromkeys(METRICS,0.)
    if tower.get('blocked'):return out
    profile=tower.get('combat') or {};mechanisms=profile.get('mechanisms') or []
    enemies=field.future if future else field.enemies
    base=sum(attack_dps(a) for a in effective_attacks(profile) if not a.get('power') and not a.get('conditional'))
    for m in mechanisms:
        if m.get('level',0)<=0:continue
        out['partial']=max(out['partial'],float(m.get('partial',True) or not m.get('known')))
        if not m.get('known'):continue
        kind=m.get('kind');eligible=[]
        for e in enemies:
            mechanics=e.get('mechanics') or {}
            if not (tower.get('defense_paths') or {}).get(str(e.get('pi'))):continue
            if not future and m.get('range') and math.hypot(float(e.get('x') or 0)-float(tower.get('attack_x') or tower.get('x') or 0),
                    float(e.get('y') or 0)-float(tower.get('attack_y') or tower.get('y') or 0))>float(m['range']):continue
            if mechanics.get('immune') or (e.get('flying') and not m.get('flying')):continue
            if m.get('boss_immune') and mechanics.get('boss'):continue
            if kind=='dot' and mechanics.get('poison_immune'):continue
            if kind=='execute' and mechanics.get('instakill_immune'):continue
            if kind=='transform' and (mechanics.get('polymorph_immune') or mechanics.get('sheep')):continue
            if kind=='root' and mechanics.get('root_immune'):continue
            if kind=='rewind' and mechanics.get('teleport_immune'):continue
            count=float(e.get('count') or 0) if future else 1.
            if count>0:eligible.append((e,count))
        population=sum(count for _,count in eligible)
        if not population:continue
        if population<float(m.get('min_targets') or 1):continue
        out['eligible_count']+=population
        for e,count in eligible:
            hp=max(0,float(e.get('hp') or 0));exposure=max(0,float(field.exposure(tower,e,future)))
            if future and m.get('range'):exposure*=min(1,float(m['range'])/max(1,float(tower.get('attack_range') or 0)))
            if exposure<=0:continue
            share=count/population;cooldown=max(.1,float(m.get('cooldown') or 1))
            if e.get('flying') and m.get('air_cooldown'):cooldown=float(m['air_cooldown'])
            casts=min(count,exposure/cooldown*share)
            armor=max(0,min(1,float(e.get('armor') or 0)))
            targets=density(field,e,float(m.get('radius') or 0),future,bool(m.get('flying')))
            damage=0.;control=0.
            if kind in ('dot','armor_debuff'):
                lifespan=min(exposure,hp/max(1,base*(1-armor)))
                delay=float(m.get('tick_interval') or 1)
                damage=float(m.get('damage_per_second') or 0)*max(0,lifespan-delay)*share
                # Native ranger switches to unpoisoned enemies. Same-target
                # refresh never stacks, and saturation is bounded by delivery.
                saturation=min(population,max(1,float(m.get('duration') or 0)/cooldown))
                damage*=1+DOT_SPREAD_EFFICIENCY*(saturation-1)
                damage=min(hp*count,damage);out['dot_damage']+=damage
                out['armored_match']+=damage*armor
                if (e.get('mechanics') or {}).get('regen'):out['regen_match']+=damage
                if kind=='armor_debuff':
                    samples=set((tower.get('defense_samples') or {}).get(str(e.get('pi')),[]))
                    physical=0.
                    for t in defense:
                        if t.get('hid')==tower.get('hid') or t.get('blocked'):continue
                        other=set((t.get('defense_samples') or {}).get(str(e.get('pi')),[]))
                        physical+=len(samples&other)/max(1,len(samples))*sum(attack_dps(a) for a in effective_attacks(t.get('combat') or {}) if a.get('channel')=='physical')
                    out['support_damage']+=min(hp*count,physical*exposure*share*armor*float(m.get('armor_reduction') or 0))
            elif kind=='splash':
                hits=targets*float(m.get('projectiles') or 1)
                if m.get('spread'):hits*=.5  # Unknown fragment overlap is explicitly partial.
                damage=min(hp*count,float(m.get('damage') or 0)*casts*hits)
                if m.get('channel')=='physical':damage*=max(0,1-armor)
                out['splash_damage']+=damage
            elif kind=='execute':
                chance=max(0,min(1,float(m.get('chance') or 0)))
                relief=chance+(1-chance)*max(0,min(1,float(m.get('hp_fraction') or 0)))
                damage=min(hp*count,hp*casts*relief);out['execute_hp']+=damage
                out['armored_match']+=damage*armor
            elif kind=='transform':
                damage=hp*casts*max(0,1-float(m.get('health_factor',1)))/max(1,float(m.get('speed_factor') or 1))
                out['transform_hp']+=min(hp*count,damage)
            elif kind in ('root','rewind'):
                duration=float(m.get('duration') or 0)
                if kind=='rewind':
                    g=(field.o.get('route_geometry') or {}).get(str(e.get('pi')),{})
                    duration=float(m.get('rewind_nodes') or 0)*float(g.get('total_length') or 0)/max(1,float(g.get('node_count') or 200))/max(1,float(e.get('speed') or 30))
                control=min(count*exposure,duration*casts*min(targets,float(m.get('max_targets') or 1)))
                out['control_seconds']+=control
                damage=float(m.get('damage_per_second') or 0)*control*(1-armor)
                out['dot_damage']+=min(hp*count,damage)
                supporting=0.
                samples=set((tower.get('defense_samples') or {}).get(str(e.get('pi')),[]))
                for t in defense:
                    if t.get('hid')==tower.get('hid') or t.get('blocked'):continue
                    other=set((t.get('defense_samples') or {}).get(str(e.get('pi')),[]))
                    overlap=len(samples&other)/max(1,len(samples))
                    supporting+=overlap*sum(attack_dps(a) for a in effective_attacks(t.get('combat') or {}) if not e.get('flying') or a.get('flying'))
                out['support_damage']+=min(hp*count,control*supporting)
            elif kind=='chain':
                added=max(0,min(targets,float(m.get('max_targets') or 1))-min(targets,float(m.get('base_targets') or 1)))
                damage=min(hp*count,base*exposure*share*added*(1-armor));out['chain_damage']+=damage
            if e.get('flying'):out['air_damage']+=damage
    return out


def feature_values(field,candidate):
    source=candidate.get('spot') or {};before=source if source.get('state')=='tower' else dict(combat={})
    changing=candidate['type'] in ('build','upgrade','tower_skill','replace')
    after=field.placed(candidate) if changing else before
    values={}
    for skill in SKILLS:
        for stage,tower in (('before',before),('after',after)):
            levels={s['name']:s.get('level',0) for s in (tower.get('combat') or {}).get('skills',[])}
            levels.update({m['name']:m.get('level',0) for m in (tower.get('combat') or {}).get('mechanisms',[])})
            values[f'skill_{skill}_{stage}']=levels.get(skill,0)/3
    for stage,tower in (('before',before),('after',after)):
        raw=dict.fromkeys(RAW_SCALES,0.)
        for m in (tower.get('combat') or {}).get('mechanisms',[]):
            if m.get('level',0)<=0 or not m.get('known'):continue
            kind=m.get('kind')
            if kind in ('dot','armor_debuff'):
                raw['dot_dps']+=float(m.get('damage_per_second') or 0);raw['dot_duration']=max(raw['dot_duration'],float(m.get('duration') or 0))
                raw['armor_reduction']=max(raw['armor_reduction'],float(m.get('armor_reduction') or 0))
            elif kind=='execute':
                raw['execute_chance']=max(raw['execute_chance'],float(m.get('chance') or 0));raw['execute_cooldown']=max(raw['execute_cooldown'],float(m.get('cooldown') or 0))
            elif kind=='root':raw['root_duration']=float(m.get('duration') or 0);raw['root_targets']=float(m.get('max_targets') or 0)
            elif kind=='rewind':raw['rewind_nodes']=float(m.get('rewind_nodes') or 0)
            elif kind=='splash':
                raw['splash_radius']=max(raw['splash_radius'],float(m.get('radius') or 0));raw['burst_damage']+=float(m.get('damage') or 0)*float(m.get('projectiles') or 1)
            elif kind=='chain':raw['chain_targets']=float(m.get('max_targets') or 0)
            elif kind=='transform':raw['transform_health']=float(m.get('health_factor') or 0);raw['transform_speed']=float(m.get('speed_factor') or 0)
        for name,scale in RAW_SCALES.items():values[f'skill_raw_{name}_{stage}']=raw[name]/scale
    for horizon in HORIZONS:
        horizon_field=field.horizon_field(horizon);future=horizon!='current'
        if not hasattr(horizon_field,'_skill_before'):horizon_field._skill_before={}
        # The source tower and baseline support layout are fixed in this
        # observation. Global actions and every skill choice reuse that utility.
        before_key=id(source) if source.get('state')=='tower' else None
        if before_key not in horizon_field._skill_before:
            horizon_field._skill_before[before_key]=utility(horizon_field,before,field.towers,future)
        a=horizon_field._skill_before[before_key]
        b=utility(horizon_field,after,field.candidate_towers(candidate),future) if changing else a
        for metric in METRICS:
            scale=1 if metric=='partial' else 30 if metric in ('control_seconds','eligible_count') else 1000
            for stage,value in (('before',a[metric]),('after',b[metric]),('gain',b[metric]-a[metric])):
                values[f'skill_{horizon}_{metric}_{stage}']=value/scale
    return values
