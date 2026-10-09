"""Bounded future-wave kill estimate with shared cooldowns and target removal.

Geometry is sampled; targeting, projectiles, blocking and scripted mechanics
remain approximations. No fitting to calibration outcomes is performed here.
"""
import heapq
import math
from combat_estimates import effective_attacks, attack_dps


def coverage_windows(geometry,samples,speed):
    covered={round(float(p),6) for p in samples};distance=0;windows=[]
    for s in geometry.get('segments',[]):
        length=float(s['length']);left=round(float(s['start']),6) in covered;right=round(float(s['end']),6) in covered
        if left or right:
            start=(distance+(0 if left else length/2))/speed
            end=(distance+(length if right else length/2))/speed
            if windows and abs(windows[-1][1]-start)<1e-8:windows[-1]=(windows[-1][0],end)
            else:windows.append((start,end))
        distance+=length
    return windows


def prepare_actors(enemies,geometries):
    actors=[]
    for batch,e in enumerate(enemies):
        geometry=geometries.get(str(e.get('pi')),{});speed=float(e.get('speed') or 0)
        count=float(e.get('count') or 0)
        if not e.get('timing_known') or speed<=0 or not geometry.get('segments') or count!=int(count):return None
        if e.get('hp') is None or float(e['hp'])<=0:return None
        length=sum(float(s['length']) for s in geometry['segments'])
        for j in range(int(count)):
            spawn=float(e.get('spawn_start_seconds') or 0)+j*float(e.get('interval_seconds') or 0)
            actors.append(dict(batch=batch,enemy=e,speed=speed,spawn=spawn,exit=spawn+length/speed,
                               hp=float(e.get('hp') or 0),initial_hp=float(e.get('hp') or 0),last=spawn))
        if len(actors)>128:return None
    return actors


class SurvivalContext:
    """Read-only spawn schedules and targeting windows shared by candidates."""
    def __init__(self,enemies,geometries):
        self.enemies=enemies;self.geometries=geometries
        self.actors=prepare_actors(enemies,geometries);self.windows={}

    def attack_windows(self,tower,attack,coverage_cache):
        samples_by_route=tower.get('defense_samples') or {}
        key=(bool(attack.get('flying')),tuple((pi,tuple(samples)) for pi,samples in sorted(samples_by_route.items())))
        if key in self.windows:return self.windows[key]
        windows={}
        for i,a in enumerate(self.actors):
            e=a['enemy']
            if (e.get('mechanics') or {}).get('immune') or (e.get('flying') and not attack.get('flying')):continue
            samples=samples_by_route.get(str(e.get('pi')),[])
            cache_key=(id(self.geometries[str(e['pi'])]),str(e['pi']),a['speed'],tuple(samples))
            if cache_key not in coverage_cache:
                coverage_cache[cache_key]=coverage_windows(self.geometries[str(e['pi'])],samples,a['speed'])
            for left,right in coverage_cache[cache_key]:
                windows.setdefault(i,[]).append((a['spawn']+left,a['spawn']+right))
        self.windows[key]=windows
        return windows


def estimate_survival(enemies,towers,geometries,coverage_cache=None,context=None):
    coverage_cache={} if coverage_cache is None else coverage_cache
    context=SurvivalContext(enemies,geometries) if context is None else context
    if context.enemies is not enemies or context.geometries is not geometries:
        raise ValueError('SurvivalContext must belong to the same observation and horizon')
    if context.actors is None:return None
    actors=[dict(a) for a in context.actors]
    if not actors:return {}
    schedule=[];weapons=[]
    for tower in towers:
        if tower.get('blocked'):continue
        for attack in effective_attacks(tower.get('combat') or {}):
            if not attack.get('enabled') or attack_dps(attack)<=0 or attack.get('effect'):continue
            interval=float(attack.get('interval') or 0)
            if interval<.1:return None
            windows=context.attack_windows(tower,attack,coverage_cache)
            if windows:
                index=len(weapons);weapons.append((attack,interval,windows))
                heapq.heappush(schedule,(min(left for ws in windows.values() for left,_ in ws),index))
    events=0
    while schedule:
        now,index=heapq.heappop(schedule);attack,interval,windows=weapons[index];events+=1
        if events>8192:return None
        available=[i for i,ws in windows.items() if actors[i]['hp']>0 and now<actors[i]['exit'] and any(left<=now<right for left,right in ws)]
        if not available:
            upcoming=[max(left,now+.05) for i,ws in windows.items() if actors[i]['hp']>0 for left,right in ws if right>now+.05]
            if upcoming:heapq.heappush(schedule,(min(upcoming),index))
            continue
        # Closest to the exit: spend attacks on kills instead of equally
        # spreading partial HP relief over every reachable batch.
        target=min(available,key=lambda i:actors[i]['exit']-now)
        primary=actors[target];victims=[target]
        radius=float(attack.get('radius') or 0)
        if radius>0:
            pos=(now-primary['spawn'])*primary['speed']
            victims += [i for i,a in enumerate(actors) if i!=target and a['hp']>0 and a['spawn']<=now<a['exit']
                        and a['enemy']['pi']==primary['enemy']['pi'] and (not a['enemy'].get('flying') or attack.get('flying'))
                        and abs((now-a['spawn'])*a['speed']-pos)<=radius
                        and not (a['enemy'].get('mechanics') or {}).get('immune')]
            if attack.get('chain'):victims=victims[:max(1,int(attack.get('max_targets') or 1))]
        damage=attack_dps(attack)*interval
        for i in victims:
            a=actors[i];e=a['enemy'];mechanics=e.get('mechanics') or {}
            # Conservative upper recovery bound; actual trigger rules may differ.
            a['hp']=min(a['initial_hp'],a['hp']+max(0,float(mechanics.get('regen_rate') or 0))*(now-a['last']))
            a['last']=now
            armor=float(e.get('magic_armor') or 0) if attack.get('channel')=='magic' else float(e.get('armor') or 0)
            if attack.get('channel')=='true':armor=0
            factor=float(attack.get('chain_factor',1)) if attack.get('chain') and i!=target else 1
            a['hp']-=damage*max(0,1-armor)*factor
        heapq.heappush(schedule,(now+interval,index))
    result={}
    for a in actors:
        r=result.setdefault(a['enemy']['pi'],dict(leaks=0.,life=0.,regular_leaks=0.))
        if a['hp']>0:
            r['leaks']+=1
            if not a['enemy'].get('special') and not a['enemy'].get('expected'):r['regular_leaks']+=1
            r['life']+=max(0,float(a['enemy'].get('lives_cost') if a['enemy'].get('lives_cost') is not None else 1))
    return result
