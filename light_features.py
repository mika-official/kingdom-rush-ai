"""v8 facts and local geometry only; no policy search or combat prediction."""
import math
import numpy as np
from candidate_features import Battlefield,TYPES,BRANCHES,SCENE_NAMES as BASE_SCENE_NAMES,as_list
from kr_macro import KINDS
from skill_features import SKILLS
from combat_estimates import attack_dps

PARAMETERS={
 'poison':('damage_per_second','duration','tick_interval','cooldown','flying'),
 'thorn':('damage_per_second','duration','max_targets','min_targets','radius','cooldown','boss_immune'),
 'sniper':('chance','hp_fraction','cooldown','range','flying','boss_immune'),
 'shrapnel':('damage','projectiles','radius','cooldown','range'),
 'disintegrate':('chance','cooldown','flying','boss_immune'),
 'teleport':('rewind_nodes','max_targets','radius','cooldown','flying','boss_immune'),
 'polymorph':('health_factor','speed_factor','cooldown','flying','boss_immune'),
 'bolt':('max_targets','base_targets','radius','cooldown','flying'),
 'overcharge':('damage','projectiles','radius','cooldown','flying'),
 'curse':('damage_per_second','duration','tick_interval','armor_reduction','cooldown','flying'),
 'missile':('damage','projectiles','radius','cooldown','air_cooldown','range','flying'),
 'cluster':('damage','projectiles','radius','cooldown','range','spread')}
SCALES=dict(damage=500,damage_per_second=100,duration=10,tick_interval=5,cooldown=30,
    air_cooldown=30,max_targets=10,min_targets=10,base_targets=10,projectiles=10,radius=250,range=500,rewind_nodes=100,speed_factor=3)
BASIC=('physical_dps','magic_dps','true_dps','interval','range','area_radius','chain_targets','block','partial')
DETAIL=('position_known','x','y','cost','refund','net_cost','reserve','loss','source_grade','target_grade',
    'range_change','covered_routes','current_overlap_towers','overlap_physical_dps','overlap_magic_dps',
    'overlap_block','cooldown','cooldown_known','spawn_pending','next_available','second_available','effect_radius','effect_spread')
ENEMY=('count','hp_total','hp_mean','armor','magic_armor','flying','speed','regen_rate','boss','immune',
       'poison_immune','instakill_immune','root_immune','teleport_immune','polymorph_immune',
       'summon','replication','death_spawn','sheep','interval','hp_known','regen_known','speed_known',
       'lives_cost','progress','nearest_exit','exit_known')
HORIZONS=('current','next','second')
SCENE_NAMES=BASE_SCENE_NAMES+('hero_present','hero_hp_fraction','hero_dead','hero_x','hero_y','hero_rally_x','hero_rally_y')
SCENE_DIM=len(SCENE_NAMES)
FEATURE_NAMES=('valid',)+tuple('type_'+v for v in TYPES)+tuple('kind_'+v for v in KINDS)+tuple('branch_'+v for v in BRANCHES)+DETAIL
FEATURE_NAMES+=tuple(f'{stage}_{name}' for stage in ('source','target','delta') for name in BASIC)
FEATURE_NAMES+=tuple(f'skill_{skill}_{stage}_{name}' for skill in SKILLS for stage in ('source','target')
    for name in ('level','known','partial')+PARAMETERS.get(skill,()))
FEATURE_NAMES+=tuple(f'{horizon}_{scope}_{name}' for horizon in HORIZONS for scope in ('global','local') for name in ENEMY)
FEATURE_DIM=len(FEATURE_NAMES);F={name:i for i,name in enumerate(FEATURE_NAMES)}


def mapping(value):return value if isinstance(value,dict) else {}


class LightField(Battlefield):
    def __init__(self,env,observation):
        self.env,self.o=env,observation
        self.enemies=as_list(observation.get('enemies'))
        self.horizons={'current':self.enemies,'next':as_list(mapping(observation.get('forecast')).get('spawns')),
            'second':as_list(mapping(observation.get('forecast_second')).get('spawns'))}
        self.future=self.horizons['next']+[dict(e,count=.5*float(e.get('count') or 0),expected=True) for e in self.horizons['second']]
        self.towers=[s for s in as_list(observation.get('spots')) if s.get('state')=='tower']
        self._profiles={};self._summaries={}
        self._tower_facts=[(t,self.basic(t.get('combat'),t.get('attack_range'))) for t in self.towers]

    def scene(self):
        base=super().scene();hero=mapping(self.o.get('hero'))
        extra=[bool(hero),float(hero.get('hp') or 0)/max(1,float(hero.get('hp_max') or 0)),bool(hero.get('dead')),
            float(hero.get('x') or 0)/1024,float(hero.get('y') or 0)/768,
            float(hero.get('rally_x') or 0)/1024,float(hero.get('rally_y') or 0)/768]
        return np.concatenate((base,np.asarray(extra,np.float32)))

    def basic(self,profile,radius=None):
        profile=mapping(profile);key=(id(profile),radius)
        if key in self._profiles:return self._profiles[key][1]
        result=dict.fromkeys(BASIC,0.);intervals=[]
        for attack in as_list(profile.get('attacks')):
            if not attack.get('enabled') or attack.get('power') or attack.get('conditional'):continue
            if not attack.get('known'):result['partial']=1;continue
            channel=attack.get('channel','physical')
            if channel in ('physical','magic','true'):result[channel+'_dps']+=attack_dps(attack)/100
            if attack.get('interval'):intervals.append(float(attack['interval']))
            result['area_radius']=max(result['area_radius'],float(attack.get('radius') or 0)/250)
            result['chain_targets']=max(result['chain_targets'],float(attack.get('max_targets') or 1)/10)
        result['interval']=min(intervals,default=0)/5
        result['range']=float(radius if radius is not None else profile.get('range') or 0)/500
        result['block']=float(profile.get('block') or 0)/3
        result['partial']=max(result['partial'],float(not profile.get('known')))
        self._profiles[key]=(profile,result)
        return result

    def summary(self,horizon,paths=None,center=None,radius=0):
        key=(horizon,None if paths is None else tuple(sorted(paths)),center,radius)
        if key in self._summaries:return self._summaries[key]
        result=dict.fromkeys(ENEMY,0.);population=0.;nearest=None
        for enemy in self.horizons[horizon]:
            if paths is not None and str(enemy.get('pi')) not in paths:continue
            if center is not None and math.hypot(float(enemy.get('x') or 0)-center[0],float(enemy.get('y') or 0)-center[1])>radius:continue
            count=1. if horizon=='current' else max(0,float(enemy.get('count') or 0))
            population+=count;hp=max(0,float(enemy.get('hp') or 0));m=mapping(enemy.get('mechanics'))
            result['hp_total']+=count*hp
            for name in ('armor','magic_armor','speed'):result[name]+=count*float(enemy.get(name) or 0)
            result['flying']+=count*bool(enemy.get('flying'))
            result['regen_rate']+=count*float(m.get('regen_rate') or 0)
            result['interval']+=count*float(enemy.get('interval_seconds') or 0)
            result['hp_known']+=count*(enemy.get('hp') is not None)
            result['speed_known']+=count*(enemy.get('speed') is not None)
            result['regen_known']+=count*(bool(m.get('regen_known')) or not m.get('regen'))
            result['lives_cost']+=count*float(enemy.get('lives_cost') or 0)
            result['progress']+=count*float(enemy.get('progress') or 0)
            if enemy.get('to_goal') is not None:
                result['exit_known']+=count
                nearest=float(enemy['to_goal']) if nearest is None else min(nearest,float(enemy['to_goal']))
            for name in ENEMY[8:19]:result[name]+=count*bool(m.get(name))
        result['count']=population/30;result['hp_mean']=result['hp_total']/max(1,population)/1000
        result['hp_total']/=10000
        for name in ENEMY[3:]:result[name]/=max(1,population)
        result['speed']/=100;result['regen_rate']/=100;result['interval']/=30
        result['lives_cost']/=10;result['nearest_exit']=0 if nearest is None else nearest/200
        self._summaries[key]=result
        return result

    def vector(self,candidate):
        x=np.zeros(FEATURE_DIM,np.float32)
        def put(name,value):x[F[name]]=float(value or 0)
        put('valid',1);put('type_'+candidate['type'],1)
        spot=candidate.get('spot') or {};kind=candidate.get('kind') or (self.env.tower_kind(spot) if spot else None)
        if kind in KINDS:put('kind_'+kind,1)
        if candidate.get('branch') in BRANCHES:put('branch_'+candidate['branch'],1)
        changing=candidate['type'] in ('build','upgrade','replace','tower_skill')
        source=mapping(spot.get('combat')) if spot.get('state')=='tower' else {}
        target=(mapping(candidate.get('combat')) or source) if changing else source
        source_range=float(spot.get('attack_range') or source.get('range') or 0)
        target_range=float(candidate.get('target_range') or target.get('range') or source_range)
        old=self.basic(source,source_range);new=self.basic(target,target_range)
        for name in BASIC:
            put('source_'+name,old[name]);put('target_'+name,new[name]);put('delta_'+name,new[name]-old[name])
        paths=mapping(candidate.get('target_paths') or spot.get('defense_paths'))
        position=candidate.get('target') or spot or {}
        center=(float(position['x']),float(position.get('y') or 0)) if position.get('x') is not None else None
        put('position_known',center is not None)
        if center:put('x',center[0]/1024);put('y',center[1]/768)
        cost=float(candidate.get('cost') or 0);refund=float(candidate.get('refund') or 0)
        for name,value in dict(cost=cost,refund=refund,net_cost=cost-refund,
            reserve=float(self.o.get('gold') or 0)-cost+refund,loss=candidate.get('loss') or 0).items():put(name,value/1000)
        put('source_grade',self.env.effective_level(spot)/4 if spot.get('state')=='tower' else 0)
        put('target_grade',float(candidate.get('grade') or (self.env.effective_level(spot) if spot else 0))/4)
        put('range_change',(target_range-source_range)/500);put('covered_routes',len(paths)/16)
        put('cooldown',float(candidate.get('cooldown') or 0)/120);put('cooldown_known','cooldown' in candidate)
        put('spawn_pending',bool(self.o.get('wave_spawn_pending') or self.o.get('active_spawners')))
        put('next_available',mapping(self.o.get('forecast')).get('available'))
        put('second_available',mapping(self.o.get('forecast_second')).get('available'))
        for stage,profile in (('source',source),('target',target)):
            powers={s['name']:s for s in as_list(profile.get('skills'))}
            mechanisms={m['name']:m for m in as_list(profile.get('mechanisms'))}
            for skill in SKILLS:
                mechanism=mechanisms.get(skill,{});power=powers.get(skill,{})
                level=float(mechanism.get('level',power.get('level',0)) or 0)
                prefix=f'skill_{skill}_{stage}_';put(prefix+'level',level/3)
                if level<=0:continue
                put(prefix+'known',mechanism.get('known',False));put(prefix+'partial',mechanism.get('partial',True))
                for name in PARAMETERS.get(skill,()):put(prefix+name,float(mechanism.get(name) or 0)/SCALES.get(name,1))
        for tower,facts in self._tower_facts:
            if tower.get('blocked'):continue
            nearby=bool(center and math.hypot(center[0]-float(tower.get('x') or 0),
                center[1]-float(tower.get('y') or 0))<=target_range+float(tower.get('attack_range') or 0))
            shared=nearby and (bool(set(paths)&set(mapping(tower.get('defense_paths')))) if paths else True)
            if shared:
                put('current_overlap_towers',x[F['current_overlap_towers']]+1/20)
                for name in ('physical_dps','magic_dps','block'):put('overlap_'+name,x[F['overlap_'+name]]+facts[name])
        for horizon in HORIZONS:
            radius=target_range or 90
            if candidate['type']=='rain':
                p=mapping(self.o.get('rain_profile'));radius=float(p.get('radius') or 60)
                put('effect_radius',radius/250);put('effect_spread',float(p.get('spread') or 0)/100)
            elif candidate['type'] in ('reinf','ultimate','hero_move','rally'):radius=90
            global_values=self.summary(horizon)
            local=self.summary(horizon,center=center,radius=radius) if horizon=='current' and center else self.summary(horizon,set(paths)) if paths else global_values
            for scope,values in (('global',global_values),('local',local)):
                for name,value in values.items():put(f'{horizon}_{scope}_{name}',value)
        return x
