"""Versioned candidate features. Estimates describe coverage, never simulated DPS."""
import math
import numpy as np
from kr_macro import KINDS
from kr_strategy import SKILLS, log, mechanism_profile
from kr_tactical import enemy_profile, inside, support_at
from combat_estimates import capacity, reference_catalog

TYPES=('wait','wave','build','upgrade','tower_skill','replace','rain','reinf','ultimate','hero_move','rally','meta')
BRANCHES=('tower_ranger','tower_musketeer','tower_arcane_wizard','tower_sorcerer',
          'tower_paladin','tower_barbarian','tower_bfg','tower_tesla')
SCENE_NAMES=('gold','life','wave','wave_total','time','next_time','next_time_known',
             'tower_count','empty_count','blocked_count','forecast_known','special_unknown',
             'rain_ready','rain_cd','reinf_ready','reinf_cd','hero_ready','hero_cd')
SCENE_NAMES += tuple('current_'+str(i) for i in range(10)) + tuple('future_'+str(i) for i in range(10))
SCENE_NAMES += tuple('mechanism_'+str(i) for i in range(6))
SCENE_NAMES += tuple('tower_'+k for k in KINDS)+tuple('grade_'+k for k in KINDS)
SCENE_NAMES += ('second_wave_known','current_spawn_pending','current_remaining_unknown')
SCENE_DIM=len(SCENE_NAMES)
FEATURE_NAMES=('valid',)+tuple('type_'+x for x in TYPES)+tuple('kind_'+x for x in KINDS)
FEATURE_NAMES += tuple('branch_'+x for x in BRANCHES)+tuple('skill_'+x for x in SKILLS)
DETAIL_NAMES=('position_valid','x','y','cost_valid','cost','refund','net_cost','reserve_after_action',
    'reserve_fraction','loss','source_grade','target_grade','current_skill_level',
    'coverage','rear','covered_routes','future_count','future_hp','special_count','arrival_time',
    'arrival_known','future_armor','future_magic','future_flying','future_density','future_regen',
    'future_transform','current_count','current_hp','current_density','current_armor','current_magic',
    'current_flying','current_regen','current_transform','nearest_exit','exit_known',
    'existing_physical','existing_magic','existing_area','existing_air','existing_block',
    'support','missing_physical','missing_magic','missing_area','missing_air',
    'delta_physical','delta_magic','delta_area','delta_air','delta_block','delta_control',
    'exposure','covered_unprotected_hp','future_pressure','reserve_risk','reserve_risk_known',
    'cooldown_cost','cooldown_valid','replacement','forecast_uncertain','future_regen_density',
    'blocking_output_synergy','repeat_output','forecast_hp_known','future_min_interval',
    'benefit_physical','benefit_magic','benefit_area','benefit_air','power_urgency','cooldown_future_overlap',
    'target_range_known','range_change','route_coverage_change')
FEATURE_NAMES += DETAIL_NAMES
FEATURE_DIM=len(FEATURE_NAMES)
F={name:i for i,name in enumerate(FEATURE_NAMES)}


def as_list(x): return x if isinstance(x,list) else []


class Battlefield:
    def __init__(self,env,o):
        self.env,self.o=env,o
        self.enemies=as_list(o.get('enemies'))
        self.future=as_list((o.get('forecast') or {}).get('spawns'))
        # Second-wave counts are discounted planning evidence, with unknown ETA.
        self.future=self.future+[dict(e,count=.5*float(e.get('count',0)),delay=None,expected=True,horizon=2)
                                 for e in as_list((o.get('forecast_second') or {}).get('spawns'))]
        self.towers=[s for s in as_list(o.get('spots')) if s.get('state')=='tower']
        self.cm=env._cmax(o)
        self.combat_catalog=o.get('combat_catalog') or reference_catalog()

    def scene(self):
        o=self.o;powers=as_list(o.get('powers'))
        f=[log(o.get('gold'),3500),float(o.get('lives',0))/max(1,self.env.lives0),
           float(o.get('wave',0))/max(1,o.get('wave_total',1)),float(o.get('wave_total',0))/20,
           self.env.t/max(1,self.env.max_ticks),float(o.get('next_wave_left') or 0)/120,
           float(o.get('next_wave_left') is not None),len(self.towers)/18,
           sum(s.get('state')=='empty' for s in as_list(o.get('spots')))/18,
           sum(bool(s.get('blocked')) for s in self.towers)/18,
           float((o.get('forecast') or {}).get('available',False)),
           log(((o.get('forecast') or {}).get('special') or {}).get('unknown',0))]
        for i in range(3):
            p=powers[i] if i<len(powers) else {}
            f.extend([float(p.get('ready',False)),float(p.get('cooldown_left') or 0)/120])
        f.extend(enemy_profile(self.enemies));f.extend(enemy_profile(self.future,True))
        f.extend(mechanism_profile(self.enemies))
        f.extend(sum(self.env.tower_kind(s)==k for s in self.towers)/18 for k in KINDS)
        f.extend(sum(self.env.effective_level(s) for s in self.towers if self.env.tower_kind(s)==k)/72 for k in KINDS)
        pending=bool(o.get('wave_spawn_pending') or o.get('active_spawners'))
        f.extend([float((o.get('forecast_second') or {}).get('available',False)),float(pending),float(pending)])
        return np.asarray(f,dtype=np.float32)

    def strength(self,kind,grade=1,skills=0,profile=None,template=None):
        if profile is not None:
            return capacity(profile)
        if not grade or kind not in KINDS:
            return np.zeros(5)
        name=template or ('tower_'+kind+'_'+str(min(3,int(grade))))
        profile=self.combat_catalog.get(name)
        if profile is not None:
            return capacity(profile)
        # Missing data is not evidence of a linear grade multiplier. This
        # uncalibrated unit fallback is only for old observations without data.
        return np.array({'archer':[1,0,0,1,0],'mage':[0,1,0,1,0],
                         'engineer':[1,0,1,0,0],'barrack':[0,0,0,0,1]}[kind],dtype=float)

    def coverage(self,sp,kind):
        cov=(sp.get('cov') or {}).get(kind,{})
        paths=(sp.get('defense_paths') or {}) if sp.get('state')=='tower' else cov.get('paths',{})
        # Empty serialized maps are Lua arrays; treat them as empty maps.
        if not isinstance(paths,dict): paths={}
        radius=float(sp.get('attack_range') or cov.get('r') or 0)
        center={'x':sp.get('attack_x',sp.get('x',0)),'y':sp.get('attack_y',sp.get('y',0))}
        return cov,paths,radius,center

    def power_radius(self,c): return 60 if c['type']=='rain' else 100

    def vector(self,c):
        x=np.zeros(FEATURE_DIM,dtype=np.float32)
        def put(name,value): x[F[name]]=float(value or 0)
        put('valid',1);put('type_'+c['type'],1)
        kind=c.get('kind');sp=c.get('spot') or {};target=c.get('target')
        if kind in KINDS: put('kind_'+kind,1)
        if c.get('branch') in BRANCHES: put('branch_'+c['branch'],1)
        if c.get('skill') in SKILLS: put('skill_'+c['skill'],1)
        pos=sp or target or {}
        if 'x' in pos:
            put('position_valid',1);put('x',float(pos['x'])/1024);put('y',float(pos.get('y',0))/768)
        cost=float(c.get('cost') or 0);refund=float(c.get('refund') or 0);gold=float(self.o.get('gold') or 0)
        reserve=gold+refund-cost
        put('cost_valid',c.get('cost') is not None);put('cost',cost/1000);put('refund',refund/1000)
        put('net_cost',(cost-refund)/1000);put('reserve_after_action',reserve/3500)
        put('reserve_fraction',reserve/max(1,gold));put('loss',float(c.get('loss') or 0)/1000)
        put('replacement',c['type']=='replace')
        old_kind=self.env.tower_kind(sp) if sp else None
        old_grade=self.env.effective_level(sp) if sp.get('state')=='tower' else 0
        grade=c.get('grade',old_grade)
        put('source_grade',old_grade/4);put('target_grade',grade/4)
        put('current_skill_level',float(c.get('skill_level') or 0)/3)
        cov,paths,radius,center=self.coverage(sp,kind or old_kind) if sp else ({},{},90,pos)
        old_radius=radius;old_paths=paths
        if c.get('target_range') is not None:
            put('target_range_known',1);radius=float(c['target_range'])
            paths=c.get('target_paths') or {}
            if not isinstance(paths,dict): paths={}
            put('range_change',(radius-old_radius)/200)
            put('route_coverage_change',(len(paths)-len(old_paths))/9)
        if target: radius=self.power_radius(c); center=target
        local=[e for e in self.enemies if pos and inside(e.get('x',0),e.get('y',0),center,radius)] if pos else self.enemies
        future=[e for e in self.future if str(e.get('pi')) in paths] if sp else self.future
        # Each entity and forecast batch contributes once, even at intersections.
        p=enemy_profile(local);q=enemy_profile(future,True)
        m=mechanism_profile(local);n=mechanism_profile(future,True)
        put('coverage',cov.get('c',0));put('rear',cov.get('p',0));put('covered_routes',len(paths)/9)
        put('future_count',q[0]);put('future_hp',log(sum(float(e.get('count',0))*float(e.get('hp') or 0) for e in future),100000))
        put('special_count',log(sum(float(e.get('count',0)) for e in future if e.get('special'))))
        put('future_armor',q[2]);put('future_magic',q[3]);put('future_flying',q[1]);put('future_density',q[9])
        put('future_regen',n[2]);put('future_transform',n[0]);put('current_count',p[0])
        put('current_hp',log(sum(float(e.get('hp') or 0) for e in local),10000))
        for name,i in [('density',9),('armor',2),('magic',3),('flying',1)]: put('current_'+name,p[i])
        put('current_regen',m[2]);put('current_transform',m[0])
        goals=[float(e['to_goal']) for e in local if e.get('to_goal') is not None]
        put('exit_known',bool(goals));put('nearest_exit',min(goals,default=200)/200)
        delayed=[float(e['delay']) for e in future if e.get('delay') is not None]
        put('arrival_known',bool(delayed));put('arrival_time',(float(self.o.get('next_wave_left') or 0)+min(delayed,default=0))/120)
        put('future_min_interval',min((float(e['interval_seconds']) for e in future if e.get('interval_seconds') is not None),default=0)/30)
        put('forecast_uncertain',any(e.get('expected') for e in future) or not (self.o.get('forecast') or {}).get('available'))
        put('forecast_hp_known',bool(future) and all(e.get('hp') is not None for e in future))
        existing=np.zeros(5)
        neighboring=[]
        for tower in self.towers:
            if tower.get('blocked'): continue
            tk=self.env.tower_kind(tower);_,tp,tr,tc=self.coverage(tower,tk)
            shared=set(paths)&set(tp)
            # Weight route overlap by target coverage. All overlapping bins stay one tower.
            overlap=len(shared)/max(1,len(paths)) if paths else float(bool(pos) and inside(pos.get('x',0),pos.get('y',0),tc,tr))
            if overlap:
                neighboring.append(tower)
                existing += self.strength(tk,self.env.effective_level(tower),profile=tower.get('combat'),template=tower.get('name'))*overlap
        for i,name in enumerate(('physical','magic','area','air','block')): put('existing_'+name,existing[i]/20)
        support=support_at(pos,self.towers) if pos else 0
        put('support',support/10)
        # Threat-weighted shortage proxies; not a calibrated damage prediction.
        threat=max(float(q[0]),float(p[0]))
        needs=np.array([threat*(1-q[2]),threat*q[2]*(1-q[3]),threat*q[9]*(1-q[1]),threat*q[1],0])
        shortages=np.maximum(0,needs-existing/10)
        for i,name in enumerate(('physical','magic','area','air')): put('missing_'+name,shortages[i])
        delta=np.zeros(5)
        old=self.strength(old_kind,old_grade,profile=sp.get('combat'),template=sp.get('name'))
        if c['type'] in ('build','upgrade','replace','tower_skill'):
            new=self.strength(kind or old_kind,grade,profile=c.get('combat'),template=c.get('branch'))
            delta=new if c['type']=='build' else new-old
            if c['type']=='tower_skill' and c.get('combat') is None:
                delta=.2*old  # Legacy v5 lacks per-skill native profiles.
        for i,name in enumerate(('physical','magic','area','air','block')): put('delta_'+name,delta[i]/4)
        for i,name in enumerate(('physical','magic','area','air')): put('benefit_'+name,delta[i]*shortages[i]/4)
        control=c.get('skill') in ('thorn','teleport','polymorph','twister')
        put('delta_control',control)
        put('blocking_output_synergy',delta[4]*support/10)
        put('repeat_output',max(0,float(np.dot(np.maximum(delta,0),existing)))/80)
        uncovered=0
        for e in local:
            if not any(self.env.tower_kind(t) in ('archer','mage','engineer') and
                inside(e.get('x',0),e.get('y',0),self.coverage(t,self.env.tower_kind(t))[3],
                       self.coverage(t,self.env.tower_kind(t))[2]) for t in neighboring): uncovered+=float(e.get('hp') or 0)
        put('covered_unprotected_hp',log(uncovered,10000))
        put('exposure',radius/200);put('future_pressure',threat)
        # Preserve cash as a feature; no invented required-spend estimate.
        put('reserve_risk',threat*max(0,1-reserve/500));put('reserve_risk_known',bool(future))
        put('future_regen_density',n[2]*q[9])
        if c['type'] in ('rain','reinf','ultimate'):
            put('cooldown_valid',1);put('cooldown_cost',c.get('cooldown',60)/120)
            put('power_urgency',float(x[F['current_hp']])*(.25+float(p[9]))*
                (1+float(x[F['covered_unprotected_hp']]))*(1.5-min(goals,default=200)/200))
            next_time=self.o.get('next_wave_left')
            put('cooldown_future_overlap',float(next_time is not None and next_time<c.get('cooldown',60))*float(q[0]))
        return x
