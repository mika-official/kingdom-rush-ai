"""Native-template planning estimates, with explicit unknowns and no reward shaping."""
import math
import numpy as np
from candidate_features import Battlefield, FEATURE_NAMES as V5_NAMES, SCENE_NAMES, as_list
from combat_estimates import rates, effective_attacks, attack_dps

EXTRA_NAMES=(
    'combat_known','combat_partial','source_physical_dps','source_magic_dps','source_true_dps',
    'target_physical_dps','target_magic_dps','target_true_dps','target_area_radius','target_chain_targets',
    'source_control','target_control','lost_control','lost_block','lost_skill_levels','lost_advanced',
    'current_damage_gain','future_damage_gain','current_gap_before','current_gap_after',
    'future_gap_before','future_gap_after','keep_value','best_upgrade_gain','best_upgrade_cost',
    'rebuild_cost','rebuild_shortfall','rebuild_known','since_investment','recent_investment',
    'holder_replacements','rain_hit_count','rain_effective_hp','rain_avoidable_hp',
    'rain_emergency','rain_reserve_value','rain_forecast_confidence','rain_radius',
    'rain_spread','rain_delay_estimate','rain_delay_known','rain_motion_uncertainty',
    'save_available','save_shortfall','save_cost','save_benefit','save_current_gain',
    'save_future_gain','save_target_grade','save_skill','save_archer','save_mage','save_engineer',
    'saving_emergency','save_income_active','output_tier3_count','output_tier4_count','output_skill_levels')
FEATURE_NAMES=V5_NAMES+EXTRA_NAMES
FEATURE_DIM=len(FEATURE_NAMES)
F={name:i for i,name in enumerate(FEATURE_NAMES)}
SCENE_DIM=len(SCENE_NAMES)


def mapping(value): return value if isinstance(value,dict) else {}


def rain_hits(o,target):
    """Expected local hits at an approximate impact time; never exact combat credit."""
    p=mapping(o.get('rain_profile'));radius=float(p.get('radius') or 60)
    spread=float(p.get('spread') or 20)
    # Burst spacing is verified in scripts.lua. Flight acceleration is unknown;
    # use max speed only for an estimate and expose that uncertainty.
    delay=max(0,(float(p.get('start_y') or 768)-float(target.get('y') or 0))/max(1,float(p.get('max_speed') or 450)))
    delay+=float(p.get('burst_interval') or .33)*max(0,float(p.get('count') or 3)-1)/2
    damage=((float(p.get('damage_min') or 0)+float(p.get('damage_max') or 0))/2)*float(p.get('count') or 3)
    hits=[];uncertain=0
    for e in as_list(o.get('enemies')):
        x=float(e.get('x') or 0);y=float(e.get('y') or 0);ahead=mapping(e.get('ahead'))
        if ahead:
            dx=float(ahead.get('x',x))-x;dy=float(ahead.get('y',y))-y;length=math.hypot(dx,dy)
            distance=float(e.get('speed') or 0)*delay
            if length: x+=dx/length*distance;y+=dy/length*distance
        else: uncertain+=1
        distance=math.hypot(x-float(target.get('x') or 0),y-float(target.get('y') or 0))
        # Approximate random spread; hard legality uses the same outer envelope.
        envelope=radius+math.sqrt(2)*spread
        if distance>envelope: continue
        weight=min(1,max(0,(envelope-distance)/max(1,envelope-radius)))
        if mapping(e.get('mechanics')).get('immune'): weight=0
        hits.append((e,weight,min(float(e.get('hp') or 0),damage)*weight))
    return hits,delay,uncertain/max(1,len(as_list(o.get('enemies'))))


class PlanningField(Battlefield):
    def __init__(self,env,o):
        super().__init__(env,o)
        self._contributions={};self._effects={};self._base={};self._comparisons={}

    def power_radius(self,c):
        if c['type']=='rain':
            p=mapping(self.o.get('rain_profile'))
            return float(p.get('radius') or 60)+math.sqrt(2)*float(p.get('spread') or 20)
        return super().power_radius(c)

    def placed(self,c):
        sp=c.get('spot') or {}
        return dict(sp,combat=c.get('combat') or sp.get('combat'),
                    attack_range=c.get('target_range',sp.get('attack_range',0)),
                    defense_paths=c.get('target_paths',sp.get('defense_paths',{})),
                    defense_samples=c.get('target_samples',sp.get('defense_samples',{})),
                    defense_lengths=c.get('target_lengths',sp.get('defense_lengths',{})))

    def exposure(self,tower,enemy,future):
        profile=mapping(tower.get('combat'))
        radius=float(tower.get('attack_range') or profile.get('range') or 0)
        lengths=mapping(tower.get('defense_lengths'))
        value=min(20,float(lengths.get(str(enemy.get('pi'))) or radius)/max(1,float(enemy.get('speed') or 30)))
        return value if future else min(6,value)

    def attack_population(self,enemies,eligible,attack,future):
        return sum(float(enemies[i].get('count') or 0) if future else 1 for i in eligible)


    def contribution(self,tower,future,effects=False):
        profile=mapping(tower.get('combat'));enemies=self.future if future else self.enemies
        if not profile.get('attacks'): return np.zeros(len(enemies))
        cache=(id(profile),tower.get('hid'),id(tower.get('defense_paths')),id(tower.get('defense_lengths')),id(tower.get('defense_samples')),future)
        if cache in self._contributions: return self._effects[cache] if effects else self._contributions[cache]
        paths=mapping(tower.get('defense_paths'));lengths=mapping(tower.get('defense_lengths'))
        vector=np.zeros(len(enemies));polymorph=np.zeros(len(enemies));eligible=[]
        radius=float(tower.get('attack_range') or profile.get('range') or 0)
        for i,e in enumerate(enemies):
            bins=paths.get(str(e.get('pi')))
            if not bins: continue
            if not future and isinstance(bins,list):
                index=min(2,int(float(e.get('progress') or 0)*3))
                if not any(bins[index:]): continue
            eligible.append(i)
        attacks=effective_attacks(profile)
        for i in eligible:
            e=enemies[i];count=float(e.get('count') or 0) if future else 1
            if tower.get('blocked') or mapping(e.get('mechanics')).get('immune'): continue
            speed=float(e.get('speed') or 0)
            exposure=self.exposure(tower,e,future)
            for a in attacks:
                if not a.get('enabled') or (e.get('flying') and not a.get('flying')): continue
                population=self.attack_population(enemies,eligible,a,future)
                if a.get('effect')=='polymorph' and a.get('effect_known'):
                    m=mapping(e.get('mechanics'))
                    if m.get('boss') or m.get('polymorph_immune') or m.get('sheep'): continue
                    # Current HP is transferred, not an instant kill. Faster
                    # sheep still leak: discount HP relief by their speed gain.
                    casts=exposure/max(.01,float(a.get('interval') or 20))*count/max(1,population)
                    fraction=max(0,1-float(a.get('health_factor',.5)))
                    relief=float(e.get('hp') or 0)*fraction*min(count,casts)/max(1,float(a.get('speed_factor') or 1))
                    vector[i]+=relief;polymorph[i]+=relief
                    continue
                resistance=float(e.get('magic_armor') or 0) if a.get('channel')=='magic' else float(e.get('armor') or 0)
                if a.get('channel')=='true': resistance=0
                dps=attack_dps(a)*max(0,1-resistance)
                targets=1
                if a.get('radius',0)>0:
                    interval=float(e.get('interval_seconds') or 0)
                    if future:
                        density=count if interval<=0 else min(count,1+2*float(a['radius'])/max(1,speed or 30)/interval)
                    else:
                        density=sum(math.hypot(float(z.get('x') or 0)-float(e.get('x') or 0),float(z.get('y') or 0)-float(e.get('y') or 0))<=float(a['radius'])
                                    and (a.get('flying') or not z.get('flying')) for z in enemies)
                    targets=max(1,density)
                    if a.get('chain'):
                        targets=min(targets,float(a.get('max_targets') or 1))
                        targets=1+(targets-1)*float(a.get('chain_factor',1))
                vector[i]+=dps*exposure*count/max(1,population)*targets
        self._contributions[cache]=vector
        self._effects[cache]=polymorph
        return polymorph if effects else vector

    def aggregate(self,towers,future):
        """Conservative Tesla marginal coverage, not a combat simulation.

        At shared path samples use the stronger Tesla density, preserving both
        sequential, disjoint sections. Missing geometry stays explicitly partial.
        """
        enemies=self.future if future else self.enemies
        total=np.zeros(len(enemies));poly=np.zeros(len(enemies));cells=[{} for _ in enemies]
        for t in towers:
            v=self.contribution(t,future)
            effect=self.contribution(t,future,effects=True);poly+=effect;v=v-effect
            tesla=any(a.get('nonstack_group')=='tesla' for a in as_list(mapping(t.get('combat')).get('attacks')))
            for i,e in enumerate(enemies):
                samples=as_list(mapping(t.get('defense_samples')).get(str(e.get('pi')))) if tesla else []
                if not future: samples=[p for p in samples if p>=float(e.get('progress') or 0)]
                if not samples: total[i]+=v[i];continue
                density=v[i]/len(samples)
                for p in samples: cells[i][p]=max(cells[i].get(p,0),density)
        # The same monster can only become a sheep once. Subsequent casts on
        # that original target cannot erase the remaining sheep HP.
        cap=np.asarray([float(e.get('hp') or 0)*(float(e.get('count') or 0) if future else 1)/3 for e in enemies])
        return total+np.asarray([sum(c.values()) for c in cells])+np.minimum(poly,cap)

    def comparison(self,c,future):
        source=c.get('spot') or {}
        key=(future,c['type'],source.get('hid'),id(c.get('combat')),
             c.get('target_range'),id(c.get('target_paths')),id(c.get('target_lengths')),id(c.get('target_samples')))
        if key in self._comparisons: return self._comparisons[key]
        enemies=self.future if future else self.enemies
        hp=np.asarray([max(0,float(e.get('hp') or 0))*(float(e.get('count') or 0) if future else 1) for e in enemies])
        if future not in self._base:
            self._base[future]=self.aggregate(self.towers,future)
        before=self._base[future];towers=self.towers
        if c['type'] in ('replace','upgrade','tower_skill') and source.get('state')=='tower':
            towers=[t for t in towers if t.get('hid')!=source.get('hid')]
        if c['type'] in ('build','replace','upgrade','tower_skill'):
            towers=towers+[self.placed(c)]
        after=before if c['type'] not in ('build','replace','upgrade','tower_skill') else self.aggregate(towers,future)
        # Count each enemy/batch once after summing all overlapping defenses.
        gap_before=np.maximum(0,hp-before);gap_after=np.maximum(0,hp-after)
        result=float((gap_before-gap_after).sum()),float(gap_before.sum()),float(gap_after.sum()),gap_before
        self._comparisons[key]=result
        return result

    def development(self,sp,op):
        return dict(type='upgrade',spot=sp,kind=self.env.tower_kind(sp),combat=op.get('combat'),
                    target_range=mapping(op.get('combat')).get('range',sp.get('attack_range',0)),
                    target_paths=op.get('target_paths',sp.get('defense_paths',{})),
                    target_samples=op.get('target_samples',sp.get('defense_samples',{})),
                    target_lengths=op.get('target_lengths',sp.get('defense_lengths',{})),cost=op.get('price',op.get('cost')),
                    grade=op.get('grade',min(4,self.env.effective_level(sp)+1)))

    def value(self,c):
        a=self.comparison(c,False);b=self.comparison(c,True)
        return (2*a[0]+b[0])/1000,a[0],b[0]

    def core_plan(self,c):
        """Discounted outcome of an affordable path to a strong final tower.

        Cost includes this action, later upgrades and the first defining skill.
        This is a planning hint, never added to immediate damage or PPO reward.
        """
        if c['type'] not in ('build','upgrade','tower_skill'): return None
        sp=c.get('spot') or {};best=None
        plans=as_list(c.get('development',sp.get('development')))
        for op in plans:
            if not op.get('core') or not op.get('combat'): continue
            goal=self.development(sp,op)
            if sp.get('state')!='tower': goal['type']='build'
            benefit,current,future=self.value(goal)
            cost=float(op.get('cost') or 0)+float(c.get('cost') or 0)
            # Discount distant goals so expanding an undefended route remains possible.
            score=benefit/(1+cost/500)
            if benefit>0 and (best is None or score>best['score']):
                best=dict(score=score,cost=cost,benefit=benefit,current=current,future=future,
                    grade=op['grade'],skill=bool(op.get('skill')),kind=c.get('kind') or self.env.tower_kind(sp),hid=sp.get('hid'))
        return best

    def saving_plan(self):
        best=None;gold=float(self.o.get('gold') or 0)
        for sp in self.towers:
            if sp.get('blocked'): continue
            plan=self.core_plan(dict(type='upgrade',spot=sp,cost=0))
            if plan and plan['cost']>gold:
                plan=dict(plan,shortfall=plan['cost']-gold)
                if best is None or plan['score']>best['score']: best=plan
            ops=[(op,False) for op in as_list(sp.get('options')) if not op.get('locked') and op.get('price') is not None]
            ops += [(op,True) for op in as_list(sp.get('skills')) if op.get('eligible')]
            for op,skill in ops:
                cost=float(op.get('price') or 0)
                if cost<=gold: continue
                c=self.development(sp,op)
                if skill: c['type']='tower_skill'
                benefit,current,future=self.value(c)
                score=benefit/(1+cost/500)
                if benefit>0 and (best is None or score>best['score']):
                    best=dict(score=score,cost=cost,shortfall=cost-gold,benefit=benefit,current=current,
                              future=future,grade=c['grade'],skill=skill,kind=c['kind'],hid=sp['hid'])
        return best

    def vector(self,c):
        x=np.zeros(FEATURE_DIM,np.float32);x[:len(V5_NAMES)]=super().vector(c)
        def put(name,value): x[F[name]]=float(value or 0)
        sp=c.get('spot') or {};source=mapping(sp.get('combat'));target=mapping(c.get('combat')) or source
        old,old_area,old_chain,old_control,old_partial=rates(source)
        new,area,chain,control,partial=rates(target)
        put('combat_known',target.get('known'));put('combat_partial',partial or old_partial)
        for i,name in enumerate(('physical','magic','true')):
            put('source_'+name+'_dps',old[i]/100);put('target_'+name+'_dps',new[i]/100)
        put('target_area_radius',area/200);put('target_chain_targets',chain/10)
        put('source_control',old_control/3);put('target_control',control/3)
        put('lost_control',max(0,old_control-control)/3)
        put('lost_block',max(0,float(source.get('block') or 0)-float(target.get('block') or 0))/3)
        if c['type']=='replace':
            put('lost_skill_levels',sum(s.get('level',0) for s in as_list(sp.get('skills')))/10)
            put('lost_advanced',self.env.effective_level(sp)>=4)
            plans=[p for p in as_list(c.get('development')) if p.get('grade',0)>=self.env.effective_level(sp)]
            put('rebuild_known',bool(plans) or self.env.effective_level(sp)==1)
            if plans or self.env.effective_level(sp)==1:
                extra=min((float(p['cost']) for p in plans),default=0)
                total=float(c.get('cost') or 0)+extra
                put('rebuild_cost',total/1000)
                put('rebuild_shortfall',max(0,total-float(self.o.get('gold') or 0)-float(c.get('refund') or 0))/1000)
        for future,prefix in ((False,'current'),(True,'future')):
            gain,before,after,_=self.comparison(c,future)
            put(prefix+'_damage_gain',gain/1000);put(prefix+'_gap_before',before/10000);put(prefix+'_gap_after',after/10000)
        put('keep_value',max(0,self.comparison(dict(c,type='replace',combat={'attacks':[]}),True)[2]-self.comparison(c,True)[1])/1000 if sp.get('state')=='tower' else 0)
        best_gain=0;best_cost=0
        if sp:
            for op in as_list(sp.get('options')):
                if op.get('locked') or op.get('price') is None: continue
                benefit,_,_=self.value(self.development(sp,op))
                if benefit>best_gain: best_gain=benefit;best_cost=op['price']
        put('best_upgrade_gain',best_gain);put('best_upgrade_cost',best_cost/1000)
        plan=self.core_plan(c)
        if plan and plan['score']>best_gain:
            put('best_upgrade_gain',plan['score']);put('best_upgrade_cost',plan['cost']/1000)
        if any(a.get('nonstack_group')=='tesla' for a in as_list(target.get('attacks'))):
            placed=self.placed(c);raw=self.contribution(placed,True).sum()
            without=[t for t in self.towers if t.get('hid')!=sp.get('hid')]
            marginal=(self.aggregate(without+[placed],True)-self.aggregate(without,True)).sum()
            put('repeat_output',max(0,raw-marginal)/1000)
        history=self.env.investments.get(sp.get('hid'),{})
        put('since_investment',min(1,max(0,self.env.t-history.get('tick',-10**9))*self.env.tick_seconds/120))
        put('recent_investment',history.get('cost',0)/1000 if self.env.t-history.get('tick',-10**9)<120/self.env.tick_seconds else 0)
        put('holder_replacements',self.env.holder_replacements.get(sp.get('hid'),0)/10)
        put('output_tier3_count',sum(self.env.tower_kind(t)!='barrack' and self.env.effective_level(t)>=3 for t in self.towers)/18)
        put('output_tier4_count',sum(self.env.tower_kind(t)!='barrack' and self.env.effective_level(t)>=4 for t in self.towers)/18)
        put('output_skill_levels',sum(s.get('level',0) for t in self.towers if self.env.tower_kind(t)!='barrack' for s in as_list(t.get('skills')))/30)
        if c['type']=='rain':
            hits,delay,uncertain=rain_hits(self.o,c['target']);p=mapping(self.o.get('rain_profile'))
            _,_,_,gaps=self.comparison(dict(type='wait'),False)
            indices={id(e):i for i,e in enumerate(self.enemies)}
            effective=sum(v for e,w,v in hits);avoidable=sum(min(v,gaps[indices[id(e)]]) for e,w,v in hits)
            emergency=sum(min(v,gaps[indices[id(e)]]) for e,w,v in hits if e.get('to_goal') is not None and float(e['to_goal'])<30)
            put('rain_hit_count',sum(w for e,w,v in hits)/20);put('rain_effective_hp',effective/1000)
            put('rain_avoidable_hp',avoidable/1000);put('rain_emergency',emergency/1000)
            cooldown=float(c.get('cooldown') or 60);left=self.o.get('next_wave_left')
            _,_,_,future_gaps=self.comparison(dict(type='wait'),True)
            reserve=0;known_hp=0;total_hp=0
            for i,e in enumerate(self.future):
                hp=float(e.get('count') or 0)*float(e.get('hp') or 0);total_hp+=hp
                if e.get('horizon')==2 or e.get('delay') is None or left is None: continue
                eta=float(left)+float(e['delay'])
                if 0<=eta<cooldown:
                    count=float(e.get('count') or 0);interval=float(e.get('interval_seconds') or 0)
                    arriving=count if interval<=0 else min(count,1+math.floor((cooldown-eta)/interval))
                    portion=arriving/max(1,count)
                    reserve+=future_gaps[i]*portion;known_hp+=hp*portion*(.5 if e.get('expected') else 1)
            put('rain_reserve_value',reserve/10000);put('rain_forecast_confidence',known_hp/max(1,total_hp))
            put('rain_radius',p.get('radius',60)/200);put('rain_spread',p.get('spread',20)/100)
            put('rain_delay_estimate',delay/10);put('rain_delay_known',p.get('delay_known'))
            put('rain_motion_uncertainty',uncertain)
        if c['type']=='wait':
            put('save_income_active',bool(self.enemies or self.o.get('wave_spawn_pending') or self.o.get('active_spawners')
                                         or (self.o.get('next_wave_left') is not None and float(self.o['next_wave_left'])>0)))
            plan=self.saving_plan()
            if plan:
                put('save_available',1);put('save_shortfall',plan['shortfall']/1000);put('save_cost',plan['cost']/1000)
                put('save_benefit',plan['benefit']);put('save_current_gain',plan['current']/1000)
                put('save_future_gain',plan['future']/1000);put('save_target_grade',plan['grade']/4)
                put('save_skill',plan['skill'])
                if plan['kind'] in ('archer','mage','engineer'): put('save_'+plan['kind'],1)
            _,_,_,gaps=self.comparison(c,False)
            put('saving_emergency',sum(gaps[i] for i,e in enumerate(self.enemies) if e.get('to_goal') is not None and float(e['to_goal'])<30)/1000)
        return x
