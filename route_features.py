"""v7 route handling estimates. All values are observations, never rewards.

Leak estimates are continuous risk proxies, not a simulated kill count. A
route crosses the handling threshold only when every visible enemy batch fits
its estimated damage budget. Geometry and unparsed regeneration stay partial.
"""
import math
import numpy as np
from planning_features import PlanningField, mapping, rates
from planning_features import FEATURE_NAMES as V6_NAMES, SCENE_NAMES, as_list
from candidate_features import BRANCHES
from combat_estimates import effective_attacks,attack_dps
from route_statistics import finalize_route
from route_survival import estimate_survival,SurvivalContext
from skill_features import FEATURE_NAMES as SKILL_NAMES,feature_values as skill_values

ROUTES = tuple(range(1,17))+(0,)  # 0 retains unknown/out-of-range paths.
HORIZONS = ('current','next','second')
ROUTE_METRICS = ('present','margin_before','margin_after','bottleneck_after',
                 'leaks_before','leaks_after','life_before','life_after','useful_gain',
                 'crosses_threshold','shortfall_after','regen_before','regen_after','partial',
                 'leak_fraction_before','leak_fraction_after','life_fraction_before','life_fraction_after',
                 'worst_shortfall_before','worst_shortfall_after','generation_risk')
PLAN_METRICS = ('chain_margin','spread_margin','chain_leaks','spread_leaks','chain_crosses','spread_crosses')
FEATURE_NAMES = V6_NAMES+tuple(f'route_{h}_{r}_{m}' for h in HORIZONS for r in ROUTES for m in ROUTE_METRICS)
FEATURE_NAMES += tuple('chain_'+branch for branch in BRANCHES)
FEATURE_NAMES += ('chain_available','chain_budget','chain_partial','spread_available','spread_spent','spread_unspent')
FEATURE_NAMES += tuple(f'plan_{r}_{m}' for r in ROUTES for m in PLAN_METRICS)
FEATURE_NAMES += SKILL_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
SCENE_DIM = len(SCENE_NAMES)
F = {name:i for i,name in enumerate(FEATURE_NAMES)}


def route_id(enemy):
    try:
        value = int(enemy.get('pi') or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value in ROUTES else 0


class RouteDamageField(PlanningField):
    def __init__(self,env,observation):
        super().__init__(env,observation)
        self._sample_sets={};self._exposure_cache={}
        self._segments={str(pi):tuple((float(s['start']),float(s['end']),float(s['length']),
            round(float(s['start']),6),round(float(s['end']),6)) for s in as_list(g.get('segments')))
            for pi,g in mapping(observation.get('route_geometry')).items()}

    def samples(self,tower,pi):
        values=tuple(as_list(mapping(tower.get('defense_samples')).get(str(pi))))
        if values not in self._sample_sets:
            self._sample_sets[values]=frozenset(round(float(p),6) for p in values)
        return self._sample_sets[values]

    def attack_population(self,enemies,eligible,attack,future):
        # An attack's budget is not diluted by enemies it cannot target.
        return sum(float(enemies[i].get('count') or 0) if future else 1 for i in eligible
                   if not mapping(enemies[i].get('mechanics')).get('immune')
                   and (not enemies[i].get('flying') or attack.get('flying')))

    def exposure(self,tower,enemy,future):
        geometry=mapping(mapping(self.o.get('route_geometry')).get(str(enemy.get('pi'))))
        samples=self.samples(tower,enemy.get('pi'))
        segments=self._segments.get(str(enemy.get('pi')),())
        if not segments or not samples or float(enemy.get('speed') or 0)<=0:
            return super().exposure(tower,enemy,future)
        progress=0 if future else float(enemy.get('progress') or 0)
        key=(str(enemy.get('pi')),samples,progress,float(enemy['speed']))
        if key in self._exposure_cache:return self._exposure_cache[key]
        length=0.0
        for start,end,segment_length,left,right in segments:
            if end<=progress:
                continue
            weight=(float(left in samples)+float(right in samples))/2
            length+=segment_length*weight*(end-max(start,progress))/max(1e-9,end-start)
        result=length/float(enemy['speed']);self._exposure_cache[key]=result
        return result


class RouteField(RouteDamageField):
    def __init__(self, env, observation):
        super().__init__(env, observation)
        self._horizon_fields = {'current':self}
        self._route_before = {}
        self._route_cache = {}
        self._chain_cache = {}
        self._spread_cache = {}
        self._uncovered_cache={};self._survival_windows={}
        self._state_cache={};self._value_ids={};self._object_ids={};self._goals={}
        self._coverage_unions={};self._output_flags={}
        self._survival_contexts={}

    def value_id(self,value):
        """Intern JSON values once, retaining objects to avoid Python id reuse."""
        if isinstance(value,(dict,list,tuple)):
            cached=self._object_ids.get(id(value))
            if cached is not None and cached[0] is value:return cached[1]
            key=('dict',tuple((k,self.value_id(v)) for k,v in sorted(value.items()))) if isinstance(value,dict) else (
                'sequence',tuple(self.value_id(v) for v in value))
        else:key=(type(value).__name__,value)
        token=self._value_ids.setdefault(key,len(self._value_ids))
        if isinstance(value,(dict,list,tuple)):self._object_ids[id(value)]=(value,token)
        return token

    def tower_key(self,tower):
        return (bool(tower.get('blocked')),tower.get('attack_range'),
            *(self.value_id(tower.get(k)) for k in ('combat','defense_paths','defense_lengths','defense_samples')))

    def development(self,spot,option):
        key=(id(spot),id(option))
        if key not in self._goals:self._goals[key]=super().development(spot,option)
        return self._goals[key]

    def horizon_field(self, horizon):
        if horizon not in self._horizon_fields:
            # No count discount: each horizon describes a separate wave.
            obs = dict(self.o, enemies=[], forecast_second={})
            obs['forecast'] = self.o.get('forecast' if horizon=='next' else 'forecast_second') or {}
            self._horizon_fields[horizon] = RouteDamageField(self.env, obs)
        return self._horizon_fields[horizon]

    def candidate_towers(self, candidate):
        typ = candidate['type']; source = candidate.get('spot') or {}
        towers = self.towers
        if typ in ('replace','upgrade','tower_skill') and source.get('state')=='tower':
            towers = [t for t in towers if t.get('hid')!=source.get('hid')]
        if typ in ('build','replace','upgrade','tower_skill'):
            towers = towers+[self.placed(candidate)]
        return towers

    def uncovered_time(self, enemy, towers, future, state_key=None):
        geometry = mapping(mapping(self.o.get('route_geometry')).get(str(enemy.get('pi'))))
        segments = self._segments.get(str(enemy.get('pi')),())
        speed = float(enemy.get('speed') or 0)
        if not segments or speed<=0:
            return 0.0, True
        progress = 0 if future else float(enemy.get('progress') or 0)
        layout=state_key if state_key is not None else tuple(self.tower_key(t) for t in towers)
        union_key=(layout,str(enemy.get('pi')),bool(enemy.get('flying')))
        if union_key not in self._coverage_unions:
            covered=set()
            for tower in towers:
                if tower.get('blocked'):continue
                profile=mapping(tower.get('combat'))
                flag_key=(self.value_id(profile),bool(enemy.get('flying')))
                if flag_key not in self._output_flags:
                    self._output_flags[flag_key]=any(attack_dps(a)>0 and (not enemy.get('flying') or a.get('flying'))
                        for a in effective_attacks(profile))
                # Barracks alone still provide no output coverage.
                if self._output_flags[flag_key]:covered.update(self.samples(tower,enemy.get('pi')))
            self._coverage_unions[union_key]=frozenset(covered)
        covered=self._coverage_unions[union_key]
        key=(str(enemy.get('pi')),speed,progress,frozenset(covered))
        if key in self._uncovered_cache:return self._uncovered_cache[key]
        remaining = 0.0
        for start,end,segment_length,left_node,right_node in segments:
            if end<=progress:
                continue
            fraction = (end-max(start,progress))/max(1e-9,end-start)
            left = left_node in covered; right = right_node in covered
            # Union of samples, not sum of tower coverage; gaps retain HP cost.
            coverage = (float(left)+float(right))/2
            remaining += segment_length*fraction*(1-coverage)
        result=remaining/speed, bool(geometry.get('partial', True))
        self._uncovered_cache[key]=result
        return result

    def route_state(self, horizon, towers):
        key=(horizon,tuple(self.tower_key(t) for t in towers))
        if key in self._state_cache:return self._state_cache[key]
        field = self.horizon_field(horizon)
        future = horizon!='current'
        enemies = field.future if future else field.enemies
        damage = field.aggregate(towers, future)
        rows = {r:dict(present=False,hp=0.0,damage=0.0,leaks=0.0,life=0.0,
                       count=0.0,regular_count=0.0,regular_leaks=0.0,life_budget=0.0,bottleneck=float('inf'),regen=0.0,partial=False,generation_risk=False) for r in ROUTES}
        for i,enemy in enumerate(enemies):
            count = max(0,float(enemy.get('count') or 0)) if future else 1.0
            if count<=0:
                continue
            row = rows[route_id(enemy)]; row['present'] = True
            hp = enemy.get('hp')
            mechanics = mapping(enemy.get('mechanics'))
            row['generation_risk'] |= bool(mechanics.get('summon') or mechanics.get('replication') or mechanics.get('transform'))
            gap_time, geometry_partial = self.uncovered_time(enemy,towers,future,key)
            regen_known = bool(mechanics.get('regen_known')) or not mechanics.get('regen')
            regen = max(0,float(mechanics.get('regen_rate') or 0))*gap_time if regen_known else 0.0
            effective_hp = max(1,float(hp or 0)+regen)*count
            fraction = float(np.clip(1-damage[i]/effective_hp,0,1))
            leaks = count*fraction
            row['hp'] += effective_hp; row['damage'] += float(damage[i])
            row['count'] += count; row['leaks'] += leaks
            if not enemy.get('special') and not enemy.get('expected'):
                row['regular_count']+=count;row['regular_leaks']+=leaks
            lives_cost=max(0,float(enemy.get('lives_cost') if enemy.get('lives_cost') is not None else 1))
            row['life'] += leaks*lives_cost
            row['life_budget'] += count*lives_cost
            row['regen'] += regen*count
            row['bottleneck'] = min(row['bottleneck'],float(damage[i])/effective_hp)
            row['partial'] |= hp is None or not regen_known or geometry_partial or route_id(enemy)==0
            row['partial'] |= bool(mechanics.get('regen')) and not mechanics.get('regen_rule_known')
            row['partial'] |= enemy.get('lives_cost') is None or bool(enemy.get('expected'))
            row['partial'] |= bool(mechanics.get('death_spawn') or mechanics.get('transform') or mechanics.get('summon') or mechanics.get('replication'))
            row['partial'] |= future and not enemy.get('timing_known')
        forecast = mapping(self.o.get('forecast' if horizon=='next' else 'forecast_second'))
        horizon_partial = (bool(self.o.get('wave_spawn_pending') or self.o.get('active_spawners'))
                           if horizon=='current' else not forecast.get('available'))
        combat_partial = any(rates(mapping(t.get('combat')))[-1] for t in towers if not t.get('blocked'))
        survival=None
        if future:
            geometries=mapping(self.o.get('route_geometry'))
            if horizon not in self._survival_contexts:
                self._survival_contexts[horizon]=SurvivalContext(enemies,geometries)
            survival=estimate_survival(enemies,towers,geometries,self._survival_windows,self._survival_contexts[horizon])
        if survival is not None:
            for r,row in rows.items():
                if not row['present']:continue
                row['damage_gap_leaks']=row['leaks']
                result=survival.get(r,{})
                row['leaks']=result.get('leaks',0.)
                row['life']=result.get('life',0.)
                row['regular_leaks']=result.get('regular_leaks',0.)
                row['partial']=True  # Sampling, target ordering and scripted mechanics remain approximate.
        elif future:
            for row in rows.values():row['partial']=True
        for row in rows.values():
            row['partial'] |= horizon_partial or combat_partial
            finalize_route(row)
        self._state_cache[key]=rows
        return rows

    def route_comparison(self, candidate, horizon):
        source = candidate.get('spot') or {}
        key = (horizon,candidate['type'],source.get('hid'),id(candidate.get('combat')),
               id(candidate.get('target_paths')),id(candidate.get('target_samples')),
               id(candidate.get('target_lengths')),candidate.get('target_range'))
        if key not in self._route_cache:
            if horizon not in self._route_before:
                self._route_before[horizon] = self.route_state(horizon,self.towers)
            before = self._route_before[horizon]
            after = self.route_state(horizon,self.candidate_towers(candidate)) if candidate['type'] in ('build','upgrade','replace','tower_skill') else before
            self._route_cache[key] = before,after
        return self._route_cache[key]

    def value(self, candidate):
        current = self.route_comparison(candidate,'current')
        future = self.route_comparison(candidate,'next')
        def benefit(pair):
            return sum(pair[0][r]['life']-pair[1][r]['life'] for r in ROUTES)
        a,b = benefit(current),benefit(future)
        return 2*a+b,a,b

    def core_plan(self, candidate):
        if candidate['type'] not in ('build','upgrade','tower_skill'):
            return None
        spot = candidate.get('spot') or {}; best = None
        key=(candidate['type'],spot.get('hid'),candidate.get('cost'),candidate.get('kind'),
             id(candidate.get('development',spot.get('development'))))
        if key in self._chain_cache:
            return self._chain_cache[key]
        for option in as_list(candidate.get('development',spot.get('development'))):
            if option.get('grade',0)<4 or not option.get('combat'):
                continue
            goal = self.development(spot,option)
            if spot.get('state')!='tower':
                goal['type']='build'
            benefit,current,future = self.value(goal)
            cost = float(option.get('cost') or 0)+float(candidate.get('cost') or 0)
            score = benefit/(1+cost/500)
            if best is None or score>best['score']:
                best = dict(score=score,cost=cost,benefit=benefit,current=current,future=future,
                            grade=option['grade'],skill=bool(option.get('skill')),
                            kind=candidate.get('kind') or self.env.tower_kind(spot),hid=spot.get('hid'),
                            goal=goal,branch=option.get('tpl'))
        self._chain_cache[key]=best
        return best

    def spread_reference(self, candidate, budget):
        """Feasible same-budget expansion reference, not an optimal build search."""
        source = candidate.get('spot') or {}; routes = set(mapping(candidate.get('target_paths') or source.get('defense_paths')))
        key=(budget,tuple(sorted(routes)))
        if key in self._spread_cache:
            return self._spread_cache[key]
        offers = []
        for spot in as_list(self.o.get('spots')):
            if spot.get('state')!='empty' or spot.get('blocked'):
                continue
            for op in as_list(spot.get('options')):
                cost = float(op.get('price') or 0)
                if op.get('locked') or cost<=0 or not op.get('combat'):
                    continue
                name = op.get('tpl',''); kind = name.removeprefix('tower_build_')
                if kind not in ('archer','mage','engineer','barrack'):
                    continue
                paths = mapping(op.get('target_paths')); shared = len(routes & set(paths))
                score = (1+shared)*rates(op['combat'])[0].sum()/cost
                offers.append((score,spot['hid'],cost,dict(type='build',spot=spot,kind=kind,grade=1,
                    combat=op['combat'],target_paths=paths,target_range=op.get('target_range'),
                    target_samples=op.get('target_samples'),target_lengths=op.get('target_lengths'))))
        used = set(); towers = list(self.towers); spent = 0.0
        for score,hid,cost,offer in sorted(offers,key=lambda item:(-item[0],item[1])):
            if hid in used or spent+cost>budget:
                continue
            towers.append(self.placed(offer)); used.add(hid); spent+=cost
        result=self.route_state('next',towers),spent,bool(used)
        self._spread_cache[key]=result
        return result

    def vector(self, candidate):
        x = np.zeros(FEATURE_DIM,np.float32)
        x[:len(V6_NAMES)] = super().vector(candidate)
        # v7 planning benefits use life/leak units, not the v6 HP / 1000 scale.
        x[F['save_current_gain']]*=1000
        x[F['save_future_gain']]*=1000
        def put(name,value):
            x[F[name]] = float(value or 0)
        for name,value in skill_values(self,candidate).items():put(name,value)
        for horizon in HORIZONS:
            before,after = self.route_comparison(candidate,horizon)
            for route in ROUTES:
                a,b = before[route],after[route]; prefix = f'route_{horizon}_{route}_'
                values = dict(present=b['present'],margin_before=a['margin'],margin_after=b['margin'],
                    bottleneck_after=b['bottleneck_margin'],leaks_before=a['leaks']/20,leaks_after=b['leaks']/20,
                    life_before=a['life']/20,life_after=b['life']/20,
                    useful_gain=(a['leaks']-b['leaks'])/20,
                    crosses_threshold=not a['handles'] and b['handles'],shortfall_after=b['leaks']/max(1,b['count']),
                    regen_before=a['regen']/1000,regen_after=b['regen']/1000,partial=a['partial'] or b['partial'],
                    leak_fraction_before=a['leak_fraction'],leak_fraction_after=b['leak_fraction'],
                    life_fraction_before=a['life_fraction'],life_fraction_after=b['life_fraction'],
                    worst_shortfall_before=a['worst_shortfall'],worst_shortfall_after=b['worst_shortfall'],generation_risk=b['generation_risk'])
                for name,value in values.items():
                    put(prefix+name,value)
                if a['partial'] or b['partial']:
                    put('combat_partial',1)
        plan = self.core_plan(candidate)
        if plan:
            put('chain_available',1); put('chain_budget',plan['cost']/1000)
            if plan['branch'] in BRANCHES:
                put('chain_'+plan['branch'],1)
            before,goal = self.route_comparison(plan['goal'],'next')
            spread,spent,available = self.spread_reference(candidate,plan['cost'])
            put('spread_available',available);put('spread_spent',spent/1000)
            put('spread_unspent',(plan['cost']-spent)/1000)
            put('chain_partial',any(v['partial'] for v in goal.values()))
            for route in ROUTES:
                prefix = f'plan_{route}_'
                for name,value in dict(chain_margin=goal[route]['margin'],spread_margin=spread[route]['margin'],
                    chain_leaks=goal[route]['leaks']/20,spread_leaks=spread[route]['leaks']/20,
                    chain_crosses=not before[route]['handles'] and goal[route]['handles'],
                    spread_crosses=not before[route]['handles'] and spread[route]['handles']).items():
                    put(prefix+name,value)
        return x
