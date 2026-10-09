"""v5 concrete candidates with stable identities and explicit overflow handling."""
import math
from collections import Counter
import numpy as np
from gymnasium import spaces
from kr_strategy import KRStrategyEnv
from kr_macro import KRMacroEnv, KINDS
from candidate_features import Battlefield, FEATURE_DIM, SCENE_DIM, as_list

MAX_CANDIDATES=512
SCHEMA='candidate-v5'


class FocusSchedule:
    """Full shuffled sweeps plus bounded hard-level revisits, training levels only."""
    def __init__(self,base,every=3):
        self.base,self.every=base,every
        self.calls=0;self.failures=Counter();self.revisits=Counter()
    @property
    def pairs(self): return self.base.pairs
    def record(self,pair,won):
        self.failures[pair]=0 if won else self.failures[pair]+1
    def next(self,rng):
        self.calls+=1
        hard=[p for p in self.pairs if p[0]>=13 and self.failures[p]>0]
        if self.every and hard and self.calls%self.every==0:
            weights=np.array([min(3,self.failures[p])/(1+self.revisits[p]) for p in hard],float)
            selected=hard[int(rng.choice(len(hard),p=weights/weights.sum()))]
            self.revisits[selected]+=1
            return selected
        return self.base.next(rng)


class KRCandidateEnv(KRStrategyEnv):
    schema=SCHEMA
    base_meta=MAX_CANDIDATES
    n_actions=MAX_CANDIDATES+7
    battlefield_class=Battlefield
    feature_dim=FEATURE_DIM
    replacement_loss_cap=500

    def __init__(self,focus_every=3,**kwargs):
        self._table_key=None;self._table=[];self._executing=None
        self._vector_key=None;self._vector=None
        self._feature_field=None
        super().__init__(**kwargs)
        self.config['candidates']=True
        self.schedule=FocusSchedule(self.schedule,focus_every)
        self.observation_space=spaces.Dict({
            'scene':spaces.Box(-np.inf,np.inf,(SCENE_DIM,),np.float32),
            'candidates':spaces.Box(-np.inf,np.inf,(self.n_actions,FEATURE_DIM),np.float32)})

    def invalidate_candidates(self): self._table_key=None;self._vector_key=None

    def _offer(self,items,typ,command=None,spot=None,**kwargs):
        command=list(command) if command else []
        uid=repr(tuple(tuple(x) for x in command)) if command else 'wait'
        items.append(dict(type=typ,commands=command,spot=spot,uid=uid,**kwargs))

    def _legal_replacements(self,o,sp):
        wave=int(o.get('wave') or 0);source=self.tower_kind(sp)
        if source not in KINDS or sp.get('blocked') or wave==0: return []
        if self.wave_replacements.get(wave,0)>=self._quota(o) or self.t-self.last_replacement<self._interval(o): return []
        if wave-self.holder_wave.get(sp['hid'],-100)<2: return []
        left=o.get('next_wave_left')
        if left is not None and 0<float(left)<8: return []
        if not (o.get('forecast') or {}).get('spawns'): return []
        towers=[s for s in as_list(o.get('spots')) if s.get('state')=='tower']
        if len(towers)<4 or (source!='barrack' and sum(self.tower_kind(s)!='barrack' for s in towers)<=2): return []
        radius=float(sp.get('attack_range') or 120)+40
        if any(math.hypot(float(e.get('x',0))-float(sp.get('x',0)),float(e.get('y',0))-float(sp.get('y',0)))<radius
               for e in as_list(o.get('enemies'))): return []
        return [op for op in as_list(sp.get('replace_options')) if op.get('kind') in ('archer','mage','engineer')
                and op['kind']!=source and not op.get('locked') and op.get('price') is not None
                and float(o.get('gold') or 0)+float(sp.get('sell') or 0)>=op['price']]

    def power_target(self,power,target,index,o):
        if power in (0,2) and not any((e.get('x',0)-target['x'])**2+(e.get('y',0)-target['y'])**2<=90**2
                                      for e in as_list(o.get('enemies'))): return None
        return target

    def _generate(self,o):
        items=[];self._offer(items,'wait',cost=0)
        if o.get('outcome') or o.get('replacement_pending'): return items
        if o.get('can_call_wave') and (not self.auto_wave or int(o.get('wave',0))==0):
            self._offer(items,'wave',[('wave',)],cost=0)
        for i,sp in enumerate(as_list(o.get('spots'))):
            if sp.get('blocked'): continue
            if sp.get('state')=='empty':
                for k in KINDS:
                    for op in as_list(sp.get('options')):
                        if op.get('tpl')=='tower_build_'+k and op.get('ok') and not op.get('locked'):
                            self._offer(items,'build',[('build',sp['hid'],k)],sp,kind=k,cost=op['price'],grade=1,index=i,
                                        target_range=op.get('target_range'),target_paths=op.get('target_paths'))
            elif sp.get('state')=='tower':
                kind=self.tower_kind(sp)
                for op in as_list(sp.get('options')):
                    if op.get('ok') and not op.get('locked'):
                        self._offer(items,'upgrade',[('upgrade',sp['hid'],op['tpl'])],sp,kind=kind,
                                    branch=op['tpl'],cost=op.get('price'),grade=min(4,self.effective_level(sp)+1),index=i,
                                    target_range=op.get('target_range'),target_paths=op.get('target_paths'))
                for skill in as_list(sp.get('skills')):
                    if skill.get('ok'):
                        self._offer(items,'tower_skill',[('tower_skill',sp['hid'],skill['name'])],sp,
                                    kind=kind,skill=skill['name'],cost=skill['price'],skill_level=skill['level'],index=i)
                for op in self._legal_replacements(o,sp):
                    self._offer(items,'replace',[('replace',sp['hid'],op['kind'])],sp,kind=op['kind'],
                                cost=op['price'],refund=sp.get('sell',0),grade=1,index=i,
                                target_range=op.get('target_range'),target_paths=op.get('target_paths'),
                                loss=max(0,float(sp.get('spent') or 0)-float(sp.get('sell') or 0)))
                if self.rally_used_wave.get(i)!=int(o.get('wave') or 0):
                    for j,p in enumerate(as_list(sp.get('rally_points'))):
                        legal=as_list(sp.get('rally_ok'))
                        if p and j<len(legal) and legal[j] and math.hypot(p['x']-float(sp.get('rally_x') or 0),p['y']-float(sp.get('rally_y') or 0))>=5:
                            self._offer(items,'rally',[('rally',sp['hid'],j+1)],sp,kind=kind,target=p,cost=0,index=i)
        targets=as_list(o.get('targets'));powers=as_list(o.get('powers'))
        for power,typ in enumerate(('rain','reinf','ultimate')):
            if power>=len(powers): continue
            legal=as_list(powers[power].get('ok'))
            for j,target in enumerate(targets):
                if j>=len(legal) or not legal[j] or target.get('x') is None: continue
                # Rain keeps urgent small groups; density is a feature, not a ban.
                landing=self.power_target(power,target,j,o)
                if landing is None: continue
                self._offer(items,typ,[('power',power+1,j+1)],target=landing,cost=0,power=power+1,
                            target_index=j+1,cooldown=powers[power].get('cooldown_total',60))
        hero=o.get('hero') or {};legal=as_list(hero.get('move_ok'))
        interval=math.ceil(30/float(self.info_env.get('tick_length') or 1/30))
        if hero and not hero.get('dead') and self.t-self.last_hero_move_tick>=interval:
            if math.hypot(float(hero.get('x') or 0)-float(hero.get('rally_x',hero.get('x',0))),
                          float(hero.get('y') or 0)-float(hero.get('rally_y',hero.get('y',0))))<20:
                for j,target in enumerate(targets):
                    if j<len(legal) and legal[j] and target.get('x') is not None and math.hypot(
                        target['x']-float(hero.get('x') or 0),target['y']-float(hero.get('y') or 0))>=60:
                        self._offer(items,'hero_move',[('hero_move',j+1)],target=target,cost=0)
        unique={c['uid']:c for c in items}
        result=[unique.pop('wait')]+sorted(unique.values(),key=lambda c:c['uid'])
        if len(result)>MAX_CANDIDATES:
            raise RuntimeError(f'Candidate capacity exceeded: {len(result)} > {MAX_CANDIDATES}; no sites were truncated')
        return result

    def _entries(self,o):
        key=(id(o),self.t,self.phase,self.last_replacement,self.last_hero_move_tick)
        if key!=self._table_key:
            self._table=self._generate(o);self._table_key=key
        return self._table

    def _candidates(self,o):
        result={}
        for i,c in enumerate(self._entries(o)):
            typ=c['type'];extra=dict(kind=c.get('kind'),index=c.get('index'),power=c.get('power'),target=c.get('target_index'))
            attribution='power' if typ in ('rain','reinf','ultimate') else 'noop' if typ=='wait' else typ
            result[i]=(c['commands'],(attribution,c.get('spot'),extra),0)
        return result

    def _action_info(self,action,o):
        entries=self._executing if self._executing is not None and o is self.prev_o else self._entries(o)
        if not 0<=action<len(entries): return ('invalid',None,{})
        c=entries[action];typ=c['type']
        return ('power' if typ in ('rain','reinf','ultimate') else 'noop' if typ=='wait' else typ,c.get('spot'),
                dict(kind=c.get('kind'),index=c.get('index'),power=c.get('power'),target=c.get('target_index')))

    def action_kind(self,action):
        if action>=self.base_meta: return 'meta'
        entries=self._entries(self.o)
        if action>=len(entries): return 'invalid'
        typ=entries[action]['type']
        return {'wait':'noop','rain':'power1','reinf':'power2','ultimate':'power3'}.get(typ,typ)

    def reinforcements_available(self,mask):
        return any(mask[i] and c['type']=='reinf' for i,c in enumerate(self._entries(self.o)))

    def _vec(self,o):
        if o.get('strategy_version')!=1: raise RuntimeError('v5 requires v4 Lua strategy observation')
        if self.phase!='meta':
            self._entries(o)
            if self._vector_key==self._table_key: return self._vector
        field=self.battlefield_class(self,o)
        self._feature_field=field
        matrix=np.zeros((self.n_actions,self.feature_dim),dtype=np.float32)
        if self.phase=='meta':
            for i,ok in enumerate(self._meta_mask()[self.base_meta:]):
                if ok: matrix[self.base_meta+i]=field.vector(dict(type='meta',cost=None))
        else:
            for i,c in enumerate(self._entries(o)): matrix[i]=field.vector(c)
        result={'scene':field.scene(),'candidates':matrix}
        if self.phase!='meta': self._vector=result;self._vector_key=self._table_key
        return result

    def step(self,action):
        if self.phase=='meta': return KRMacroEnv.step(self,action)
        previous=self._entries(self.o)
        previous_vector=self._vector if self._vector_key==self._table_key else None
        selected=previous[action] if action<len(previous) else None
        self._executing=previous
        old_result=self.o.get('replacement_result');old_wave=int(self.o.get('wave') or 0)
        result=KRMacroEnv.step(self,action)
        obs,reward,term,trunc,info=result
        if selected and info.get('agent_action_ok'):
            typ=selected['type']
            if typ=='replace':
                self.last_replacement=self.t;self.wave_replacements[old_wave]=self.wave_replacements.get(old_wave,0)+1
                self.holder_wave[selected['spot']['hid']]=old_wave
            elif typ=='hero_move': self.last_hero_move_tick=self.t
            elif typ=='tower_skill':
                self.ep.setdefault('tower_skills',[]).append(dict(hid=selected['spot']['hid'],skill=selected['skill'],
                    cost=selected['cost'],wave=old_wave,tick=self.t))
        replacement=self.o.get('replacement_result')
        if replacement and replacement!=old_result:
            if not replacement.get('ok'): raise RuntimeError('Replacement rebuild failed: '+str(replacement))
            loss=float(replacement.get('loss') or 0)
            if self.replacement_loss_cap is not None: loss=min(loss,self.replacement_loss_cap)
            loss=-loss*.001
            reward+=loss;self.reward_totals['replacement_loss']+=loss
            info.setdefault('reward_components',{})['replacement_loss']=loss
            self.ep.setdefault('replacements',[]).append(replacement)
        info['candidate_action_kinds']=[{'wait':'noop','rain':'power1','reinf':'power2','ultimate':'power3'}.get(c['type'],c['type']) for c in previous]
        if selected and (selected['type']!='wait' or old_wave!=int(self.o.get('wave') or 0)):
            info['candidate_event']=dict(uid=selected['uid'],type=selected['type'],wave=old_wave,tick=self.t,
                gold_before=self.prev_o.get('gold'),gold_after=self.o.get('gold'),cost=selected.get('cost'),
                refund=selected.get('refund',0),candidate_count=len(previous),ok=info.get('agent_action_ok'),
                target=selected.get('target'),features=(previous_vector['candidates'][action] if previous_vector is not None
                    else self.battlefield_class(self,self.prev_o).vector(selected)).tolist())
            if selected['type'] in ('rain','replace','tower_skill','wave') or self.t%(20*self.ticks_per_step)==0:
                info['candidate_catalog']=[dict(uid=c['uid'],type=c['type']) for c in previous]
        if selected and selected['type']=='rain' and info.get('agent_action_ok'):
            self.ep.setdefault('rain_events',[]).append(info['candidate_event'])
            self.ep['rain_window_until']=self.t+math.ceil(selected.get('cooldown',60)/float(self.info_env.get('tick_length') or 1/30))
        if self.t<self.ep.get('rain_window_until',0): self.ep['rain_cooldown_leaks']=self.ep.get('rain_cooldown_leaks',0)+max(0,-int(info.get('d_lives',0)))
        if 'ep_stats' in info:
            info['ep_stats'].update(tower_skills=self.ep.get('tower_skills',[]),replacements=self.ep.get('replacements',[]),
                rain_events=self.ep.get('rain_events',[]),rain_cooldown_leaks=self.ep.get('rain_cooldown_leaks',0),
                reward_parts=dict(self.reward_totals))
            if self.evaluation_pair is None: self.schedule.record(self.selected_pair,info.get('outcome')=='victory')
        self._executing=None
        # Quotas and cooldowns changed after the native step; refresh stored rollout obs.
        return self._vec(self.o),float(reward),term,trunc,info
