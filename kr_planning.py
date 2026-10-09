"""v6: preserve defense investments and plan powers and unaffordable upgrades."""
import math
from collections import Counter
import numpy as np
from gymnasium import spaces
from kr_candidates import KRCandidateEnv,MAX_CANDIDATES,FocusSchedule
from candidate_features import as_list
from planning_features import PlanningField,FEATURE_DIM,SCENE_DIM,mapping,rain_hits

SCHEMA='candidate-v6'
PLANNING_REVISION=3


class ProgressFocus(FocusSchedule):
    """Prefer failed levels with some progress, while retaining complete sweeps."""
    def __init__(self,base,every=3):
        super().__init__(base,every);self.progress={}
    def record_progress(self,pair,progress): self.progress[pair]=float(progress)
    def next(self,rng):
        self.calls+=1
        hard=[p for p in self.pairs if p[0]>=13 and self.failures[p]>0]
        if self.every and hard and self.calls%self.every==0:
            weights=np.asarray([(.25+min(.8,self.progress.get(p,0)))*min(3,self.failures[p])/(1+self.revisits[p]) for p in hard])
            selected=hard[int(rng.choice(len(hard),p=weights/weights.sum()))]
            self.revisits[selected]+=1;return selected
        return self.base.next(rng)


class KRPlanningEnv(KRCandidateEnv):
    schema=SCHEMA
    battlefield_class=PlanningField
    feature_dim=FEATURE_DIM
    replacement_loss_cap=None

    def _quota(self,o):
        if int(o.get('level_idx') or 1)>=13:
            return 2 if float(o.get('gold') or 0)>=1500 else 1
        return super()._quota(o)

    def __init__(self,fixed_game_seed=None,**kwargs):
        if fixed_game_seed is not None and not 1<=int(fixed_game_seed)<2**31-1:
            raise ValueError('fixed_game_seed must be in 1..2147483646')
        self.fixed_game_seed=None if fixed_game_seed is None else int(fixed_game_seed)
        self.investments={};self.holder_replacements=Counter();self.rain_tracking=[]
        self.game_seed=None
        super().__init__(**kwargs)
        self.config['planning']=True
        self.config['fixed_game_seed']=self.fixed_game_seed
        self.schedule=ProgressFocus(self.schedule.base,self.schedule.every)
        self.observation_space=spaces.Dict({
            'scene':spaces.Box(-np.inf,np.inf,(SCENE_DIM,),np.float32),
            'candidates':spaces.Box(-np.inf,np.inf,(self.n_actions,FEATURE_DIM),np.float32)})

    @property
    def tick_seconds(self): return float(self.info_env.get('tick_length') or 1/30)

    def reset(self,**kwargs):
        import gymnasium as gym
        gym.Env.reset(self,seed=kwargs.get('seed'))
        self.game_seed=self.fixed_game_seed if self.fixed_game_seed is not None else int(self.np_random.integers(1,2**31-1))
        # Seed before loading and restarting the next level, not after enemies
        # have already been spawned. Every game instance owns its Lua RNG.
        self.client.call('(function() math.randomseed(%d); return "{}" end)()'%self.game_seed)
        self.investments={};self.holder_replacements.clear();self.rain_tracking=[]
        self.invalidate_candidates()
        return super().reset(**kwargs)

    def _info(self,o):
        info=super()._info(o);info['game_seed']=self.game_seed
        info['planning_revision']=PLANNING_REVISION
        return info

    def power_target(self,power,target,index,o):
        powers=as_list(o.get('powers'));points=as_list(powers[power].get('points')) if power<len(powers) else []
        point=points[index] if index<len(points) else None
        landing=dict(target,**mapping(point))
        if power==0:
            profile=mapping(o.get('rain_profile'))
            envelope=float(profile.get('radius') or 60)+math.sqrt(2)*float(profile.get('spread') or 20)
            if not any(math.hypot(float(e.get('x') or 0)-landing['x'],float(e.get('y') or 0)-landing['y'])<=envelope
                       for e in as_list(o.get('enemies'))): return None
            return landing
        return super().power_target(power,landing,index,o)

    def _generate(self,o):
        entries=super()._generate(o)
        for c in entries:
            sp=c.get('spot') or {};typ=c['type']
            if typ in ('build','upgrade'):
                op=next((op for op in as_list(sp.get('options')) if op.get('tpl')==c['commands'][0][-1]
                         or (typ=='build' and op.get('tpl')=='tower_build_'+c['kind'])),{})
            elif typ=='tower_skill':
                op=next((op for op in as_list(sp.get('skills')) if op['name']==c['skill']),{})
            elif typ=='replace':
                op=next((op for op in as_list(sp.get('replace_options')) if op['kind']==c['kind']),{})
            else: continue
            for name in ('combat','target_lengths','target_samples','development'):
                if name in op: c[name]=op[name]
        return entries

    def _vec(self,o):
        if o.get('planning_version')!=1:
            raise RuntimeError('v6 requires planning Lua observation; reload current kr_env.lua')
        return super()._vec(o)

    def step(self,action):
        old=self.o;entries=self._entries(old)
        selected=entries[action] if 0<=action<len(entries) else None
        tick=self.t
        result=super().step(action)
        obs,reward,term,trunc,info=result
        if selected and info.get('agent_action_ok'):
            typ=selected['type'];sp=selected.get('spot') or {};hid=sp.get('hid')
            if typ in ('build','upgrade','tower_skill','replace'):
                self.investments[hid]=dict(tick=self.t,cost=selected.get('cost') or 0,type=typ)
                if typ=='replace': self.holder_replacements[hid]+=1
                if typ=='replace' and self.effective_level(sp)>=3 and self.tower_kind(sp)!='barrack':
                    self.ep['advanced_output_replacements']=self.ep.get('advanced_output_replacements',0)+1
                self.invalidate_candidates()
            if typ=='rain':
                hits,delay,uncertain=rain_hits(old,selected['target'])
                self.rain_tracking.append(dict(start=tick,until=self.t+math.ceil((delay+3)/self.tick_seconds),
                    ids=[e['id'] for e,w,v in hits if w>0 and e.get('id') is not None],
                    hp_before=sum(float(e.get('hp') or 0) for e,w,v in hits if w>0),
                    predicted_hits=sum(w for e,w,v in hits),predicted_damage=sum(v for e,w,v in hits),
                    target=selected['target'],motion_uncertainty=uncertain))
        alive={e.get('id'):e for e in as_list(self.o.get('enemies'))}
        pending=[]
        for event in self.rain_tracking:
            if self.t<event['until'] and not (term or trunc): pending.append(event);continue
            hp_after=sum(float(alive[eid].get('hp') or 0) for eid in event['ids'] if eid in alive)
            # Disappearance can mean killed or leaked. This is joint-defense
            # diagnostic evidence, not damage credited to fire rain or a reward.
            self.ep.setdefault('rain_outcomes',[]).append(dict(event,hp_after=hp_after,
                disappeared=sum(eid not in alive for eid in event['ids']),joint_hp_change=event['hp_before']-hp_after))
        self.rain_tracking=pending
        outputs=[t for t in as_list(self.o.get('spots')) if t.get('state')=='tower' and self.tower_kind(t)!='barrack']
        for grade in (3,4):
            key=f'tier{grade}_output_seconds'
            self.ep[key]=self.ep.get(key,0)+(self.t-tick)*self.tick_seconds*sum(self.effective_level(t)>=grade for t in outputs)
        if 'ep_stats' in info:
            planning=dict(revision=PLANNING_REVISION,investments=len(self.investments),holder_replacements=dict(self.holder_replacements),
                tier3_outputs=sum(self.tower_kind(t)!='barrack' and self.effective_level(t)>=3 for t in as_list(self.o.get('spots')) if t.get('state')=='tower'),
                tier4_outputs=sum(self.tower_kind(t)!='barrack' and self.effective_level(t)>=4 for t in as_list(self.o.get('spots')) if t.get('state')=='tower'))
            planning.update(tier3_output_seconds=self.ep.get('tier3_output_seconds',0),
                tier4_output_seconds=self.ep.get('tier4_output_seconds',0),advanced_output_replacements=self.ep.get('advanced_output_replacements',0))
            info['ep_stats'].update(planning=planning,rain_outcomes=self.ep.get('rain_outcomes',[]))
            if self.evaluation_pair is None:
                self.schedule.record_progress(self.selected_pair,max(0,int(self.o.get('wave') or 0)-1)/max(1,int(self.o.get('wave_total') or 1)))
        return self._vec(self.o),reward,term,trunc,info
