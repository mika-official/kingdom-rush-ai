"""Conservative sell-and-rebuild macros; original tactical policies remain v2."""
import math
import numpy as np
from gymnasium import spaces
from kr_tactical import KRTacticalEnv, TACTICAL_MACROS, enemy_profile, support_at
from kr_macro import KINDS

REPLACEMENT_MACROS = TACTICAL_MACROS + tuple('replace_with_'+k for k in KINDS)
BRANCHES = {'ranger':'archer','musketeer':'archer','paladin':'barrack','barbarian':'barrack',
            'arcane':'mage','sorcerer':'mage','bfg':'engineer','tesla':'engineer'}


class KRReplacementEnv(KRTacticalEnv):
    schema = 'macro-v3-replacement'
    base_meta = len(REPLACEMENT_MACROS)
    n_actions = base_meta + 7
    rich_gold_threshold = 1000

    def __init__(self, **kwargs):
        self.last_replacement = -10**9
        self.replaced_wave = -1
        self.wave_replacements = {}
        self.holder_wave = {}
        super().__init__(**kwargs)
        n = self.observation_space['summary'].shape[0]+4*12+2
        self.observation_space = spaces.Dict({'summary':spaces.Box(-np.inf,np.inf,(n,),np.float32)})

    @staticmethod
    def tower_kind(sp):
        kind=sp.get('kind') or sp.get('name','').removeprefix('tower_')
        return BRANCHES.get(kind) or KRTacticalEnv.tower_kind(sp)

    @staticmethod
    def effective_level(sp):
        name=sp.get('name','').removeprefix('tower_')
        return 4 if name in BRANCHES or sp.get('kind') in BRANCHES else max(1,int(sp.get('level') or 1))

    def _quota(self, o):
        # Campaign stage, not the game's separate difficulty setting.
        level = int(o.get('level_idx') or 1)
        if level <= 6:
            return 1
        if level <= 12:
            return 2
        return 4 if float(o.get('gold') or 0) >= self.rich_gold_threshold else 3

    def _interval(self, o=None):
        if int((self.o if o is None else o).get('level_idx') or 1) > 6:
            return 0
        return math.ceil(60/float(self.info_env.get('tick_length') or (1/30)))

    def _replacement_plans(self,o):
        wave=int(o.get('wave') or 0)
        forecast=(o.get('forecast') or {}).get('spawns') or []
        if (o.get('outcome') or o.get('replacement_pending') or not forecast or wave == 0
            or self.wave_replacements.get(wave,0) >= self._quota(o)
            or self.t-self.last_replacement < self._interval(o)):
            return {}
        if o.get('next_wave_left') is not None and 0 < float(o['next_wave_left']) < 8:
            return {}
        towers=[s for s in o.get('spots') or [] if s.get('state')=='tower']
        if len(towers)<4:
            return {}
        out={}
        cm=self._cmax(o)
        for i,sp in enumerate(o.get('spots') or []):
            source=self.tower_kind(sp)
            if sp.get('state')!='tower' or source not in KINDS or sp.get('blocked'):
                continue
            if wave-self.holder_wave.get(sp['hid'],-100) < 2:
                continue
            radius=float(sp.get('attack_range') or sp.get('cov',{}).get(source,{}).get('r') or 120)+40
            if any(math.hypot(float(e.get('x') or 0)-float(sp.get('x') or 0),
                              float(e.get('y') or 0)-float(sp.get('y') or 0)) < radius for e in o.get('enemies') or []):
                continue
            routes=sp.get('cov',{}).get(source,{}).get('paths') or {}
            local=[e for e in forecast if not routes or routes.get(str(e.get('pi')),0)>0]
            if not local:
                continue
            p=enemy_profile(local,True)
            affinity={'archer':float(p[5]),'mage':float(p[7]),
                      'engineer':float(p[8])*(.3+float(p[9])),
                      'barrack':float(p[6])*(.2+min(.4,support_at(sp,towers)*.1))}
            level=self.effective_level(sp)
            old=affinity[source]*(1+.4*(level-1))
            # Retain the last two output towers; do not turn a defended route
            # into mostly blockers. Replacement itself never selects barracks.
            if source!='barrack' and sum(self.tower_kind(t)!='barrack' for t in towers)<=2:
                continue
            for option in sp.get('replace_options') or []:
                target=option.get('kind')
                if target not in ('archer','mage','engineer') or target==source or option.get('locked'):
                    continue
                price=option.get('price')
                if price is None or float(o.get('gold') or 0)+float(sp.get('sell') or 0)<price:
                    continue
                new=affinity[target]
                if new < old*1.35 or new-old < .2:
                    continue
                # Prefer expansion over destructive rebuilding when an affordable
                # empty site offers at least this site's coverage for that kind.
                quality=self._quality(sp,target,cm)
                target_routes=sp.get('cov',{}).get(target,{}).get('paths') or {}
                if target_routes and not any(target_routes.get(str(e.get('pi')),0)>0 for e in local):
                    continue
                if any(s.get('state')=='empty' and self._quality(s,target,cm)>=quality and
                       any(op.get('tpl')=='tower_build_'+target and op.get('ok') and not op.get('locked')
                           for op in s.get('options') or []) for s in o.get('spots') or []):
                    continue
                loss=max(0,float(sp.get('spent') or 0)-float(sp.get('sell') or 0))
                features=[float(source==k) for k in KINDS]+[level/4,old,new,
                    float(sp.get('sell') or 0)/1000,loss/1000,price/1000,quality,(new-old)]
                action=23+KINDS.index(target)
                # Weak matchup first; higher levels of the same type are retained.
                rank=(old,loss,-quality,i)
                if action not in out or rank<out[action]['rank']:
                    out[action]=dict(rank=rank,features=features,hid=sp['hid'],kind=target,index=i,spot=sp)
        return out

    def _candidates(self,o):
        candidates=super()._candidates(o)
        if o.get('replacement_pending'):
            return {0:([],('noop',None,{}),0)}
        for action,p in self._replacement_plans(o).items():
            candidates[action]=([('replace',p['hid'],p['kind'])],('replace',p['spot'],{'index':p['index']}),-p['rank'][0])
        return candidates

    def action_kind(self,action):
        return 'replace' if 23<=action<27 else super().action_kind(action)

    def _vec(self,o):
        original=super()._vec(o)['summary']
        plans=self._replacement_plans(o)
        features=[]
        for action in range(23,27):features.extend(plans[action]['features'] if action in plans else [0]*12)
        interval = self._interval(o)
        cooldown = max(0,1-(self.t-self.last_replacement)/interval) if interval else 0
        features += [float(o.get('replacement_pending',False)),cooldown]
        return {'summary':np.concatenate((original,np.asarray(features,dtype=np.float32)))}

    def _start_play(self):
        self.last_replacement=-10**9;self.replaced_wave=-1;self.holder_wave={};self.wave_replacements={}
        return super()._start_play()

    def step(self,action):
        old_result=self.o.get('replacement_result')
        plan=self._replacement_plans(self.o).get(action)
        obs,reward,term,trunc,info=super().step(action)
        if plan and info.get('agent_action_ok'):
            self.last_replacement=self.t
            self.replaced_wave=int(self.prev_o.get('wave') or 0)
            self.wave_replacements[self.replaced_wave]=self.wave_replacements.get(self.replaced_wave,0)+1
            self.holder_wave[plan['hid']]=self.replaced_wave
        result=self.o.get('replacement_result')
        if result and result!=old_result:
            if not result.get('ok'):
                raise RuntimeError('Replacement rebuild failed: '+str(result))
            cost=-min(float(result.get('loss') or 0),500)*.001
            reward+=cost
            self.reward_totals['replacement_loss']+=cost
            info.setdefault('reward_components',{})['replacement_loss']=cost
            if 'ep_stats' in info:info['ep_stats']['reward_parts']['replacement_loss']=self.reward_totals['replacement_loss']
            info['replacement']=result
            self.ep.setdefault('replacements',[]).append(result)
        if 'ep_stats' in info:
            info['ep_stats']['replacements']=list(self.ep.get('replacements',[]))
        info['replacement_limit']=self._quota(self.o)
        info['replacement_used']=self.wave_replacements.get(int(self.o.get('wave') or 0),0)
        return self._vec(self.o),reward,term,trunc,info
