"""Independent v9 schema; v8 checkpoints and demonstrations remain unchanged."""
import numpy as np
from gymnasium import spaces
from kr_light import KRLightEnv
from layout_features import LayoutField, FEATURE_DIM, SCENE_DIM
from candidate_features import as_list

SCHEMA = 'candidate-v9'


class KRLayoutEnv(KRLightEnv):
    schema = SCHEMA
    battlefield_class = LayoutField
    feature_dim = FEATURE_DIM

    def __init__(self, core_investment=False, **kwargs):
        self.core_investment=bool(core_investment)
        self.core_goal_hid=None
        super().__init__(**kwargs)
        self.config['core_investment']=self.core_investment
        self.observation_space = spaces.Dict({
            'scene': spaces.Box(-np.inf, np.inf, (SCENE_DIM,), np.float32),
            'candidates': spaces.Box(-np.inf, np.inf, (self.n_actions, FEATURE_DIM), np.float32)})

    def reset(self,**kwargs):
        self.core_goal_hid=None
        return super().reset(**kwargs)

    def _core_option(self,spot):
        if spot.get('state')!='tower' or spot.get('blocked') or self.tower_kind(spot)!='engineer' or self.effective_level(spot)>=4:
            return None
        choices=[op for op in as_list(spot.get('options')) if not op.get('locked') and op.get('price') is not None
                 and op.get('tpl') in ('tower_engineer_2','tower_engineer_3','tower_tesla')]
        return choices[0] if choices else None

    def _generate(self,o):
        entries=super()._generate(o)
        if not self.core_investment or o.get('outcome') or o.get('replacement_pending'):return entries
        spots=as_list(o.get('spots'))
        spot=next((s for s in spots if s.get('hid')==self.core_goal_hid),None)
        if spot and spot.get('state')=='building':return entries
        option=self._core_option(spot) if spot else None
        if option is None:
            self.core_goal_hid=None
            # Aggregate future pressure once, not once per tower or candidate.
            pressure={}
            for enemy in as_list((o.get('forecast') or {}).get('spawns')):
                route=str(enemy.get('pi'))
                pressure[route]=pressure.get(route,0)+float(enemy.get('count') or 0)*float(enemy.get('hp') or 0)
            total=max(1.,sum(pressure.values()))
            choices=[]
            for s in spots:
                op=self._core_option(s)
                if op is None:continue
                paths=op.get('target_paths') or s.get('defense_paths') or {}
                if not isinstance(paths,dict):continue
                rear=sum(sum(float(v or 0) for v in as_list(bins)[1:3]) for bins in paths.values())/100
                coverage=max(min(1.,rear),min(1.,max(0,len(paths)-1)))
                if coverage<=0:continue
                share=sum(pressure.get(str(r),0) for r in paths)/total
                rank=coverage*(.5+.5*share)*(1+.2*self.effective_level(s))
                choices.append((rank,-int(s['hid']),s,op))
            if not choices:return entries
            _,_,spot,option=max(choices,key=lambda row:row[:2])
            self.core_goal_hid=spot['hid']
        # One read-only goal row, including unaffordable upgrades. It is never
        # executable. Its normal v9 features expose price, location and grade.
        if len(entries)>=self.base_meta:raise RuntimeError('No candidate capacity for core investment goal')
        self._offer(entries,'upgrade',[('upgrade',spot['hid'],option['tpl'])],spot,
            kind='engineer',branch=option['tpl'],cost=option['price'],grade=min(4,self.effective_level(spot)+1),
            target_range=option.get('target_range'),target_paths=option.get('target_paths'),combat=option.get('combat'),
            advisory=True)
        entries[-1]['uid']='core_goal:'+str(spot['hid'])+':'+option['tpl']
        return entries

    def _candidates(self,o):
        choices=super()._candidates(o)
        for i,candidate in enumerate(self._entries(o)):
            if candidate.get('advisory'):choices.pop(i,None)
        return choices

    def _info(self,o):
        info=super()._info(o)
        if self.core_investment and self.phase!='meta':
            goal=next((c for c in self._entries(o) if c.get('advisory')),None)
            info['core_investment']=None if goal is None else dict(hid=self.core_goal_hid,
                target=goal['branch'],cost=goal['cost'],shortfall=max(0,goal['cost']-float(o.get('gold') or 0)))
        return info
