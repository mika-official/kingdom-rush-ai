"""Independent v7 schema: per-route handling, regeneration and branch budgets."""
import numpy as np
from gymnasium import spaces
from kr_planning import KRPlanningEnv
from route_features import RouteField,FEATURE_DIM,SCENE_DIM
from candidate_features import as_list
from route_features import FEATURE_NAMES

SCHEMA = 'candidate-v7'
ROUTE_PLANNING_REVISION = 4


class KRRoutePlanningEnv(KRPlanningEnv):
    schema = SCHEMA
    battlefield_class = RouteField
    feature_dim = FEATURE_DIM

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config['route_planning'] = True
        self.config['skill_mechanisms'] = True
        self.observation_space = spaces.Dict({
            'scene':spaces.Box(-np.inf,np.inf,(SCENE_DIM,),np.float32),
            'candidates':spaces.Box(-np.inf,np.inf,(self.n_actions,FEATURE_DIM),np.float32)})

    def _vec(self, observation):
        if not observation.get('route_geometry'):
            raise RuntimeError('v7 requires route geometry; reload the current Lua environment')
        if observation.get('skill_mechanisms_version')!=1:
            raise RuntimeError('v7 revision 4 requires native skill mechanisms; reload the current Lua environment')
        return super()._vec(observation)

    def step(self, action):
        old = self.o
        old_tick=self.t
        entries = self._entries(old)
        selected = entries[action] if 0<=action<len(entries) else None
        old_field=self._feature_field
        result = super().step(action)
        obs,reward,term,trunc,info = result
        towers=[t for t in as_list(old.get('spots')) if t.get('state')=='tower']
        seconds=max(0,self.t-old_tick)*self.tick_seconds
        stats=self.ep.setdefault('route_development',dict(tower_seconds=0.0,tier1_tower_seconds=0.0,tier4_tower_seconds=0.0))
        stats['tower_seconds']+=len(towers)*seconds
        stats['tier1_tower_seconds']+=sum(self.effective_level(t)==1 for t in towers)*seconds
        stats['tier4_tower_seconds']+=sum(self.effective_level(t)>=4 for t in towers)*seconds
        now=[t for t in as_list(self.o.get('spots')) if t.get('state')=='tower']
        stats['tier1_fraction_end']=sum(self.effective_level(t)==1 for t in now)/max(1,len(now))
        stats['tier1_fraction_time']=stats['tier1_tower_seconds']/max(1,stats['tower_seconds'])
        leaked=[int(w) for w,count in self.ep.get('leaks_by_wave',{}).items() if count>0]
        stats['first_leak_wave']=min(leaked) if leaked else None
        stats['collapse_wave']=info.get('wave') if info.get('outcome')=='defeat' else None
        if 'ep_stats' in info:
            info['ep_stats']['route_development']=dict(stats)
        if selected and info.get('candidate_event'):
            field = old_field if old_field is not None and old_field.o is old else RouteField(self,old)
            prediction = {}
            for horizon in ('current','next','second'):
                before,after = field.route_comparison(selected,horizon)
                prediction[horizon] = {str(r):dict(margin_before=before[r]['margin'],margin_after=row['margin'],
                    leaks_before=before[r]['leaks'],leaks_after=row['leaks'],life_after=row['life'],
                    count=row['count'],leak_fraction_before=before[r]['leak_fraction'],
                    leak_fraction_after=row['leak_fraction'],life_fraction_after=row['life_fraction'],
                    worst_shortfall_after=row['worst_shortfall'],
                    crosses_threshold=not before[r]['handles'] and row['handles'],partial=row['partial'])
                    for r,row in after.items() if row['present']}
            info['candidate_event']['route_predictions'] = prediction
            event=info['candidate_event']
            event['skill_values']={name:value for name,value in zip(FEATURE_NAMES,event.get('features',[]))
                if name.startswith('skill_') and value!=0}
            event['skill_mechanisms_before']=((selected.get('spot') or {}).get('combat') or {}).get('mechanisms',[])
            event['skill_mechanisms_after']=(field.placed(selected).get('combat') or {}).get('mechanisms',[]) if selected['type'] in ('build','upgrade','tower_skill','replace') else event['skill_mechanisms_before']
            info['candidate_event']['route_prediction_waves']=dict(current=old.get('wave'),
                next=(old.get('forecast') or {}).get('index'),second=(old.get('forecast_second') or {}).get('index'))
        return obs,reward,term,trunc,info
