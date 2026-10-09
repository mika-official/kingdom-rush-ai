"""Inspectable linear candidate scoring; no imitation reward or PPO teacher."""
import numpy as np
from candidate_features import F, FEATURE_NAMES

WEIGHTS={
    'missing_physical':.7,'missing_magic':1.2,'missing_area':1.2,'missing_air':1.5,
    'delta_physical':.4,'delta_magic':.6,'delta_area':.6,'delta_air':.3,
    'blocking_output_synergy':.6,'delta_control':.25,'repeat_output':-.5,
    'coverage':.7,'future_count':.15,'current_hp':.2,'special_count':.1,
    'net_cost':-.15,'loss':-.7,'reserve_risk':-.5,'reserve_fraction':.03,
    'type_replace':-.25,'type_wait':-.05,'type_wave':.08,
    'type_hero_move':-.2,'type_rally':-.1,'type_rain':-.25,'cooldown_cost':-.4,
    'covered_unprotected_hp':.3,'current_density':.3,'type_tower_skill':.05}
WEIGHTS.update(benefit_physical=1.0,benefit_magic=1.5,benefit_area=1.5,benefit_air=2,
               power_urgency=.8,cooldown_future_overlap=-.7)


class CandidateBaseline:
    num_timesteps=0
    def __init__(self,env,weights=None):
        self.env=env.unwrapped
        self.weights=np.zeros(len(FEATURE_NAMES),dtype=np.float32)
        for name,value in (WEIGHTS if weights is None else weights).items(): self.weights[F[name]]=value
    def scores(self,obs): return np.asarray(obs['candidates'])@self.weights
    def predict(self,obs,action_masks,deterministic=True):
        scores=self.scores(obs)
        scores=np.where(action_masks,scores,-np.inf)
        return np.int64(np.argmax(scores)),None
