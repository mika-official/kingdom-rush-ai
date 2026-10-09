"""v8 local facts, native skill mechanisms, no strategic observation estimator."""
import numpy as np
from gymnasium import spaces
from kr_planning import KRPlanningEnv
from light_features import LightField,FEATURE_DIM,SCENE_DIM

SCHEMA='candidate-v8'


class KRLightEnv(KRPlanningEnv):
    schema=SCHEMA
    battlefield_class=LightField
    feature_dim=FEATURE_DIM

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.config.update(light_observations=True,skill_mechanisms=True,route_planning=False,support_mechanisms=False)
        self.observation_space=spaces.Dict({'scene':spaces.Box(-np.inf,np.inf,(SCENE_DIM,),np.float32),
            'candidates':spaces.Box(-np.inf,np.inf,(self.n_actions,FEATURE_DIM),np.float32)})

    def _vec(self,observation):
        if observation.get('light_observations_version')!=1 or observation.get('skill_mechanisms_version')!=1:
            raise RuntimeError('v8 requires native lightweight observations and skill mechanisms; reload current Lua environment')
        return super()._vec(observation)
