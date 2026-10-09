"""Independent v10 support-tower schema; v9 weights and features stay intact."""
import numpy as np
from gymnasium import spaces
from kr_layout import KRLayoutEnv
from support_features import SupportField, FEATURE_DIM, SCENE_DIM

SCHEMA='candidate-v10'


class KRSupportEnv(KRLayoutEnv):
    schema=SCHEMA
    battlefield_class=SupportField
    feature_dim=FEATURE_DIM

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.config['support_mechanisms']=True
        self.observation_space=spaces.Dict({
            'scene':spaces.Box(-np.inf,np.inf,(SCENE_DIM,),np.float32),
            'candidates':spaces.Box(-np.inf,np.inf,(self.n_actions,FEATURE_DIM),np.float32)})

    def _vec(self,observation):
        if observation.get('support_mechanisms_version')!=1:
            raise RuntimeError('v10 requires native support-unit mechanisms; reload current Lua environment')
        return super()._vec(observation)
