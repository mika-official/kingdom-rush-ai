"""v10 support-unit facts appended to the unchanged v9 feature prefix.

No battle simulation, marginal-damage estimator or per-map rules.
"""
import numpy as np
from layout_features import LayoutField, FEATURE_NAMES as BASE_FEATURES, SCENE_NAMES, SCENE_DIM, ROUTES, HORIZONS
from light_features import mapping
from candidate_features import as_list

ELEMENTAL_SCALES=dict(hp_max=1000,armor=1,damage_min=100,damage_max=100,cooldown=5,
    radius=250,max_targets=10,respawn_seconds=30,regen_amount=100,regen_interval=5,regen_delay=5,block_count=3)
FEATURE_NAMES=BASE_FEATURES+tuple(f'skill_elemental_{stage}_{name}' for stage in ('source','target') for name in ELEMENTAL_SCALES)
FEATURE_DIM=len(FEATURE_NAMES)
F={name:i for i,name in enumerate(FEATURE_NAMES)}


class SupportField(LayoutField):
    def vector(self,candidate):
        base=super().vector(candidate)
        spot=mapping(candidate.get('spot'));source=mapping(spot.get('combat'))
        changing=candidate['type'] in ('build','upgrade','tower_skill','replace')
        target=(mapping(candidate.get('combat')) or source) if changing else source
        values=[]
        for profile in (source,target):
            elemental=next((m for m in as_list(profile.get('mechanisms')) if m.get('name')=='elemental'),{})
            enabled=float(elemental.get('level') or 0)>0
            values.extend(float(elemental.get(name) or 0)/scale if enabled else 0. for name,scale in ELEMENTAL_SCALES.items())
        return np.concatenate((base,np.asarray(values,np.float32)))
