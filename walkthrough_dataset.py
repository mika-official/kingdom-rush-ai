"""Read verified victories and explicitly scoped, separately marked corrections."""
import json
from pathlib import Path
import numpy as np


def iter_successful_shards(manifest_path, expected_schema=None):
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('format') in ('v8-walkthrough-catalog-v1','v9-walkthrough-catalog-v1','v10-walkthrough-catalog-v1'):
        for entry in manifest['datasets']:
            path=Path(entry['manifest'])
            if not path.is_absolute():path=manifest_path.parent/path
            yield from iter_successful_shards(path,expected_schema)
        return
    schema=manifest.get('schema')
    replay=schema in ('candidate-v9','candidate-v10') and manifest.get('format')==schema.replace('candidate-','')+'-success-replay-v1'
    correction=schema in ('candidate-v9','candidate-v10') and manifest.get('format')==schema.replace('candidate-','')+'-correction-shards-v1'
    if schema not in ('candidate-v8','candidate-v9','candidate-v10') or (not correction and not replay and manifest.get('format')!=schema.replace('candidate-','')+'-walkthrough-shards-v1'):
        raise ValueError('Expected versioned walkthrough dataset')
    if expected_schema is not None and schema!=expected_schema:
        raise ValueError('Demonstration schema does not match training schema')
    for episode in manifest['episodes']:
        if replay and (episode.get('start_kind') not in ('cold','demo_prefix') or not episode.get('trajectory_sha256')):
            raise ValueError('Success replay lacks origin or trajectory identity')
        if correction:
            if (not episode.get('correction_eligible') or episode.get('outcome')!='prefix_end'
                    or episode.get('label_rule') not in ('early_archer_concentration_v1','early_archer_concentration_v2')
                    or episode.get('label_max_wave')!=2 or episode.get('rejected_actions',0)!=0):continue
        elif (not episode.get('expert_eligible') or episode.get('outcome')!='victory'
                or episode.get('rejected_actions',0)!=0):continue
        for filename in episode['shards']:
            path=Path(filename)
            if not path.is_absolute():path=manifest_path.parent/path
            with np.load(path,allow_pickle=False) as archive:
                shard={key:archive[key].copy() for key in archive.files}
            actions=shard['actions'];masks=shard['masks'];n=len(actions)
            if correction:
                waves=shard['obs_scene'][:,2]*shard['obs_scene'][:,3]*20
                min_wave=0 if episode['label_rule']=='early_archer_concentration_v2' else .5
                if np.any(waves<min_wave) or np.any(waves>2.01):
                    raise ValueError('Early corrections exceed their labelled wave scope')
            width=masks.shape[1] if masks.ndim==2 else 0
            capacity=manifest['candidate_shape'][0]
            valid_width=0<width<=capacity if replay else width==capacity
            if (not valid_width or masks.shape!=(n,width) or
                shard['obs_candidates'].shape!=(n,width,manifest['candidate_shape'][1]) or
                shard['obs_scene'].shape!=(n,manifest['scene_dim']) or
                np.any(actions<0) or np.any(actions>=masks.shape[1]) or
                not masks[np.arange(n),actions].all() or
                any(not np.isfinite(shard[k]).all() for k in ('obs_scene','obs_candidates'))):
                raise ValueError(f'Invalid demonstration shard: {path}')
            yield dict(episode,dataset_id=str(manifest_path.resolve()),is_correction=correction,
                       level=manifest.get('level'),level_mode=manifest.get('level_mode',1)),shard
