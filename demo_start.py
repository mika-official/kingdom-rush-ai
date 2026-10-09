"""Optional curriculum: replay verified native prefixes, then hand control to PPO."""
import json
from pathlib import Path
import gymnasium as gym


def successful_prefixes(path):
    path=Path(path);manifest=json.loads(path.read_text(encoding='utf8'));cases=[]
    if manifest.get('format') in ('v9-walkthrough-catalog-v1','v10-walkthrough-catalog-v1'):
        for entry in manifest['datasets']:
            target=Path(entry['manifest'])
            cases.extend(successful_prefixes(target if target.is_absolute() else path.parent/target))
        return cases
    if manifest.get('format') not in ('v9-walkthrough-shards-v1','v10-walkthrough-shards-v1'):return []
    for episode in manifest['episodes']:
        if not episode.get('expert_eligible') or episode.get('outcome')!='victory' or episode.get('rejected_actions',0):continue
        first=Path(episode['shards'][0]);folder=(first if first.is_absolute() else path.parent/first).parent
        actions=[json.loads(s) for s in (folder/'actions.jsonl').read_text(encoding='utf8').splitlines()]
        cases.append(dict(level=manifest['level'],mode=1,seed=episode['seed'],game_seed=episode['game_seed'],
                          ticks_per_step=manifest['config']['ticks_per_step'],
                          difficulty=manifest['config']['difficulty'],loadout=manifest['config']['loadout'],
                          actions={r['step']:r for r in actions if r.get('action')},source=str(folder)))
    return cases


class DemonstrationStart(gym.Wrapper):
    def __init__(self,env,dataset,probability=.5,min_wave=4,max_wave=10):
        super().__init__(env)
        self.cases=successful_prefixes(dataset)
        self.probability=probability;self.min_wave=min_wave;self.max_wave=max_wave
        self.start_info=None
        if not self.cases:raise ValueError('Demonstration starts require verified native victories')

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);base=self.env.unwrapped;self.start_info=None
        cases=[c for c in self.cases if (c['level'],c['mode'])==base.selected_pair
               and c['difficulty']==base.difficulty and c['loadout']==base.loadout
               and c['ticks_per_step']==base.ticks_per_step
               and (base.fixed_game_seed is None or c['game_seed']==base.fixed_game_seed)]
        if not cases or base.np_random.random()>=self.probability:return obs,info
        case=cases[int(base.np_random.integers(len(cases)))]
        max_wave=min(self.max_wave,int(base.o.get('wave_total') or self.max_wave))
        min_wave=min(self.min_wave,max_wave)
        wave=int(base.np_random.integers(min_wave,max_wave+1))
        schedule_rng_state=base.np_random.bit_generator.state
        saved_seed=base.fixed_game_seed
        try:
            base.fixed_game_seed=case['game_seed']
            obs,info=self.env.reset(seed=case['seed'],options={'level_mode':base.selected_pair})
            base.np_random.bit_generator.state=schedule_rng_state
        finally:base.fixed_game_seed=saved_seed
        step=0
        while int(base.o.get('wave') or 0)<wave:
            row=case['actions'].get(step);action=0
            if row is not None:
                mask=base.action_masks()
                matches=[i for i,c in enumerate(base._entries(base.o)) if c['uid']==row['uid'] and mask[i]]
                if not matches:raise ValueError(f'Demonstration prefix diverged at step {step}: {row["uid"]}')
                action=matches[0]
            obs,_,term,trunc,info=self.env.step(action);step+=1
            if action and not info.get('agent_action_ok'):raise ValueError('Native prefix action rejected')
            if term or trunc:raise ValueError('Demonstration prefix ended before target wave')
            if step>10000:raise ValueError('Demonstration prefix exceeded step limit')
        self.start_info=dict(wave=wave,prefix_steps=step,source=case['source'],game_seed=case['game_seed'])
        info['demonstration_start']=self.start_info
        return obs,info

    def step(self,action):
        obs,reward,term,trunc,info=self.env.step(action)
        if self.start_info is not None:info['demonstration_start']=self.start_info
        if self.start_info is not None and 'ep_stats' in info:
            info['ep_stats']['demonstration_start']=self.start_info
        return obs,reward,term,trunc,info

    def action_masks(self):return self.env.unwrapped.action_masks()

    def snapshot_hash(self):return self.env.unwrapped.snapshot_hash()
