"""Native victory archives and bounded replay for continuous BC.

Capture existing observations only: no feature recomputation or combat search.
Archives contain every policy-controlled transition, including waits. RAM replay
subsamples waits, keeps episode provenance, and never changes validation data.
"""
import hashlib
import json
from pathlib import Path
import uuid
from collections import Counter,defaultdict
import gymnasium as gym
import numpy as np
from candidate_features import TYPES


class VictoryRecorder(gym.Wrapper):
    def __init__(self,env,directory,source='training',chunk_size=64,allowed_levels=None):
        super().__init__(env)
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.source=source;self.chunk_size=chunk_size;self.allowed_levels=allowed_levels
        self.folder=None;self.rows=[];self.files=[];self.completed=[];self.trace=None
        self.obs=None;self.step_index=0;self.digest=None;self.invalid=False

    def _discard(self):
        if self.trace:self.trace.close();self.trace=None
        if self.folder:
            assert self.folder.resolve().parent==self.directory
            for path in self.folder.iterdir():
                assert path.resolve().parent==self.folder.resolve()
                if path.is_file():path.unlink()
            self.folder.rmdir()
        self.folder=None;self.rows=[];self.files=[]

    def reset(self,**kwargs):
        self._discard()
        obs,info=self.env.reset(**kwargs);base=self.unwrapped
        self.obs=obs;self.step_index=0;self.invalid=False;self.digest=hashlib.sha256()
        self.start=dict(level=base.selected_pair[0],level_mode=base.selected_pair[1],
            seed=kwargs.get('seed'),game_seed=base.game_seed,
            difficulty=base.difficulty,loadout=base.loadout,ticks_per_step=base.ticks_per_step,
            schema=base.schema,source=self.source,
            start_kind='demo_prefix' if info.get('demonstration_start') else 'cold',
            demonstration_start=info.get('demonstration_start'),start_wave=int(base.o.get('wave') or 0))
        if self.allowed_levels is None or self.start['level'] in self.allowed_levels:
            self.folder=self.directory/('.pending_'+uuid.uuid4().hex);self.folder.mkdir()
            self.trace=(self.folder/'actions.jsonl').open('w',encoding='utf8')
            identity={k:self.start[k] for k in ('level','level_mode','game_seed','difficulty','loadout','start_kind','start_wave')}
            self.digest.update(json.dumps(identity,sort_keys=True).encode())
        return obs,info

    def _append(self,mask,action,uid):
        kept=np.flatnonzero(mask|(self.obs['candidates'][:,0]>.5))
        if not 0<=action<len(mask) or not mask[action]:raise ValueError('Recorded action is illegal')
        scene=self.obs['scene'].copy();candidates=self.obs['candidates'][kept].copy();legal=mask[kept].copy()
        if not np.isfinite(scene).all() or not np.isfinite(candidates).all():raise ValueError('Nonfinite victory frame')
        label=int(np.flatnonzero(kept==action)[0])
        self.rows.append((scene,candidates,legal,label,self.step_index,uid))
        for value in (scene,candidates,legal):self.digest.update(value.tobytes())
        self.digest.update(str((label,uid)).encode())
        if len(self.rows)>=self.chunk_size:self._flush()

    def _flush(self):
        if not self.rows:return
        width=max(len(r[1]) for r in self.rows);n=len(self.rows);dim=self.rows[0][1].shape[1]
        candidates=np.zeros((n,width,dim),np.float32);masks=np.zeros((n,width),bool)
        for i,r in enumerate(self.rows):candidates[i,:len(r[1])]=r[1];masks[i,:len(r[2])]=r[2]
        filename=f'shard_{len(self.files):04d}.npz'
        # Uncompressed writes avoid spending scarce worker CPU on compression.
        np.savez(self.folder/filename,obs_scene=np.stack([r[0] for r in self.rows]),
            obs_candidates=candidates,masks=masks,actions=np.array([r[3] for r in self.rows]),
            steps=np.array([r[4] for r in self.rows]),candidate_uids=np.array([r[5] for r in self.rows]))
        self.files.append(filename);self.rows=[]

    def step(self,action):
        base=self.unwrapped;action=int(action)
        if self.folder and not self.invalid:
            mask=base.action_masks().copy();entries=base._entries(base.o)
            entry=entries[action] if 0<=action<len(entries) else None
            if entry is None:self.invalid=True
            else:
                try:self._append(mask,action,entry['uid'])
                except ValueError:self.invalid=True
                self.trace.write(json.dumps(dict(step=self.step_index,tick=base.t,wave=base.o.get('wave'),
                    uid=entry['uid'],action=action,commands=entry.get('commands',[]),type=entry['type']))+'\n')
        obs,reward,term,trunc,info=self.env.step(action)
        self.obs=obs;self.step_index+=1
        if action and not info.get('agent_action_ok'):self.invalid=True
        if term or trunc:
            if self.folder and not self.invalid and not trunc and info.get('outcome')=='victory':
                self._flush();self.trace.close();self.trace=None
                folder=self.directory/('victory_'+uuid.uuid4().hex)
                assert folder.parent==self.folder.parent==self.directory
                self.folder.rename(folder);self.folder=None
                episode=dict(episode=0,outcome='victory',expert_eligible=True,rejected_actions=0,
                    family='self_success',samples=self.step_index,trajectory_sha256=self.digest.hexdigest(),
                    lives=info.get('lives'),wave=info.get('wave'),**self.start,
                    shards=self.files)
                manifest=dict(format=base.schema.replace('candidate-','')+'-success-replay-v1',schema=base.schema,
                    level=self.start['level'],level_mode=self.start['level_mode'],scene_dim=obs['scene'].shape[0],
                    candidate_shape=list(obs['candidates'].shape),shard_layout='compact',episodes=[episode])
                path=folder/'manifest.json';temporary=folder/'manifest.json.tmp'
                temporary.write_text(json.dumps(manifest,indent=2),encoding='utf8');temporary.replace(path)
                self.completed.append(str(path));info['success_replay_manifest']=str(path)
            else:self._discard()
        return obs,reward,term,trunc,info

    def drain_successes(self):
        paths=self.completed;self.completed=[];return paths

    def action_masks(self):return self.unwrapped.action_masks()
    def snapshot_hash(self):return self.unwrapped.snapshot_hash()
    def close(self):
        self._discard();return self.env.close()


class VictoryPool:
    def __init__(self,schema,directory,levels,difficulty,loadout,fixed_seed=None,max_episodes=8,max_rows=2048,seed=42,max_bytes=128*1024*1024):
        self.schema=schema;self.directory=Path(directory).resolve();self.levels=set(levels)
        self.difficulty=difficulty;self.loadout=loadout;self.fixed_seed=fixed_seed
        # One reserved slot per configured map, within the same row/byte budget.
        self.max_episodes=max(max_episodes,len(self.levels));self.max_rows=max_rows;self.max_bytes=max_bytes;self.rng=np.random.default_rng(seed)
        if min(max_episodes,max_bytes)<1 or max_rows<self.max_episodes:
            raise ValueError('Replay budgets must allow at least one row per reserved episode')
        self.episodes=[];self.seen=set();self.signatures=set();self.accepted=[]
        self._map_queue=[]

    def add(self,path):
        from walkthrough_dataset import iter_successful_shards
        path=Path(path).resolve()
        if str(path) in self.seen:return False
        manifest=json.loads(path.read_text(encoding='utf8'));episode=manifest['episodes'][0]
        if (manifest.get('schema')!=self.schema or episode.get('level') not in self.levels
                or episode.get('difficulty')!=self.difficulty or episode.get('loadout')!=self.loadout
                or (self.fixed_seed is not None and episode.get('game_seed')!=self.fixed_seed)):
            return False
        self.seen.add(str(path));signature=(episode['level'],episode['trajectory_sha256'])
        if signature in self.signatures:return False
        rows=[];seen_rows=0;row_cap=max(1,self.max_rows//self.max_episodes)
        byte_cap=max(1,self.max_bytes//self.max_episodes)
        for _,shard in iter_successful_shards(path,self.schema):
            for j,action in enumerate(shard['actions']):
                vector=shard['obs_candidates'][j,action]
                typ=TYPES[int(vector[1:1+len(TYPES)].argmax())]
                wave=float(shard['obs_scene'][j,2])*float(shard['obs_scene'][j,3])*20
                if typ=='wait' and wave>2 and int(shard['steps'][j])%8:continue
                kept=np.flatnonzero(shard['masks'][j]|(shard['obs_candidates'][j,:,0]>.5))
                row=dict(scene=shard['obs_scene'][j].copy(),candidates=shard['obs_candidates'][j,kept].copy(),
                    mask=shard['masks'][j,kept].copy(),label=int(np.flatnonzero(kept==action)[0]),type=typ,self_success=True,level=episode['level'])
                seen_rows+=1
                if len(rows)<row_cap:rows.append(row)
                else:
                    slot=int(self.rng.integers(seen_rows))
                    if slot<row_cap:rows[slot]=row
                while sum(self._bytes(r) for r in rows)>byte_cap and len(rows)>1:
                    rows.pop(int(self.rng.integers(len(rows))))
        if not rows:return False
        self.signatures.add(signature);self.accepted.append(str(path))
        old_maps={e['level'] for e in self.episodes}
        self.episodes.append(dict(path=str(path),rows=rows,start_kind=episode['start_kind'],level=episode['level']))
        while len(self.episodes)>self.max_episodes:
            counts=Counter(e['level'] for e in self.episodes)
            largest=max(counts.values())
            # Remove the oldest episode from the most represented map. Never
            # evict a map's final victory to make room for another map's win.
            index=next(i for i,e in enumerate(self.episodes) if counts[e['level']]==largest and largest>1)
            self.episodes.pop(index)
        if old_maps!={e['level'] for e in self.episodes}:self._map_queue=[]
        # Equal per-episode caps keep memory bounded and multiple layouts alive.
        cap=max(1,self.max_rows//len(self.episodes))
        for case in self.episodes:
            if len(case['rows'])>cap:
                active=[i for i,r in enumerate(case['rows']) if r['type']!='wait']
                waits=[i for i,r in enumerate(case['rows']) if r['type']=='wait']
                count=min(len(active),cap//2);chosen=list(self.rng.choice(active,count,replace=False)) if count else []
                chosen_set=set(chosen);rest=[i for i in range(len(case['rows'])) if i not in chosen_set]
                chosen+=list(self.rng.choice(rest,cap-len(chosen),replace=False))
                case['rows']=[case['rows'][i] for i in sorted(chosen)]
        return True

    def scan(self):
        count=0
        for path in sorted(self.directory.glob('victory_*/manifest.json'),key=lambda p:(p.stat().st_mtime_ns,str(p))):
            count+=self.add(path)
        return count

    def sample(self,size):
        if not self.episodes:return []
        by_map=defaultdict(list)
        for case in self.episodes:by_map[case['level']].append(case)
        result=[]
        for _ in range(size):
            if not self._map_queue:self._map_queue=list(self.rng.permutation(sorted(by_map)))
            level=self._map_queue.pop()
            cases=by_map[level];case=cases[int(self.rng.integers(len(cases)))]
            active=[r for r in case['rows'] if r['type']!='wait'];waits=[r for r in case['rows'] if r['type']=='wait']
            rows=(active if self.rng.random()<.5 else waits) or case['rows']
            result.append(rows[int(self.rng.integers(len(rows)))])
        return result

    def summary(self):
        return dict(archived_episodes=len(self.accepted),memory_episodes=len(self.episodes),
            active_episodes_by_map=dict(Counter(e['level'] for e in self.episodes)),
            active_rows_by_map={level:sum(len(e['rows']) for e in self.episodes if e['level']==level) for level in sorted(self.levels)},
            missing_training_maps=sorted(self.levels-{e['level'] for e in self.episodes}),
            active_manifests=[e['path'] for e in self.episodes],
            replay_sampling='shuffled map cycles, then uniform episode and active/wait',
            memory_rows=sum(len(e['rows']) for e in self.episodes),
            memory_bytes=sum(self._bytes(r) for e in self.episodes for r in e['rows']),
            cold_episodes=sum(e['start_kind']=='cold' for e in self.episodes),
            prefix_episodes=sum(e['start_kind']=='demo_prefix' for e in self.episodes),manifests=self.accepted)

    @staticmethod
    def _bytes(row):return row['scene'].nbytes+row['candidates'].nbytes+row['mask'].nbytes
