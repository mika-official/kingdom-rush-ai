"""Compact legal-candidate demonstrations, BC initialization and a joint PPO loss.

The PPO update follows the installed sb3-contrib MaskablePPO implementation;
the only objective addition is bc_coef * masked expert cross entropy.
"""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from walkthrough_dataset import iter_successful_shards
from candidate_features import TYPES


class Demonstrations:
    def __init__(self,path,seed=42,expected_schema=None,sampling='balanced',split='native_seed'):
        if split not in ('native_seed','trajectory'):raise ValueError('Unknown demonstration split')
        self.path=Path(path).resolve();self.rng=np.random.default_rng(seed)
        self.sampling=sampling
        self.economic_wait_margin=7. if expected_schema=='candidate-v10' else 0.
        self.success_pool=None;self.success_ratio=0.
        episodes=defaultdict(list)
        episode_groups={}
        digest=hashlib.sha256()
        for episode,shard in iter_successful_shards(self.path,expected_schema):
            key=(episode['dataset_id'],episode['episode'])
            if expected_schema in ('candidate-v9','candidate-v10'):
                native_seed=episode.get('game_seed',episode.get('seed'))
                if native_seed is None:raise ValueError('v9 demonstrations require seed metadata')
                episode_groups[key]=str(native_seed)
            for j,action in enumerate(shard['actions']):
                legal=np.flatnonzero(shard['masks'][j])
                kept=np.union1d(legal,np.flatnonzero(shard['obs_candidates'][j,:,0]>.5))
                label=int(np.flatnonzero(kept==action)[0])
                row=dict(scene=shard['obs_scene'][j].astype(np.float32),
                         candidates=shard['obs_candidates'][j,kept].astype(np.float32),
                         mask=shard['masks'][j,kept].copy(),
                         label=label,active=bool(action!=0),uid=str(shard['candidate_uids'][j]))
                row['type']=TYPES[int(row['candidates'][label,1:1+len(TYPES)].argmax())]
                row['opening']=bool(row['scene'][2]==0 and row['type'] in ('build','upgrade','tower_skill','wait','wave'))
                row['family']=episode.get('family','unknown')
                row['correction']=bool(episode.get('is_correction'))
                row['level']=episode.get('level');row['level_mode']=episode.get('level_mode',1)
                if expected_schema=='candidate-v10':
                    from support_features import F as SF
                    from kr_macro import KINDS
                    from candidate_features import BRANCHES
                    vector=row['candidates'][label]
                    kind=next((name for name in KINDS if vector[SF['kind_'+name]]>.5),'none')
                    branch=next((name for name in BRANCHES if vector[SF['branch_'+name]]>.5),'none')
                    row['investment']=kind+':'+branch
                    if row['type']=='tower_skill':
                        from skill_features import SKILLS
                        purchased=next((name for name in SKILLS if vector[SF[f'skill_{name}_target_level']]>vector[SF[f'skill_{name}_source_level']]+1e-6),'unknown')
                        row['investment']+=':'+purchased
                else:row['investment']='legacy'
                episodes[key].append(row)
                for k in ('scene','candidates'):digest.update(row[k].tobytes())
                digest.update(row['mask'].tobytes())
                digest.update(str((key,label,row['uid'])).encode())
        duplicate_episodes=[]
        if expected_schema in ('candidate-v9','candidate-v10'):
            unique={}
            for key in sorted(episodes):
                fingerprint=hashlib.sha256()
                fingerprint.update(str(episodes[key][0].get('correction',False)).encode())
                for row in episodes[key]:
                    for name in ('scene','candidates'):fingerprint.update(row[name].tobytes())
                    fingerprint.update(row['mask'].tobytes())
                    fingerprint.update(str((row['label'],row['uid'])).encode())
                signature=fingerprint.hexdigest()
                if signature in unique:
                    duplicate_episodes.append(list(key));del episodes[key]
                else:unique[signature]=key
        keys=sorted(episodes)
        opening_profiles={}
        for rows in episodes.values():
            if rows[0].get('correction') or rows[0].get('level') is None:continue
            key=f"{rows[0]['level']}:{rows[0]['level_mode']}"
            profile=opening_profiles.setdefault(key,[])
            profile.append(dict(Counter(r['type'] for r in rows if r['opening'])))
        opening_profiles={key:{'max_builds':max(p.get('build',0) for p in profiles),
                               'min_upgrades':min(p.get('upgrade',0) for p in profiles),
                               'min_skills':min(p.get('tower_skill',0) for p in profiles)}
                          for key,profiles in opening_profiles.items()}
        for key,profile in opening_profiles.items():
            cases=[rows for rows in episodes.values() if not rows[0].get('correction')
                   and f"{rows[0]['level']}:{rows[0]['level_mode']}"==key]
            profile['early_waves']={}
            for wave in (1,2):
                counts=[Counter(r['type'] for r in rows if abs(float(r['scene'][2])*float(r['scene'][3])*20-wave)<.01) for rows in cases]
                profile['early_waves'][str(wave)]=dict(max_builds=max(c.get('build',0) for c in counts),
                    min_upgrades=min(c.get('upgrade',0) for c in counts),min_skills=min(c.get('tower_skill',0) for c in counts))
        if len(keys)<2:raise ValueError('BC requires at least two distinct successful full episodes')
        if expected_schema in ('candidate-v9','candidate-v10') and split=='native_seed':
            seed_groups=sorted({episode_groups[k] for k in keys})
            if len(seed_groups)<2:raise ValueError('BC requires at least two distinct native seeds')
            self.rng.shuffle(seed_groups)
            # Keep every strategy represented in training when grouped seeds
            # permit it; never break a seed group to achieve that coverage.
            all_families={rows[0]['family'] for rows in episodes.values()}
            validation_seeds=set()
            target=max(1,round(len(seed_groups)*.25))
            for native_seed in seed_groups:
                proposed=validation_seeds | {native_seed}
                remaining={episodes[k][0]['family'] for k in keys if episode_groups[k] not in proposed}
                if remaining==all_families:validation_seeds=proposed
                if len(validation_seeds)==target:break
            if not validation_seeds:validation_seeds={seed_groups[0]}
            validation={k for k in keys if episode_groups[k] in validation_seeds}
        else:
            eligible=[k for k in keys if not episodes[k][0].get('correction')]
            self.rng.shuffle(eligible)
            if len(eligible)<2:raise ValueError('Trajectory split requires at least two distinct victories')
            if expected_schema=='candidate-v10':
                # Keep a rare demonstrated skill (e.g. the only Polymorph)
                # available to BC rather than holding its entire source out.
                channels={r['investment'] for rows in episodes.values() for r in rows if r['type'] in ('build','upgrade','tower_skill','replace')}
                validation=set()
                by_map=defaultdict(list)
                for k in eligible:by_map[episodes[k][0]['level']].append(k)
                # Hold out complete trajectories on each map, retaining at
                # least one victory there for training. Single-map behavior
                # keeps its existing rare-skill selection.
                quotas=([(level,min(len(cases)-1,max(1,round(len(cases)*.25))))
                         for level,cases in by_map.items()] if len(by_map)>1
                        else [(None,max(1,round(len(eligible)*.25)))])
                for level in [level for level,count in quotas for _ in range(count)]:
                    choices=[k for k in eligible if k not in validation and
                             (len(by_map)==1 or episodes[k][0]['level']==level)]
                    selected=max(choices,key=lambda key:len({r['investment'] for k,rows in episodes.items() if k not in validation|{key}
                        for r in rows if r['type'] in ('build','upgrade','tower_skill','replace')} & channels))
                    validation.add(selected)
            else:validation=set(eligible[:max(1,round(len(eligible)*.25))])
        self.train=[r for k in keys if k not in validation for r in episodes[k]]
        self.validation=[r for k in keys if k in validation for r in episodes[k]]
        self.active=np.asarray([i for i,r in enumerate(self.train) if r['active'] and not r.get('correction')],dtype=np.int64)
        self.waits=np.asarray([i for i,r in enumerate(self.train) if not r['active'] and not r.get('correction')],dtype=np.int64)
        self.corrections=np.asarray([i for i,r in enumerate(self.train) if r.get('correction')],dtype=np.int64)
        if not len(self.active) or not len(self.waits):raise ValueError('BC requires both active actions and waits')
        self.groups={name:defaultdict(list) for name in ('opening','strategic','tactical','wait')}
        for i,row in enumerate(self.train):
            if row.get('correction'):continue
            group='opening' if row['opening'] else 'strategic' if row['type'] in ('build','upgrade','tower_skill','replace') else 'tactical' if row['active'] else 'wait'
            self.groups[group][row['type']].append(i)
        self.family_pools={g:{t:{family:[i for i in indices if self.train[i]['family']==family]
            for family in sorted({self.train[i]['family'] for i in indices})}
            for t,indices in types.items()} for g,types in self.groups.items()}
        self.investment_pools={t:{family:{investment:[i for i in indices if self.train[i]['investment']==investment]
            for investment in sorted({self.train[i]['investment'] for i in indices})}
            for family,indices in families.items()} for t,families in self.family_pools['strategic'].items()} if expected_schema=='candidate-v10' else {}
        self.map_pools={}
        if expected_schema=='candidate-v10' and len({r['level'] for r in self.train if not r['correction']})>1:
            for group,types in self.groups.items():
                for kind,indices in types.items():
                    for i in indices:
                        row=self.train[i]
                        families=self.map_pools.setdefault(row['level'],{}).setdefault(group,{}).setdefault(kind,{})
                        families.setdefault(row['family'],{}).setdefault(row['investment'] if group=='strategic' else 'all',[]).append(i)
        self.metadata=dict(dataset_sha256=digest.hexdigest(),train_episodes=[list(k) for k in keys if k not in validation],
                           train_maps=dict(Counter(r['level'] for r in self.train if not r['correction'])),
                           validation_maps=dict(Counter(r['level'] for r in self.validation if not r['correction'])),
                           map_sampling='equal maps before strategic sampling; corrections retain their bounded share' if self.map_pools else 'single map',
                           opening_profiles=opening_profiles,
                           validation_episodes=[list(k) for k in keys if k in validation],
                           train_samples=len(self.train),validation_samples=len(self.validation),
                           train_correction_samples=len(self.corrections),
                           validation_correction_samples=sum(r.get('correction',False) for r in self.validation),
                           active_wait_sampling='50/50' if sampling=='balanced' else 'opening25/strategic45/tactical20/wait10; equal types then families within each group',
                           duplicate_episodes_removed=duplicate_episodes,
                           split_unit='native_seed' if expected_schema in ('candidate-v9','candidate-v10') and split=='native_seed' else 'trajectory',
                           investment_sampling='equal investment channels within strategic type and family' if self.investment_pools else 'legacy',
                           economic_wait_margin=self.economic_wait_margin,
                           train_investments=dict(Counter(r['investment'] for r in self.train if r['type'] in ('build','upgrade','tower_skill','replace'))),
                           validation_investments=dict(Counter(r['investment'] for r in self.validation if r['type'] in ('build','upgrade','tower_skill','replace'))),
                           train_seeds=sorted({episode_groups[k] for k in keys if k not in validation}) if episode_groups else [],
                           validation_seeds=sorted({episode_groups[k] for k in validation}) if episode_groups else [],
                           train_families=dict(Counter(r['family'] for r in self.train)),
                           validation_families=dict(Counter(r['family'] for r in self.validation)),
                           train_core_goal_samples=sum(bool((~r['mask']).any()) for r in self.train),
                           validation_core_goal_samples=sum(bool((~r['mask']).any()) for r in self.validation),
                           sampling=sampling,group_counts={g:{t:len(v) for t,v in types.items()} for g,types in self.groups.items()},source=str(self.path))

    def balanced_indices(self,size):
        if self.sampling=='strategic':
            weights=dict(opening=.25,strategic=.45,tactical=.2,wait=.1)
            if self.map_pools:
                indices=[]
                for level in self.rng.choice(list(self.map_pools),size):
                    groups=self.map_pools[level]
                    names=list(groups);probabilities=np.asarray([weights[g] for g in names]);probabilities/=probabilities.sum()
                    group=self.rng.choice(names,p=probabilities)
                    kind=self.rng.choice(list(groups[group]))
                    families=groups[group][kind];family=self.rng.choice(list(families))
                    investments=families[family];investment=self.rng.choice(list(investments))
                    indices.append(self.rng.choice(investments[investment]))
                return np.asarray(indices,dtype=np.int64)
            names=[g for g in weights if self.groups[g]]
            probabilities=np.asarray([weights[g] for g in names]);probabilities/=probabilities.sum()
            indices=[]
            for group in self.rng.choice(names,size,p=probabilities):
                types=list(self.groups[group]);kind=self.rng.choice(types)
                families=self.family_pools.get(group,{}).get(kind)
                if families:
                    family=self.rng.choice(list(families))
                    investments=self.investment_pools.get(kind,{}).get(family) if group=='strategic' else None
                    if investments:
                        investment=self.rng.choice(list(investments));indices.append(self.rng.choice(investments[investment]))
                    else:indices.append(self.rng.choice(families[family]))
                else:indices.append(self.rng.choice(self.groups[group][kind]))
            return np.asarray(indices,dtype=np.int64)
        a=size//2
        indices=np.concatenate((self.rng.choice(self.active,a,replace=True),
                                self.rng.choice(self.waits,size-a,replace=True)))
        self.rng.shuffle(indices)
        return indices

    @staticmethod
    def batch(rows,device):
        count=len(rows);width=max(len(r['candidates']) for r in rows)
        features=rows[0]['candidates'].shape[1]
        candidates=np.zeros((count,width,features),np.float32)
        mask=np.zeros((count,width),bool)
        for i,r in enumerate(rows):
            n=len(r['candidates']);candidates[i,:n]=r['candidates'];mask[i,:n]=r.get('mask',True)
        return ({'scene':th.as_tensor(np.stack([r['scene'] for r in rows]),device=device),
                 'candidates':th.as_tensor(candidates,device=device)},
                th.as_tensor(mask,device=device),th.as_tensor([r['label'] for r in rows],device=device))

    def training_indices(self,size):
        indices=self.balanced_indices(size)
        # Keep corrections bounded: one quarter of BC minibatches explicitly
        # targets visited off-plan states; retain original victory supervision.
        if len(self.corrections):
            indices[:max(1,size//4)]=self.rng.choice(self.corrections,max(1,size//4))
        return indices

    def loss(self,policy,size=32):
        indices=self.training_indices(size)
        rows=[self.train[i] for i in indices]
        pool=getattr(self,'success_pool',None)
        if pool is not None and pool.episodes and self.success_ratio:
            count=min(size//4,int(size*self.success_ratio))
            if count:rows[-count:]=pool.sample(count)
        obs,mask,labels=self.batch(rows,policy.device)
        if hasattr(policy,'augment_routes'):
            obs=policy.augment_routes(obs,inplace=True)
        logits=policy.candidate_logits(obs,mask).masked_fill(~mask,-1e8)
        loss=F.cross_entropy(logits,labels)
        if self.economic_wait_margin:
            # Demonstrated saving is a repeated decision: small aggregate
            # purchase probability accumulates into frequent random sprawl.
            # Train a margin against paid choices ONLY on expert wait states.
            # This changes BC supervision, never legality or runtime actions.
            spending=sum(obs['candidates'][...,1+TYPES.index(t)] for t in ('build','upgrade','tower_skill','replace'))>.5
            spending=spending&mask
            waits=th.as_tensor([r['type']=='wait' and not r.get('self_success') for r in rows],device=policy.device)&spending.any(-1)
            if hasattr(policy,'economic_wait_eligible'):
                waits=waits&policy.economic_wait_eligible(obs,mask)
            if waits.any():
                paid=logits.masked_fill(~spending,-1e8).logsumexp(-1)
                expert=logits.gather(1,labels[:,None]).squeeze(1)
                loss=loss+.25*F.softplus(paid[waits]-expert[waits]+self.economic_wait_margin).mean()
        return loss

    def measure(self,policy,size=32):
        policy.set_training_mode(False)
        result={}
        with th.no_grad():
            for split,rows in (('train',self.train),('validation',self.validation)):
                logp=0.;correct=active_correct=wait_correct=active=waits=0
                counts=defaultdict(lambda:[0,0]);investments=defaultdict(lambda:[0,0]);opening_correct=opening_count=0
                correction_correct=correction_count=0;correction_probability=0.
                for start in range(0,len(rows),size):
                    chunk=rows[start:start+size]
                    obs,mask,labels=self.batch(chunk,policy.device)
                    logits=policy.candidate_logits(obs,mask).masked_fill(~mask,-1e8)
                    logp+=float(F.log_softmax(logits,-1).gather(1,labels[:,None]).sum())
                    matches=(logits.argmax(-1)==labels).cpu().numpy()
                    probabilities=logits.softmax(-1).gather(1,labels[:,None]).squeeze(1).cpu().numpy()
                    flags=np.asarray([r['active'] for r in chunk])
                    correct+=int(matches.sum());active+=int(flags.sum());waits+=int((~flags).sum())
                    active_correct+=int((matches & flags).sum());wait_correct+=int((matches & ~flags).sum())
                    for row,match,probability in zip(chunk,matches,probabilities):
                        counts[row['type']][0]+=int(match);counts[row['type']][1]+=1
                        if row['type'] in ('build','upgrade','tower_skill','replace'):
                            key=row.get('investment','legacy');investments[key][0]+=int(match);investments[key][1]+=1
                        if row['opening']:opening_count+=1;opening_correct+=int(match)
                        if row.get('correction'):
                            correction_count+=1;correction_correct+=int(match);correction_probability+=float(probability)
                result[split]=dict(expert_log_prob=logp/len(rows),accuracy=correct/len(rows),
                                   active_accuracy=active_correct/max(1,active),wait_accuracy=wait_correct/max(1,waits),
                                   balanced_accuracy=.5*(active_correct/max(1,active)+wait_correct/max(1,waits)))
                result[split].update(by_type={t:dict(accuracy=c/n,samples=n) for t,(c,n) in counts.items()},
                    opening_accuracy=opening_correct/max(1,opening_count),opening_samples=opening_count)
                result[split].update(correction_accuracy=correction_correct/max(1,correction_count),correction_samples=correction_count)
                result[split]['correction_expert_probability']=correction_probability/max(1,correction_count)
                result[split]['by_investment']={key:dict(accuracy=c/n,samples=n) for key,(c,n) in investments.items()}
                result[split]['investment_macro_accuracy']=float(np.mean([c/n for c,n in investments.values()])) if investments else 0.
                strategic=[c/n for t,(c,n) in counts.items() if t in ('build','upgrade','tower_skill','replace')]
                result[split]['strategic_macro_accuracy']=float(np.mean(strategic)) if strategic else 0.
        policy.set_training_mode(True)
        return result


def pretrain(model,data,epochs,lr,run,batch_size=32):
    modules={name:getattr(model.policy,name) for name in ('scorer','scene_encoder','layout_encoder','route_encoder') if hasattr(model.policy,name)}
    parameters=[p for module in modules.values() for p in module.parameters()]
    optimizer=th.optim.Adam(parameters,lr=lr)
    history=[];best=float('-inf');best_state=None
    for epoch in range(epochs):
        losses=[]
        model.policy.set_training_mode(True)
        for _ in range(max(1,int(np.ceil(len(data.train)/batch_size)))):
            loss=data.loss(model.policy,batch_size)
            optimizer.zero_grad();loss.backward()
            th.nn.utils.clip_grad_norm_(parameters,model.max_grad_norm)
            optimizer.step();losses.append(float(loss.detach()))
        metrics=data.measure(model.policy,batch_size)
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),**metrics);history.append(row)
        print('BC_PRETRAIN',json.dumps(row),flush=True)
        validation=metrics['validation']
        score=.5*validation['opening_accuracy']+.5*validation['strategic_macro_accuracy'] if data.sampling=='strategic' else validation['balanced_accuracy']
        if data.sampling=='strategic' and data.investment_pools:
            score=.5*validation['opening_accuracy']+.25*validation['strategic_macro_accuracy']+.25*validation['investment_macro_accuracy']
        if validation.get('correction_samples',0):
            score=.75*score+.25*validation['correction_accuracy']
        elif metrics['train'].get('correction_samples',0):
            # With a single fixed seed, corrections intentionally stay in
            # training. Still reject checkpoints that forgot these labelled
            # failures; this is a training safety check, not held-out evidence.
            score=.75*score+.25*metrics['train']['correction_expert_probability']
        if score>best:
            best=score
            best_state={name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for name,module in modules.items()}
        (run/'bc_pretrain.json').write_text(json.dumps(dict(split=data.metadata,epochs=history),indent=2),encoding='utf8')
    if best_state is not None:
        for name,state in best_state.items():modules[name].load_state_dict(state)


class BCPPO(MaskablePPO):
    demo_data=None
    bc_coef=0.0
    bc_batch_size=32

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['demo_data']

    def train(self):
        if self.demo_data is None or self.bc_coef==0:return super().train()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range=self.clip_range(self._current_progress_remaining)
        clip_range_vf=self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        entropy_losses=[];pg_losses=[];value_losses=[];clip_fractions=[];bc_losses=[]
        continue_training=True
        for epoch in range(self.n_epochs):
            approx_kl_divs=[]
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions=rollout_data.actions.long().flatten() if isinstance(self.action_space,spaces.Discrete) else rollout_data.actions
                values,log_prob,entropy=self.policy.evaluate_actions(rollout_data.observations,actions,
                                                                     action_masks=rollout_data.action_masks)
                values=values.flatten();advantages=rollout_data.advantages
                if self.normalize_advantage:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio=th.exp(log_prob-rollout_data.old_log_prob)
                policy_loss=-th.min(advantages*ratio,advantages*th.clamp(ratio,1-clip_range,1+clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio-1)>clip_range).float()).item())
                values_pred=values if clip_range_vf is None else rollout_data.old_values+th.clamp(
                    values-rollout_data.old_values,-clip_range_vf,clip_range_vf)
                value_loss=F.mse_loss(rollout_data.returns,values_pred);value_losses.append(value_loss.item())
                entropy_loss=-th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                with th.no_grad():
                    log_ratio=log_prob-rollout_data.old_log_prob
                    kl=th.mean((th.exp(log_ratio)-1)-log_ratio).cpu().numpy();approx_kl_divs.append(kl)
                if self.target_kl is not None and kl>1.5*self.target_kl:
                    continue_training=False
                    if self.verbose:print(f'Early stopping at step {epoch} due to reaching max kl: {kl:.2f}')
                    break
                bc_loss=self.demo_data.loss(self.policy,self.bc_batch_size);bc_losses.append(float(bc_loss.detach()))
                loss=policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss+self.bc_coef*bc_loss
                self.policy.optimizer.zero_grad();loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates+=1
            if not continue_training:break
        for name,values in (('entropy_loss',entropy_losses),('policy_gradient_loss',pg_losses),('value_loss',value_losses),
                            ('approx_kl',approx_kl_divs),('clip_fraction',clip_fractions),('bc_loss',bc_losses)):
            self.logger.record('train/'+name,float(np.mean(values)) if values else 0.)
        self.logger.record('train/bc_coef',self.bc_coef)
        self.logger.record('train/loss',loss.item() if bc_losses else 0.)
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip_range)
        if clip_range_vf is not None:self.logger.record('train/clip_range_vf',clip_range_vf)
