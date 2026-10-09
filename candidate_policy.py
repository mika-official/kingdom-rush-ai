"""Permutation-equivariant shared candidate scorer for MaskablePPO."""
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class SceneExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,observation_space['scene'].shape[0])
    def forward(self,observations): return observations['scene'].float()


def mlp(dim,widths,out):
    layers=[]
    for width in widths:
        layers.extend([nn.Linear(dim,width),nn.Tanh()]);dim=width
    layers.append(nn.Linear(dim,out))
    return nn.Sequential(*layers)


class CandidatePolicy(MaskableActorCriticPolicy):
    def __init__(self,*args,**kwargs):
        # Old checkpoints omit this argument and retain their original distribution.
        self.type_balance_alpha=float(kwargs.pop('type_balance_alpha',0.0))
        self.advanced_tower_bias=float(kwargs.pop('advanced_tower_bias',0.0))
        self.tesla_route_bias=float(kwargs.pop('tesla_route_bias',0.0))
        self.tower_skill_bias=float(kwargs.pop('tower_skill_bias',0.0))
        self.tower_focus_bias=float(kwargs.pop('tower_focus_bias',0.0))
        self.core_investment_bias=float(kwargs.pop('core_investment_bias',0.0))
        if any(not np.isfinite(v) or not 0<=v<=4 for v in (self.advanced_tower_bias,self.tesla_route_bias,self.tower_skill_bias,self.tower_focus_bias,self.core_investment_bias)):
            raise ValueError('Tower logit biases must be finite and in [0,4]')
        self._tower_prior_indices=None
        if not 0 <= self.type_balance_alpha <= 1:
            raise ValueError('type_balance_alpha must be in [0,1]')
        kwargs.setdefault('features_extractor_class',SceneExtractor)
        super().__init__(*args,**kwargs)

    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters()
        data['type_balance_alpha']=self.type_balance_alpha
        data.update(advanced_tower_bias=self.advanced_tower_bias,tesla_route_bias=self.tesla_route_bias,
                    tower_skill_bias=self.tower_skill_bias,tower_focus_bias=self.tower_focus_bias,
                    core_investment_bias=self.core_investment_bias)
        return data

    def tower_logit_prior(self,c):
        from layout_features import FEATURE_NAMES,FEATURE_DIM,ROUTES
        from support_features import FEATURE_DIM as SUPPORT_DIM
        if c.shape[-1] not in (FEATURE_DIM,SUPPORT_DIM):raise ValueError('Tower logit biases require candidate-v9/v10')
        if self._tower_prior_indices is None:
            self._tower_prior_indices={name:i for i,name in enumerate(FEATURE_NAMES)}
        f=self._tower_prior_indices
        upgrade=c[...,f['type_upgrade']]
        skill=c[...,f['type_tower_skill']]
        grade=c[...,f['target_grade']]
        # Support intermediate upgrades too: the Tesla branch is unavailable
        # until grade 3. A high-tier-only bonus would never help reach it.
        stage=th.where(grade>=.99,1.,th.where(grade>=.74,.65,.35))
        advanced=(grade>=.99).to(c.dtype)
        bonus=self.advanced_tower_bias*(upgrade*stage+skill*advanced*.5)
        tesla=th.maximum(c[...,f['branch_tower_tesla']],
            c[...,f['kind_engineer']]*(c[...,f['target_chain_targets']]>.1).to(c.dtype))*advanced
        # These are existing route samples, not a new path search. Require
        # known coverage; favor middle/rear coverage and multiple routes.
        coverage=(c[...,f['coverage_middle']]+c[...,f['coverage_rear']]).clamp(0,1)
        junction=(c[...,f['covered_routes']]*16-1).clamp(0,1)
        route=c[...,f['coverage_known']]*th.maximum(coverage,junction)
        bonus+=self.tesla_route_bias*tesla*(upgrade+.5*skill)*route
        if self.tower_skill_bias or self.tower_focus_bias:
            # Match the skill actually being purchased, not the highest level
            # of some unrelated power already installed on this tower.
            from skill_features import SKILLS
            source=c[...,[f['skill_'+s+'_source_level'] for s in SKILLS]]
            target=c[...,[f['skill_'+s+'_target_level'] for s in SKILLS]]
            skill_level=(source*(target>source).to(c.dtype)).amax(-1)
            bonus+=self.tower_skill_bias*skill*(1+.5*skill_level)
            if self.tower_focus_bias:
                # Enemy HP share is a cheap route-importance proxy. Future
                # local facts are already restricted to the covered routes.
                share=th.zeros_like(grade)
                for horizon,discount in (('current',1.),('next',1.),('second',.5)):
                    local=c[...,f[horizon+'_local_hp_total']]
                    total=c[...,f[horizon+'_global_hp_total']]
                    share=th.maximum(share,discount*(local/total.clamp(min=1e-6)).clamp(0,1))
                priority=route*(.5+.5*share)
                invested=c[...,f['source_grade']].clamp(0,1)
                bonus+=self.tower_focus_bias*priority*(upgrade*(.5+invested)+skill*(1+skill_level))
        return bonus*c[...,0]

    def core_logit_prior(self,c,action_masks):
        if action_masks is None:return th.zeros(c.shape[:2],device=c.device,dtype=c.dtype)
        from layout_features import FEATURE_NAMES,FEATURE_DIM,ROUTES
        from candidate_features import BRANCHES
        from support_features import FEATURE_DIM as SUPPORT_DIM
        if c.shape[-1] not in (FEATURE_DIM,SUPPORT_DIM):raise ValueError('Core investment requires candidate-v9/v10')
        if self._tower_prior_indices is None:self._tower_prior_indices={n:i for i,n in enumerate(FEATURE_NAMES)}
        f=self._tower_prior_indices
        legal=th.as_tensor(action_masks,device=c.device,dtype=th.bool).reshape(c.shape[:2])
        advisory=(c[...,0]>.5)&~legal&(c[...,f['type_upgrade']]>.5)&(c[...,f['kind_engineer']]>.5)
        active=advisory.any(1)
        index=advisory.to(c.dtype).argmax(1)
        goal=c[th.arange(len(c),device=c.device),index]
        same=(c[...,f['x']]-goal[:,None,f['x']]).abs()<1e-5
        same&=(c[...,f['y']]-goal[:,None,f['y']]).abs()<1e-5
        branches=[f['branch_'+b] for b in BRANCHES]
        target=goal[:,branches]
        branch=(target.sum(-1)[:,None]<.5)|((c[...,branches]*target[:,None,:]).sum(-1)>.5)
        match=same&branch&(c[...,f['type_upgrade']]>.5)
        match&=(c[...,f['target_grade']]-goal[:,None,f['target_grade']]).abs()<1e-5
        affordable=(match&legal).any(1)
        price=goal[:,f['cost']].clamp(min=1e-6)
        gold=(price+goal[:,f['reserve']]).clamp(min=0)
        progress=(gold/price).clamp(0,1)
        wait=c[...,f['type_wait']]
        can_start_wave=(c[...,f['type_wave']]*legal).sum(1)>.5
        saving=(~affordable&~can_start_wave).to(c.dtype)*progress
        routes=[f['covers_'+r] for r in ROUTES]
        shared=(c[...,routes]*goal[:,None,routes]).sum(-1)>.5
        spending=c[...,f['type_build']]+c[...,f['type_upgrade']]+c[...,f['type_tower_skill']]+c[...,f['type_replace']]
        net=(c[...,f['cost']]-c[...,f['refund']]).clamp(min=0)
        delays=(gold[:,None]-net<price[:,None])&(net>0)
        # Penalize nearby competing spending; same-family cannon spreading
        # receives the full penalty, other support towers only half.
        compete=spending*(shared|same).to(c.dtype)*(~match).to(c.dtype)*delays.to(c.dtype)
        family=.5+.5*c[...,f['kind_engineer']]
        from support_features import FEATURE_DIM as SUPPORT_DIM
        if c.shape[-1]==SUPPORT_DIM:
            # A Tesla commitment must not suppress support-tower investment.
            # No extra positive mage bonus: the scorer still judges placement,
            # armor, magic resistance, curse and elemental facts itself.
            family=family*(1-c[...,f['kind_mage']]).clamp(min=0)
        bonus=match.to(c.dtype)+wait*saving[:,None]-compete*family
        return self.core_investment_bias*bonus*active[:,None].to(c.dtype)*legal

    def _build(self,lr_schedule):
        scene_dim=self.observation_space['scene'].shape[0]
        candidate_dim=self.observation_space['candidates'].shape[1]
        arch=self.net_arch or dict(pi=[128,128],vf=[128,128])
        actor=arch.get('pi',[128,128]) if isinstance(arch,dict) else arch
        critic=arch.get('vf',[128,128]) if isinstance(arch,dict) else arch
        self.scorer=mlp(scene_dim+candidate_dim,actor,1)
        self.critic=mlp(scene_dim+2*candidate_dim,critic,1)
        # Small output logits keep the initial policy exploratory.
        for network in (self.scorer,self.critic):
            for module in network:
                if isinstance(module,nn.Linear): nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.scorer[-1].weight,.01)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def raw_candidate_logits(self,obs,action_masks=None):
        scene=obs['scene'].float();candidates=obs['candidates'].float()
        valid=candidates[:,:,0]>.5
        # Padding consumes storage, but no scorer computation or gradient.
        batch,index=valid.nonzero(as_tuple=True)
        logits=th.full(valid.shape,-1e8,device=scene.device,dtype=scene.dtype)
        if len(batch):
            scores=self.scorer(th.cat((scene[batch],candidates[batch,index]),dim=-1)).squeeze(-1)
            logits=logits.index_put((batch,index),scores)
        return logits

    def candidate_logits(self,obs,action_masks=None):
        logits=self.raw_candidate_logits(obs,action_masks)
        if self.advanced_tower_bias or self.tesla_route_bias or self.tower_skill_bias or self.tower_focus_bias:
            logits=logits+self.tower_logit_prior(obs['candidates'].float())
        if self.core_investment_bias:
            logits=logits+self.core_logit_prior(obs['candidates'].float(),action_masks)
        if self.type_balance_alpha:
            from candidate_features import TYPES
            c=obs['candidates'].float()
            valid=c[:,:,0]>.5
            if action_masks is not None:
                valid=valid & th.as_tensor(action_masks,device=c.device,dtype=th.bool).reshape(valid.shape)
            kinds=c[:,:,1:1+len(TYPES)]
            counts=(kinds*valid.unsqueeze(-1)).sum(1)
            multiplicity=(kinds*counts.unsqueeze(1)).sum(-1).clamp(min=1)
            logits=logits-self.type_balance_alpha*multiplicity.log()
        return logits

    def predict_values(self,obs):
        scene=obs['scene'].float();c=obs['candidates'].float();valid=c[:,:,0]>.5
        mean=(c*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp(min=1).unsqueeze(-1)
        maximum=c.masked_fill(~valid.unsqueeze(-1),-th.inf).amax(1)
        maximum=th.where(th.isfinite(maximum),maximum,th.zeros_like(maximum))
        return self.critic(th.cat((scene,mean,maximum),dim=-1))

    def get_distribution(self,obs,action_masks=None):
        distribution=self.action_dist.proba_distribution(action_logits=self.candidate_logits(obs,action_masks))
        if action_masks is not None: distribution.apply_masking(action_masks)
        return distribution

    def forward(self,obs,deterministic=False,action_masks=None):
        distribution=self.get_distribution(obs,action_masks)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions,self.predict_values(obs),distribution.log_prob(actions)

    def evaluate_actions(self,obs,actions,action_masks=None):
        distribution=self.get_distribution(obs,action_masks)
        return self.predict_values(obs),distribution.log_prob(actions),distribution.entropy()


class LayoutCandidatePolicy(CandidatePolicy):
    """v9 actor shares a small permutation-invariant candidate context."""
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        scene_dim = self.observation_space['scene'].shape[0]
        candidate_dim = self.observation_space['candidates'].shape[1]
        arch = self.net_arch or dict(pi=[128,128], vf=[128,128])
        widths = arch.get('pi',[128,128]) if isinstance(arch,dict) else arch
        self.layout_encoder = nn.Sequential(nn.Linear(candidate_dim,16), nn.Tanh())
        self.scene_encoder = nn.Sequential(nn.Linear(scene_dim,64), nn.Tanh())
        self.scorer = mlp(64+candidate_dim+32, widths, 1)
        for network in (self.layout_encoder,self.scene_encoder,self.scorer):
            for module in network:
                if isinstance(module,nn.Linear):
                    nn.init.orthogonal_(module.weight,np.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.scorer[-1].weight,.01)
        self.optimizer = self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def raw_candidate_logits(self, obs, action_masks=None):
        scene = obs['scene'].float(); candidates = obs['candidates'].float()
        valid = candidates[:,:,0]>.5
        if action_masks is not None:
            legal = th.as_tensor(action_masks,device=scene.device,dtype=th.bool).reshape(valid.shape)
            if self.core_investment_bias:
                from light_features import F
                advisory=valid&~legal&(candidates[:,:,F['type_upgrade']]>.5)&(candidates[:,:,F['kind_engineer']]>.5)
                valid=valid&(legal|advisory)
            else:valid = valid & legal
        batch,index = valid.nonzero(as_tuple=True)
        logits = th.full(valid.shape,-1e8,device=scene.device,dtype=scene.dtype)
        if len(batch):
            encoded = self.layout_encoder(candidates[batch,index])
            total = th.zeros((len(scene),16),device=scene.device,dtype=scene.dtype).index_add(0,batch,encoded)
            mean = total/valid.sum(1).clamp(min=1).unsqueeze(-1)
            maximum = th.full_like(total,-th.inf).scatter_reduce(0,batch[:,None].expand(-1,16),encoded,reduce='amax',include_self=True)
            maximum = th.where(th.isfinite(maximum),maximum,th.zeros_like(maximum))
            context = th.cat((mean,maximum),dim=-1)
            scene_context = self.scene_encoder(scene)
            scores = self.scorer(th.cat((scene_context[batch],candidates[batch,index],context[batch]),dim=-1)).squeeze(-1)
            logits = logits.index_put((batch,index),scores)
        return logits
