"""route-v1: shared route encoder and coverage-conditioned candidate scoring.

Uses the existing v10 observations. No route ID enters either actor or critic.
"""
import torch as th
from torch import nn
from candidate_policy import CandidatePolicy, mlp
from layout_features import ROUTES, HORIZONS, COMPOSITION, SEGMENTS, CHANNELS
from support_features import FEATURE_NAMES, SCENE_NAMES


class RouteCandidatePolicy(CandidatePolicy):
    policy_version='route-v1'

    def _build(self,lr_schedule):
        if self.observation_space['scene'].shape[0]!=len(SCENE_NAMES) or self.observation_space['candidates'].shape[-1]!=len(FEATURE_NAMES):
            raise ValueError('route-v1 requires candidate-v10 observations')
        route_names=[tuple(f'path_{r}_{h}_{n}' for h in HORIZONS for n in COMPOSITION)+
                     tuple(f'path_{r}_{s}_{n}' for s in SEGMENTS for n in CHANNELS) for r in ROUTES]
        route_indices=th.tensor([[SCENE_NAMES.index(n) for n in names] for names in route_names])
        route_set=set(route_indices.flatten().tolist())
        self.register_buffer('route_indices',route_indices)
        self.register_buffer('base_scene_indices',th.tensor([i for i in range(len(SCENE_NAMES)) if i not in route_set]))
        self.register_buffer('coverage_indices',th.tensor([FEATURE_NAMES.index('covers_'+r) for r in ROUTES]))
        self.register_buffer('base_candidate_indices',th.tensor([i for i,n in enumerate(FEATURE_NAMES) if not n.startswith('covers_')]))
        scene_dim=len(self.base_scene_indices);candidate_dim=len(self.base_candidate_indices)
        arch=self.net_arch or dict(pi=[128,128],vf=[128,128])
        actor=arch.get('pi',[128,128]) if isinstance(arch,dict) else arch
        critic=arch.get('vf',[128,128]) if isinstance(arch,dict) else arch
        self.route_encoder=nn.Sequential(nn.Linear(len(route_names[0]),16),nn.Tanh())
        self.scene_encoder=nn.Sequential(nn.Linear(scene_dim+32,64),nn.Tanh())
        self.layout_encoder=nn.Sequential(nn.Linear(candidate_dim,16),nn.Tanh())
        self.scorer=mlp(64+candidate_dim+32+16,actor,1)
        self.critic=mlp(scene_dim+32+2*candidate_dim,critic,1)
        for network in (self.route_encoder,self.scene_encoder,self.layout_encoder,self.scorer,self.critic):
            for module in network:
                if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,2**.5);nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.scorer[-1].weight,.01)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def route_context(self,obs):
        scene=obs['scene'].float();c=obs['candidates'].float()
        facts=scene[:,self.route_indices]
        encoded=self.route_encoder(facts)  # B x R x 16, once per observation.
        coverage=c.index_select(-1,self.coverage_indices)
        present=facts.ne(0).any(-1)|((coverage*(c[...,0]>.5).unsqueeze(-1)).sum(1)>0)
        mean=(encoded*present.unsqueeze(-1)).sum(1)/present.sum(1).clamp(min=1).unsqueeze(-1)
        maximum=encoded.masked_fill(~present.unsqueeze(-1),-th.inf).amax(1)
        maximum=th.where(th.isfinite(maximum),maximum,th.zeros_like(maximum))
        return encoded,th.cat((mean,maximum),-1)

    def raw_candidate_logits(self,obs,action_masks=None):
        scene=obs['scene'].float();c=obs['candidates'].float();valid=c[...,0]>.5
        if action_masks is not None:
            legal=th.as_tensor(action_masks,device=c.device,dtype=th.bool).reshape(valid.shape)
            if self.core_investment_bias:
                from support_features import F
                advisory=valid&~legal&(c[...,F['type_upgrade']]>.5)&(c[...,F['kind_engineer']]>.5)
                valid=valid&(legal|advisory)
            else:valid=valid&legal
        batch,index=valid.nonzero(as_tuple=True)
        logits=th.full(valid.shape,-1e8,device=c.device,dtype=c.dtype)
        if not len(batch):return logits
        routes,global_routes=self.route_context(obs)
        base=c[batch,index].index_select(-1,self.base_candidate_indices)
        encoded=self.layout_encoder(base)
        total=th.zeros((len(scene),16),device=c.device,dtype=c.dtype).index_add(0,batch,encoded)
        mean=total/valid.sum(1).clamp(min=1).unsqueeze(-1)
        maximum=th.full_like(total,-th.inf).scatter_reduce(0,batch[:,None].expand(-1,16),encoded,reduce='amax',include_self=True)
        maximum=th.where(th.isfinite(maximum),maximum,th.zeros_like(maximum))
        context=th.cat((mean,maximum),-1)
        scene_context=self.scene_encoder(th.cat((scene.index_select(-1,self.base_scene_indices),global_routes),-1))
        # Aggregate all candidates in one batched matrix product. Route encoder
        # outputs are shared, rather than recomputed for each tower/action.
        coverage=c.index_select(-1,self.coverage_indices)
        local=th.bmm(coverage,routes)/coverage.sum(-1,keepdim=True).clamp(min=1)
        scores=self.scorer(th.cat((scene_context[batch],base,context[batch],local[batch,index]),-1)).squeeze(-1)
        return logits.index_put((batch,index),scores)

    def predict_values(self,obs):
        scene=obs['scene'].float();c=obs['candidates'].float();valid=c[...,0]>.5
        _,global_routes=self.route_context(obs)
        base=c.index_select(-1,self.base_candidate_indices)
        mean=(base*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp(min=1).unsqueeze(-1)
        maximum=base.masked_fill(~valid.unsqueeze(-1),-th.inf).amax(1)
        maximum=th.where(th.isfinite(maximum),maximum,th.zeros_like(maximum))
        return self.critic(th.cat((scene.index_select(-1,self.base_scene_indices),global_routes,mean,maximum),-1))

    def augment_routes(self,obs,inplace=False,permutation=None):
        """Move route facts and coverage together; labels/masks stay unchanged."""
        scene=obs['scene'];c=obs['candidates'];batch=len(scene);routes=len(ROUTES)
        if permutation is None:permutation=th.rand((batch,routes),device=scene.device).argsort(-1)
        permutation=th.as_tensor(permutation,device=scene.device,dtype=th.long)
        if permutation.ndim==1:permutation=permutation.expand(batch,-1)
        if permutation.shape!=(batch,routes) or not th.equal(permutation.sort(-1).values,th.arange(routes,device=scene.device).expand(batch,-1)):
            raise ValueError('Expected one complete route permutation per observation')
        facts=scene[:,self.route_indices].gather(1,permutation.unsqueeze(-1).expand(-1,-1,self.route_indices.shape[1]))
        coverage=c.index_select(-1,self.coverage_indices).gather(2,permutation[:,None,:].expand(-1,c.shape[1],-1))
        result=obs if inplace else dict(scene=scene.clone(),candidates=c.clone())
        result['scene'][:,self.route_indices.flatten()]=facts.flatten(1)
        result['candidates'][:,:,self.coverage_indices]=coverage
        return result
