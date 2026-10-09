"""Stepwise investment scoring using the existing read-only next-upgrade row."""
import torch as th
from route_policy import RouteCandidatePolicy
from support_features import F,SCENE_NAMES
from candidate_features import BRANCHES


class StepwiseRoutePolicy(RouteCandidatePolicy):
    policy_version='route-v2-invest'

    def investment_state(self,obs,action_masks):
        c=obs['candidates'].float()
        legal=th.as_tensor(action_masks,device=c.device,dtype=th.bool).reshape(c.shape[:2])
        advisory=(c[...,0]>.5)&~legal&(c[...,F['type_upgrade']]>.5)&(c[...,F['kind_engineer']]>.5)
        active=advisory.any(1)
        goal=c[th.arange(len(c),device=c.device),advisory.to(c.dtype).argmax(1)]
        same=(c[...,F['x']]-goal[:,None,F['x']]).abs()<1e-5
        same&=(c[...,F['y']]-goal[:,None,F['y']]).abs()<1e-5
        branches=[F['branch_'+b] for b in BRANCHES];target=goal[:,branches]
        branch=(target.sum(-1)[:,None]<.5)|((c[...,branches]*target[:,None,:]).sum(-1)>.5)
        match=same&branch&(c[...,F['type_upgrade']]>.5)&legal&active[:,None]
        match&=(c[...,F['target_grade']]-goal[:,None,F['target_grade']]).abs()<1e-5
        price=goal[:,F['cost']]
        gold=price+goal[:,F['reserve']]
        ready=match.any(1)
        # An absent native action may be blocked for reasons other than gold.
        # Do not reward saving when the next step is already affordable.
        affordable=active&(gold>=price-1e-6)
        output=sum(obs['scene'][:,SCENE_NAMES.index('tower_'+kind)] for kind in ('archer','mage','engineer'))
        offense=sum(c[...,F['kind_'+kind]] for kind in ('archer','mage','engineer'))>.5
        builds=(c[...,F['type_build']]>.5)&offense&legal
        bootstrap=(output<=1e-6)&builds.any(1)
        return dict(active=active,ready=ready,affordable=affordable,match=match,goal=goal,bootstrap=bootstrap)

    def candidate_logits(self,obs,action_masks=None):
        logits=super().candidate_logits(obs,action_masks)
        if action_masks is None or not self.core_investment_bias:return logits
        state=self.investment_state(obs,action_masks)
        c=obs['candidates'].float();wait=c[...,F['type_wait']]
        legal=th.as_tensor(action_masks,device=c.device,dtype=th.bool).reshape(c.shape[:2])
        # Cancel legacy saving when native readiness and price disagree.
        blocked_affordable=state['affordable']&~state['ready']
        can_wave=((c[...,F['type_wave']]>.5)&legal).any(1)
        logits=logits-self.core_investment_bias*wait*(blocked_affordable&~can_wave)[:,None]
        # The persistent goal already identifies the next incremental upgrade.
        # Strengthen that step and remove the attraction of waiting, while
        # retaining every legal tactical and competing investment action.
        invested=state['goal'][:,F['source_grade']].clamp(0,1)
        logits=logits+self.core_investment_bias*state['match'].to(c.dtype)*(1+invested[:,None])
        stop_waiting=state['ready']|state['bootstrap']
        logits=logits-self.core_investment_bias*wait*stop_waiting[:,None]
        return logits

    def economic_wait_eligible(self,obs,action_masks):
        if not self.core_investment_bias:return th.zeros(len(obs['scene']),device=obs['scene'].device,dtype=th.bool)
        state=self.investment_state(obs,action_masks)
        # Apply the strong saving margin only when a next-step target is
        # actually unaffordable. Other expert waits retain ordinary BC labels.
        return state['active']&~state['affordable']&~state['bootstrap']
