"""Conservative planning baseline; scores are never used as PPO reward or labels."""
import numpy as np
from candidate_baseline import CandidateBaseline,WEIGHTS
from planning_features import F,FEATURE_NAMES


class PlanningBaseline(CandidateBaseline):
    def __init__(self,env,weights=None):
        self.env=env.unwrapped
        self.weights=np.zeros(len(FEATURE_NAMES),np.float32)
        adjusted=dict(WEIGHTS)
        # Native marginal damage replaces the v5 grade-based benefits.
        for name in ('benefit_physical','benefit_magic','benefit_area','benefit_air',
                     'delta_physical','delta_magic','delta_area','delta_air'):
            adjusted[name]=0
        adjusted.update(current_damage_gain=2,future_damage_gain=1,lost_control=-1,lost_block=-.8,
            lost_skill_levels=-1,lost_advanced=-.5,rebuild_shortfall=-1,recent_investment=-.3,
            rain_avoidable_hp=2,rain_emergency=5,rain_reserve_value=-1,
            save_benefit=.8,saving_emergency=-3)
        for name,value in (adjusted if weights is None else weights).items(): self.weights[F[name]]=value

    def scores(self,obs):
        c=np.asarray(obs['candidates']);scores=c[:,:len(self.weights)]@self.weights
        idle_wait=(c[:,F['type_wait']]>.5)&(c[:,F['save_income_active']]<.5)
        scores[idle_wait]-=c[idle_wait,F['save_benefit']]*self.weights[F['save_benefit']]
        active_wait=(c[:,F['type_wait']]>.5)&~idle_wait
        # Discount the distant saving outcome just like an incremental upgrade;
        # otherwise waiting for the entire chain dominates buying its first step.
        scores[active_wait]-=c[active_wait,F['save_benefit']]*self.weights[F['save_benefit']]
        scores[active_wait]+=c[active_wait,F['save_benefit']]*self.weights[F['save_benefit']]/(1+2*c[active_wait,F['save_cost']])
        replace=c[:,F['type_replace']]>.5
        net_gain=2*c[:,F['current_damage_gain']]+c[:,F['future_damage_gain']]
        sacrifice=(c[:,F['loss']]+c[:,F['lost_control']]+c[:,F['lost_block']]+
                   c[:,F['lost_skill_levels']]+c[:,F['lost_advanced']]*.5+c[:,F['rebuild_shortfall']])
        bad=replace & ((net_gain<=sacrifice+.1)|(net_gain<=c[:,F['best_upgrade_gain']]))
        scores[bad]=-1e6
        rain=c[:,F['type_rain']]>.5
        retain=rain & (c[:,F['rain_emergency']]<=0) & (
            c[:,F['rain_avoidable_hp']]*2<=c[:,F['rain_reserve_value']]*c[:,F['rain_forecast_confidence']]+.05)
        scores[retain]=-1e6
        late=int(self.env.o.get('level_idx') or 1)>=13
        if late:
            develop=(c[:,F['type_upgrade']]+c[:,F['type_tower_skill']])>.5
            # Pay for an actual improvement on threatened routes, including a
            # discounted native path to Tesla/Polymorph. No bonus for counts.
            scores[develop]+=.8*np.maximum(net_gain[develop],c[develop,F['best_upgrade_gain']])
            build=c[:,F['type_build']]>.5
            output_count=sum(self.env.tower_kind(t)!='barrack' for t in self.env.o.get('spots',[]) if t.get('state')=='tower')
            if output_count>=4:
                gain=np.maximum(0,net_gain)
                scores[build]-=.4/(1+gain[build])
            scores[build]+=.15*c[build,F['best_upgrade_gain']]
        return scores
