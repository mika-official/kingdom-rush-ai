"""v4: native tower skills and route-local defenses, threats and spawn schedules."""
import math
import numpy as np
from gymnasium import spaces
from kr_replacement import KRReplacementEnv, REPLACEMENT_MACROS
from kr_tactical import enemy_profile, inside
from kr_macro import KINDS

SKILLS = ('bolt','cluster','disintegrate','dual','healing','holystrike','missile',
          'overcharge','poison','polymorph','shield','shrapnel','sniper','teleport',
          'thorn','throwing','twister')
STRATEGY_MACROS = REPLACEMENT_MACROS + tuple(
    f'skill_{name}_site{rank}' for name in SKILLS for rank in (1,2))
SKILL_FEATURES = 16
ROUTES = tuple(range(1,17)) + (0,)  # Last bucket retains unknown/overflow routes.
SEGMENT_FEATURES = 30
ROUTE_FEATURES = 2 + 3*SEGMENT_FEATURES + 10 + 6 + 5


def log(value, scale=100):
    return math.log1p(max(0,float(value or 0)))/math.log1p(scale)


def mechanism_profile(enemies, future=False):
    total=sum(float(e.get('count',0)) if future else 1 for e in enemies)
    out=np.zeros(6,dtype=np.float32)
    for e in enemies:
        m=e.get('mechanics') or {}
        weight=(float(e.get('count',0)) if future else 1)/max(1,total)
        ratio=float(e.get('hp',0))/max(1,float(e.get('hp_max',e.get('hp',1)) or 1))
        out += weight*np.array([bool(m.get('transform')),float(m.get('transform_magic',0)),
            bool(m.get('regen')),bool(m.get('death_spawn')),bool(m.get('immune')),
            bool(m.get('transform'))*max(0,1-ratio)])
    return out.tolist()


def route_id(e):
    try: pi=int(e.get('pi',0))
    except (ValueError,TypeError): return 0
    return pi if pi in ROUTES else 0


class KRStrategyEnv(KRReplacementEnv):
    schema='macro-v4-strategy'
    base_meta=len(STRATEGY_MACROS)
    n_actions=base_meta+7

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.config['strategy']=True
        size=self.observation_space['summary'].shape[0]+len(ROUTES)*ROUTE_FEATURES+len(SKILLS)*2*SKILL_FEATURES+3
        self.observation_space=spaces.Dict({'summary':spaces.Box(-np.inf,np.inf,(size,),np.float32)})

    @staticmethod
    def tower_kind(sp):
        # Native arcane type is arcane_wizard (v3 only knew the short alias).
        if sp.get('kind')=='arcane_wizard' or sp.get('name')=='tower_arcane_wizard': return 'mage'
        return KRReplacementEnv.tower_kind(sp)

    @staticmethod
    def effective_level(sp):
        if sp.get('kind')=='arcane_wizard' or sp.get('name')=='tower_arcane_wizard': return 4
        return KRReplacementEnv.effective_level(sp)

    def _skill_plans(self,o):
        if o.get('outcome') or o.get('replacement_pending'): return {}
        choices={name:[] for name in SKILLS}
        for sp in o.get('spots') or []:
            if sp.get('state')!='tower' or sp.get('blocked'): continue
            kind=self.tower_kind(sp)
            center=dict(x=sp.get('attack_x',sp.get('x',0)),y=sp.get('attack_y',sp.get('y',0)))
            local=[e for e in o.get('enemies') or [] if inside(e.get('x',0),e.get('y',0),center,
                   float(sp.get('attack_range') or 120))]
            p=enemy_profile(local)
            paths=sp.get('defense_paths') or {}
            future=sum(float(e.get('count',0)) for e in (o.get('forecast') or {}).get('spawns',[])
                       if str(e.get('pi')) in paths)
            quality=self._quality(sp,kind,self._cmax(o)) if kind in KINDS else 0
            for skill in sp.get('skills') or []:
                if skill.get('name') not in choices or not skill.get('ok'): continue
                f=[float(kind==k) for k in KINDS]+[float(sp.get('x',0))/1024,float(sp.get('y',0))/768,
                    float(skill.get('price') or 0)/1000,float(skill.get('level',0))/3,
                    float(skill.get('max_level',0))/3,quality,log(sum(e.get('hp',0) for e in local),10000),
                    float(p[2]),float(p[3]),float(p[1]),log(future),self.effective_level(sp)/4]
                choices[skill['name']].append((sp,skill,f))
        out={}
        for k,name in enumerate(SKILLS):
            remaining=choices[name][:]
            for rank in range(2):
                if not remaining: break
                # Expose two distinct sites, not two aliases of one purchase.
                best=max(remaining,key=lambda x: ((x[2][9]/max(1,x[1].get('price',1))) if rank==0
                    else (x[2][10]+x[2][14]),-int(x[0]['hid'])))
                remaining.remove(best)
                out[27+2*k+rank]=best
        return out

    def _candidates(self,o):
        c=super()._candidates(o)
        for a,(sp,skill,f) in self._skill_plans(o).items():
            c[a]=([('tower_skill',sp['hid'],skill['name'])],('tower_skill',sp,{'skill':skill['name']}),f[9])
        return c

    def action_kind(self,a):
        if 27<=a<self.base_meta: return 'tower_skill'
        return super().action_kind(a)

    def _routes(self,o):
        features=[]
        current=o.get('enemies') or []
        future=(o.get('forecast') or {}).get('spawns') or []
        special=(o.get('special_current') or {}).get('spawns') or []
        towers=[s for s in o.get('spots') or [] if s.get('state')=='tower']
        for pi in ROUTES:
            enemies=[e for e in current if route_id(e)==pi]
            incoming=[e for e in future if route_id(e)==pi]
            planned=[e for e in special if route_id(e)==pi]
            def bins(sp):
                result=[0,0,0]
                for path,values in (sp.get('defense_paths') or {}).items():
                    if route_id({'pi':path})==pi:
                        result=[a+b for a,b in zip(result,values)]
                return result
            covers=[(s,bins(s)) for s in towers]
            features += [float(bool(enemies or incoming or planned or any(any(b) for _,b in covers))), pi/16]
            for segment in range(3):
                local=[e for e in enemies if min(2,int(max(0,min(1,float(e.get('progress') or 0)))*3))==segment]
                defense=[s for s,b in covers if b[segment]>0]
                f=list(enemy_profile(local))+mechanism_profile(local)
                for kind in KINDS:
                    same=[s for s in defense if self.tower_kind(s)==kind and not s.get('blocked')]
                    f += [len(same)/18,sum(self.effective_level(s) for s in same)/72,
                          sum(p.get('level',0) for s in same for p in s.get('skills',[]))/108]
                uncovered=0
                for e in local:
                    if not any(not s.get('blocked') and self.tower_kind(s) in ('archer','mage','engineer')
                        and inside(e.get('x',0),e.get('y',0),
                            dict(x=s.get('attack_x',s.get('x',0)),y=s.get('attack_y',s.get('y',0))),
                            float(s.get('attack_range') or 0)) for s in defense):
                        uncovered+=float(e.get('hp') or 0)
                f += [sum(bool(s.get('blocked')) for s in defense)/18,log(uncovered,10000)]
                features.extend(f)
            features.extend(enemy_profile(incoming,True))
            features.extend(mechanism_profile(incoming,True))
            features.extend([log(sum(e.get('count',0) for e in incoming if e.get('special'))),
                log(sum(e.get('count',0) for e in planned)),
                min((float(e.get('delay') or 0) for e in incoming),default=0)/120,
                max((float(e.get('delay') or 0)+float(e.get('delay_jitter') or 0) for e in incoming),default=0)/120,
                float(any(e.get('expected') for e in incoming))])
        return features

    def _vec(self,o):
        if o.get('strategy_version')!=1:
            raise RuntimeError('v4 observation missing; load the matching kr_env.lua and kr_strategy.lua')
        original=super()._vec(o)['summary']
        features=self._routes(o)
        plans=self._skill_plans(o)
        for a in range(27,self.base_meta): features.extend(plans[a][2] if a in plans else [0]*SKILL_FEATURES)
        special=(o.get('forecast') or {}).get('special') or {}
        features += [float(special.get('sources',0))/8,log(special.get('unknown',0)),
                     log((o.get('special_current') or {}).get('unknown',0))]
        return {'summary':np.concatenate((original,np.asarray(features,dtype=np.float32)))}

    def step(self,action):
        plans=self._skill_plans(self.o)
        result=super().step(action)
        if action in plans and result[4].get('agent_action_ok'):
            sp,skill,_=plans[action]
            self.ep.setdefault('tower_skills',[]).append(dict(hid=sp['hid'],skill=skill['name'],
                level=skill.get('level',0)+1,cost=skill.get('price'),wave=self.prev_o.get('wave'),tick=self.t))
        if 'ep_stats' in result[4]: result[4]['ep_stats']['tower_skills']=list(self.ep.get('tower_skills',[]))
        return result
