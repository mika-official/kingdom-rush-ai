"""Shared native attack estimates used by both capacity and v6 damage features.

Unknown offensive skills receive a conservative prior, never a known formula.
The prior is 25% of base single-target DPS at level 1, +12.5% per extra level
(capped at level 3). No actual damage credit or reward is generated here.
"""
import json
from pathlib import Path
import numpy as np

OFFENSIVE_SKILLS = frozenset(('poison', 'sniper', 'shrapnel', 'disintegrate', 'missile',
                            'cluster', 'overcharge', 'bolt', 'holystrike', 'throwing',
                            'twister', 'dual'))
SKILL_PRIOR = .25
CAPACITY_DPS_UNIT = 20.0
REFERENCE_PATH = Path(__file__).with_name('tower_combat_reference.json')


def reference_catalog():
    if not REFERENCE_PATH.exists():
        return {}
    return json.loads(REFERENCE_PATH.read_text(encoding='utf8'))['catalog']


def attack_dps(attack):
    if 'estimated_dps' in attack:
        return max(0, float(attack['estimated_dps']))
    if not attack.get('known'):
        return 0.0
    mean = (float(attack.get('damage_min') or 0)+float(attack.get('damage_max') or 0))/2
    return max(0, mean)/max(.01, float(attack.get('interval') or 1))


def effective_attacks(profile):
    """Return enabled native attacks plus non-duplicated, explicitly unknown priors."""
    profile = profile if isinstance(profile, dict) else {}
    attacks = [a for a in profile.get('attacks', []) if a.get('enabled')]
    bases = [a for a in attacks if a.get('known') and not a.get('power') and not a.get('conditional')]
    base_dps = sum(attack_dps(a) for a in bases)
    modeled = {a.get('power') for a in attacks if a.get('known') or a.get('effect_known')}
    modeled.update(m.get('name') for m in profile.get('mechanisms',[]) if m.get('known'))
    unknown = {}
    for skill in profile.get('skills', []):
        name, level = skill.get('name'), float(skill.get('level') or 0)
        if level > 0 and not skill.get('known') and name in OFFENSIVE_SKILLS and name not in modeled:
            unknown[name] = level
    for a in attacks:
        name = a.get('power')
        if not a.get('known') and not a.get('effect_known') and name in OFFENSIVE_SKILLS and name not in modeled:
            unknown[name] = max(unknown.get(name, 0), float(a.get('level') or 1))
    # Preserve all native unknown/control attacks. The synthetic entry carries
    # only the prior so a skill appearing in both lists is never counted twice.
    result = list(attacks)
    if base_dps:
        channel = max(bases, key=attack_dps).get('channel', 'physical')
        for name, level in sorted(unknown.items()):
            fraction = SKILL_PRIOR*(1+.5*(min(3, level)-1))
            result.append(dict(enabled=True, known=False, estimated=True, power=name,
                               estimated_dps=base_dps*fraction, channel=channel,
                               flying=any(a.get('flying') for a in bases), max_targets=1))
    return result


def rates(profile):
    out = np.zeros(3); area = 0; chain = 1; control = 0
    profile = profile if isinstance(profile, dict) else {}
    attacks = effective_attacks(profile)
    partial = not bool(profile.get('known')) or any(not a.get('known') for a in attacks)
    partial |= any(s.get('level', 0)>0 and not s.get('known') for s in profile.get('skills', []))
    for a in attacks:
        control += float(bool(a.get('control')))
        value = attack_dps(a)
        out[{'physical':0, 'magic':1, 'true':2}.get(a.get('channel'), 0)] += value
        area = max(area, float(a.get('radius') or 0))
        chain = max(chain, float(a.get('max_targets') or 1))
    return out, area, chain, control, bool(partial)


def capacity(profile):
    """Physical/magic/area/air/block capacity, normalized native DPS, not grade."""
    result = np.zeros(5)
    for a in effective_attacks(profile):
        value = attack_dps(a)/CAPACITY_DPS_UNIT
        result[1 if a.get('channel') == 'magic' else 0] += value
        if a.get('radius', 0)>0:
            targets = max(1, float(a.get('max_targets') or 1)-1) if a.get('chain') else 1
            result[2] += value*targets*(float(a.get('chain_factor', 1)) if a.get('chain') else 1)
        if a.get('flying'):
            result[3] += value
    result[4] = float(profile.get('block') or 0)/3
    return result
