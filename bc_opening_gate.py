"""Compare cold-start behavior with the demonstrated setup, without action rules."""


def opening_gate(records,profiles,revision='strict-v1'):
    if revision not in ('strict-v1','behavior-v2'):raise ValueError('Unknown opening gate revision')
    checks=[]
    for r in records:
        key=f"{r.get('level_idx')}:{r.get('level_mode')}"
        profile=profiles.get(key)
        if profile is None:
            actions=r.get('stats',{}).get('actions',{})
            passed=actions.get('build',0)>0 and actions.get('upgrade',0)>0
        else:
            setup=r.get('waves',{}).get('0',{})
            passed=(setup.get('selected_build',0)<=profile['max_builds']
                    and setup.get('selected_upgrade',0)>=profile['min_upgrades']
                    and setup.get('selected_tower_skill',0)>=profile['min_skills'])
            for wave,limits in profile.get('early_waves',{}).items():
                actual=r.get('waves',{}).get(wave,{})
                passed=passed and (actual.get('selected_build',0)<=limits['max_builds']
                    and actual.get('selected_upgrade',0)>=limits['min_upgrades']
                    and actual.get('selected_tower_skill',0)>=limits['min_skills'])
            if profile.get('early_waves'):passed=passed and int(r.get('wave') or 0)>=3
        checks.append(dict(level=key,passed=passed,profile=profile,
                           mode=r.get('mode'),outcome=r.get('outcome'),wave=r.get('wave'),
                           actual_setup=r.get('waves',{}).get('0',{}),
                           actual_early_waves={w:r.get('waves',{}).get(w,{}) for w in (profile or {}).get('early_waves',{})}))
        if revision=='behavior-v2':
            check=checks[-1];reasons=[]
            # A completed native cold-start victory is stronger evidence than
            # matching the teacher's purchase timing exactly.
            if r.get('outcome')=='victory' and r.get('victory') is True:
                check.update(passed=True,reason='native_cold_victory',failure_reasons=[])
                continue
            if int(r.get('wave') or 0)<3:reasons.append('defeated_before_wave_3')
            if profile is None:
                if not passed:reasons.append('no_build_and_upgrade_evidence')
            else:
                setup=check['actual_setup']
                if setup.get('selected_build',0)>profile['max_builds']:reasons.append('setup_build_sprawl')
                if profile['max_builds']>0 and setup.get('selected_build',0)==0:reasons.append('empty_setup')
                if setup.get('selected_upgrade',0)<profile['min_upgrades']:reasons.append('setup_upgrade_shortfall')
                if setup.get('selected_tower_skill',0)<profile['min_skills']:reasons.append('setup_skill_shortfall')
                limits=profile.get('early_waves',{})
                if limits:
                    actual=[r.get('waves',{}).get(w,{}) for w in limits]
                    builds=sum(row.get('selected_build',0) for row in actual)
                    investments=sum(row.get('selected_upgrade',0)+row.get('selected_tower_skill',0) for row in actual)
                    expected_builds=sum(row['max_builds'] for row in limits.values())
                    expected_investments=sum(row['min_upgrades']+row['min_skills'] for row in limits.values())
                    # One support foundation is allowed; investment must still
                    # meet the teacher's cumulative early minimum. Upgrades and
                    # skills can move between waves 1 and 2.
                    check['early_budget']=dict(builds=builds,max_builds=expected_builds+1,
                        investments=investments,min_investments=expected_investments)
                    if builds>expected_builds+1:reasons.append('early_build_sprawl')
                    if investments<expected_investments:reasons.append('early_investment_shortfall')
            check.update(passed=not reasons,reason='cumulative_early_investment',failure_reasons=reasons)
    return dict(passed=bool(checks) and all(c['passed'] for c in checks),episodes=len(checks),
                revision=revision,
                criterion=('Native cold victory or viable setup plus cumulative early investment; one additional early foundation allowed'
                    if revision=='behavior-v2' else 'Cold setup and labelled early waves respect demonstrated build counts and minimum upgrades/skills'),checks=checks)
