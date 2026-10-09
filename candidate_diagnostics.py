"""Sampled candidate score diagnostics, not a source of training reward."""
import numpy as np


def ranked_event(event,catalog,scores,selected):
    available=np.asarray(scores[:len(catalog)])
    top=np.argsort(-available,kind='stable')[:5]
    record=dict(event)
    record['selected_score']=float(available[selected])
    record['selected_rank']=int(np.sum(available>available[selected]))+1
    record['top']=[dict(catalog[int(i)],score=float(available[i])) for i in top]
    legal=np.isfinite(available)&(available>-1e7)
    probabilities=np.zeros_like(available,dtype=float)
    if legal.any():
        weights=np.exp(available[legal]-available[legal].max())
        probabilities[legal]=weights/weights.sum()
    mass={};spread={}
    for typ in {c['type'] for c in catalog}:
        indices=[i for i,c in enumerate(catalog) if c['type']==typ and legal[i]]
        mass[typ]=float(probabilities[indices].sum())
        if indices: spread[typ]=float(np.ptp(available[indices]))
    record.update(type_probability=mass,type_score_spread=spread,legal_choices=int(legal.sum()),
                  decision_entropy=float(-np.sum(probabilities[legal]*np.log(probabilities[legal].clip(min=1e-30)))))
    return record


def evaluation_event(model,obs,mask,info,selected):
    if hasattr(model,'scores'): scores=model.scores(obs)
    else:
        import torch
        tensor,_=model.policy.obs_to_tensor(obs)
        with torch.no_grad(): scores=model.policy.candidate_logits(tensor,mask)[0].cpu().numpy()
    scores=np.where(mask,scores,-1e8)
    return ranked_event(info['candidate_event'],info['candidate_catalog'],scores,selected)
