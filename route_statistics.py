"""Count/life based route statistics; overkill cannot cancel another batch's gap."""
import math


def finalize_route(row):
    present=bool(row['present'])
    count=max(1,float(row['count']))
    row['leak_fraction']=min(1,max(0,row['leaks']/count)) if present else 0.0
    row['life_fraction']=min(1,max(0,row['life']/max(1,row['life_budget']))) if present else 0.0
    row['worst_shortfall']=min(1,max(0,1-row['bottleneck'])) if present else 0.0
    row['handles']=present and row['leaks']<=1e-6
    # Below the handling threshold, use a count-weighted deficit. Above it,
    # expose only the weakest batch's headroom, never summed overkill.
    headroom=max(0,min(6,math.log(max(1e-6,row['bottleneck']))))/6 if row['handles'] else 0.0
    row['margin']=headroom if row['handles'] else -row['leak_fraction']
    row['bottleneck_margin']=max(-6,min(6,math.log(max(1e-6,row['bottleneck']))))/6 if present else 0.0
    ratio=row['damage']/max(1,row['hp'])
    row['hp_margin']=max(-6,min(6,math.log(max(1e-6,ratio))))/6 if present else 0.0
    return row
