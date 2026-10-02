"""Frozen-weight, calendar-aligned development evaluation for the manuscript.

No retraining or test-period selection. Calibration: 2010; report: 2011.
The reporting year has been inspected previously and is not an untouched test.
"""
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import dmgsr.masked_temporal as mt
from dmgsr.solar_prior import SolarPrior
from dmgsr.spatial_unet import SpatialUNet,make_spatial_inputs
from benchmark_baseline_suite import neighbor_fill,laplacian_fill
OUT=ROOT/'paper/evidence';OUT.mkdir(parents=True,exist_ok=True)

def metric(a,y):
    n=a.shape[0];mean=a.mean(0);lo,hi=np.quantile(a,[.05,.95],axis=0)
    coeff=(2*np.arange(1,n+1)-n-1)[:,None]
    return {'rmse':float(np.sqrt(np.mean((mean-y)**2))),'mae':float(np.mean(abs(mean-y))),
        'crps':float(np.mean(abs(a-y).mean(0)-(coeff*np.sort(a,axis=0)).sum(0)/n**2)),
        'coverage90':float(np.mean((y>=lo)&(y<=hi))),'width90':float(np.mean(hi-lo))}

def pool(s,mask):return s.transpose(1,0,2,3,4)[:,mask]

def fuse(u,d,w,g):
    um=u.mean(1,keepdims=True);dm=d.mean(1,keepdims=True)
    return w*um+(1-w)*dm+g*np.concatenate((w*(u-um),(1-w)*(d-dm)),axis=1)

def calibrate(u,d,mask,y):
    uc=pool(u,mask);dc=pool(d,mask);yy=y[mask];um=uc.mean(0,keepdims=True);dm=dc.mean(0,keepdims=True)
    rows=[]
    for w in np.linspace(0,1,21):
        mean=w*um+(1-w)*dm;res=np.concatenate((w*(uc-um),(1-w)*(dc-dm)))
        for g in np.linspace(.5,8,31):
            r=metric(mean+g*res,yy);r.update(w=float(w),g=float(g),objective=r['crps']+8*abs(r['coverage90']-.9));rows.append(r)
    best=min(rows,key=lambda r:r['objective'])
    # Strong comparator: spread-calibrated U-Net, with the same search range/objective.
    ur=[]
    for g in np.linspace(.5,8,31):
        r=metric(um+g*(uc-um),yy);r.update(g=float(g),objective=r['crps']+8*abs(r['coverage90']-.9));ur.append(r)
    return best,min(ur,key=lambda r:r['objective']),rows,ur

@torch.no_grad()
def infer():
    torch.set_num_threads(4);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc=np.load(ROOT/'outputs/masked_v9_graph/masked_ensemble_validation.npz')
    x,weather,v,target,weights,keys,lat=mt.prepare_masked_data()
    months=arc['months'].tolist();ids=[keys.index(k) for k in months]
    assert len(set(months))==24
    np.testing.assert_allclose(x[ids].numpy(),arc['truth'])
    y=x[ids];valid=v[ids];obs=torch.from_numpy(arc['observed']);miss=(valid.numpy()>0)&(obs.numpy()<=0)
    ck=torch.load(ROOT/'outputs/masked_v9_graph/model_11.pt',map_location='cpu',weights_only=False)
    physics=SolarPrior(ck['latitude'],ck['template'])
    goal=target[ids].to(dev);vd=valid.to(dev);wd=weights.to(dev)
    base,factor=physics.fields(goal,months,wd,vd)
    res=(y.to(dev)-base)/factor.clamp_min(1e-4)/ck['residual_scale']
    met=((weather[ids]-ck['feature_mean'][None,None,:,None,None])/ck['feature_std'][None,None,:,None,None]).to(dev)
    net=mt.MaskedTemporalVelocityNet().to(dev);net.load_state_dict(ck['model']);net.eval()
    # Correct only the target-energy conversion inside the legacy projector.
    # Original partial_project multiplies by padded length; multiplying its target
    # by valid_count/padded_count recovers the intended valid-month energy.
    original=mt.partial_project
    def corrected(values,tgt,w,valid_days,observed):
        count=valid_days.reshape(len(values),values.shape[1]).sum(1)
        return original(values,tgt*(count/values.shape[1])[:,None,None],w,valid_days,observed)
    mt.partial_project=corrected
    # Exact deterministic unit check: constant 28-day field must remain constant.
    testv=torch.ones(1,31,10,10);testv[:,28:]=0;testx=100*testv;testobs=testv.clone();testobs[:,5:8,2:4,2:4]=0
    torch.testing.assert_close(corrected(testx,torch.full((1,5,5),100.),weights,testv[:,:,0,0],testobs),testx)
    ds=[];start=time.time()
    try:
        for offset in range(0,24,4):
            ix=slice(offset,offset+4)
            samples=mt.sample_masked(net,goal[ix],months[ix],vd[ix],met[ix],res[ix]*obs[ix].to(dev),obs[ix].to(dev),wd,
                float(ck['mean_scale']),float(ck['residual_scale']),base[ix],factor[ix],seed=20260912+offset,members=12,chunk_size=8)
            ds.append(samples.numpy());print('Diffusion months',offset+4,'/24',flush=True)
    finally:mt.partial_project=original
    diff=np.concatenate(ds)
    # Match the independently trained U-Net's train-only prior and proper dates.
    train=np.flatnonzero(np.array([int(k[:4]) for k in keys])<=2009)
    up=SolarPrior.fit(lat,x,v,keys,train);ub,_=up.fields(goal,months,wd,vd)
    us=[]
    for seed in range(8):
        netu=SpatialUNet().to(dev);netu.load_state_dict(torch.load(ROOT/f'outputs/spatial_unet_ensemble/model_{seed}.pt',map_location=dev,weights_only=False));netu.eval()
        pred=y.numpy().copy()
        loc=np.argwhere(miss.any((-1,-2)))
        for starti in range(0,len(loc),64):
            locations=loc[starti:starti+64];b,d=locations.T
            yy=y[b,d].to(dev);oo=obs[b,d].to(dev);ww=weather[np.array(ids)[b],d].to(dev)
            mm=goal[b].mean((-1,-2))[:,None,None].expand_as(yy)/300
            inp=make_spatial_inputs(yy*oo,oo,ww,mm,ub[b,d])
            pp=(netu(inp)*400).clamp(0,400)
            pred[b,d]=torch.where(oo>0,yy,pp).cpu().numpy()
        us.append(pred)
    unet=np.stack(us,axis=1)
    assert np.isfinite(diff).all() and np.isfinite(unet).all()
    np.testing.assert_allclose(diff.mean(1)[obs.numpy()>0],y.numpy()[obs.numpy()>0],atol=.001)
    np.testing.assert_allclose(unet.mean(1)[obs.numpy()>0],y.numpy()[obs.numpy()>0],atol=.001)
    train_mean=(x[train]*v[train]).sum((0,1))/v[train].sum((0,1))
    np.savez_compressed(OUT/'aligned_ensembles.npz',diffusion=diff,unet=unet,truth=y.numpy(),observed=obs.numpy(),valid=valid.numpy(),missing=miss,months=np.array(months),latitude=lat,train_mean=train_mean.numpy(),target=target[ids].numpy(),weights=weights.numpy())
    (OUT/'runtime.json').write_text(json.dumps({'seconds':time.time()-start,'device':str(dev),'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if dev.type=='cuda' else None,'month_indices':ids},indent=2))

def analyze():
    a=np.load(OUT/'aligned_ensembles.npz');u=a['unet'];d=a['diffusion'];y=a['truth'];miss=a['missing'];months=a['months']
    cal=miss.copy();cal[12:]=False;rep=miss.copy();rep[:12]=False
    best,ubest,grid,ugrid=calibrate(u,d,cal,y)
    fused=fuse(u,d,best['w'],best['g']);um=u.mean(1,keepdims=True);uc=um+ubest['g']*(u-um)
    neighbor=neighbor_fill(y,a['observed']>0,miss)[:,None];lap=laplacian_fill(y,a['observed']>0,miss)[:,None]
    methods={'Neighbor':neighbor,'Laplacian':lap,'Diffusion':d,'UNet':u,'UNet_cal':uc,'Fusion':fused}
    summary={k:metric(pool(s,rep),y[rep]) for k,s in methods.items()}
    monthly=[]
    for i in range(24):
        for name,s in methods.items():
            r=metric(s[i][:,miss[i]],y[i][miss[i]]);r.update(month=str(months[i]),method=name,n=int(miss[i].sum()),gap_days=int(miss[i].any((-1,-2)).sum()));monthly.append(r)
    rng=np.random.default_rng(20260912);boot=[]
    # Paired month-block bootstrap; descriptive uncertainty, not independent-cell CI.
    for _ in range(2000):
        idx=rng.integers(12,24,12);m=miss[idx];yy=y[idx][m]
        f=pool(fused[idx],m);b=pool(uc[idx],m)
        boot.append([metric(f,yy)['rmse']-metric(b,yy)['rmse'],metric(f,yy)['crps']-metric(b,yy)['crps']])
    boot=np.array(boot)
    result={'calibration_year':2010,'report_year':2011,'status':'development evaluation, previously inspected years',
        'n_cal':int(cal.sum()),'n_report':int(rep.sum()),'fusion_selection':best,'unet_calibration':ubest,
        'summary':summary,'paired_month_bootstrap':{'draws':2000,'fusion_minus_unet_cal_rmse_95':np.quantile(boot[:,0],[.025,.975]).tolist(),'fusion_minus_unet_cal_crps_95':np.quantile(boot[:,1],[.025,.975]).tolist()},
        'negative_fused_fraction':float((pool(fused,rep)<0).mean()),'months':months.tolist()}
    for fn,rows in [('monthly_metrics.csv',monthly),('fusion_search.csv',grid),('unet_search.csv',ugrid)]:
        with (OUT/fn).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'final_ensembles.npz',**methods,truth=y,missing=miss,months=months,bootstrap=boot)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    if not (OUT/'aligned_ensembles.npz').exists():infer()
    analyze()
