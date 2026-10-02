"""Compare saved latest weights on the exact earlier illustrative gap.

Figure contract: matched fixed-case accuracy and temporal fidelity, with no
selection or retraining. Recompute U-Net with calendar-aligned conditions.
Python image plate + quantitative comparison; screen layout, 300 dpi + vectors.
"""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,'C:/Users/13256/.agents/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
from dmgsr.masked_temporal import prepare_masked_data
from dmgsr.solar_prior import SolarPrior
from dmgsr.spatial_unet import SpatialUNet,make_spatial_inputs
OUT=ROOT/'outputs/figures/latest_comparison'
CASE='2011-07'
DAY=11
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei'],
 'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
 'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def score(samples,truth,missing):
    a=samples[:,missing];y=truth[missing];n=len(a)
    mean=a.mean(0);lo,hi=np.quantile(a,[.05,.95],axis=0)
    coeff=(2*np.arange(1,n+1)-n-1)[:,None]
    return dict(rmse=float(np.sqrt(np.mean((mean-y)**2))),mae=float(np.mean(abs(mean-y))),
     crps=float(np.mean(np.mean(abs(a-y),0)-(coeff*np.sort(a,axis=0)).sum(0)/n**2)),
     coverage90=float(np.mean((y>=lo)&(y<=hi))),width90=float(np.mean(hi-lo)))

@torch.no_grad()
def compute():
    torch.set_num_threads(4)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    previous=np.load(ROOT/'outputs/figures/diffusion_evolution/source_data.npz')
    truth=previous['truth'];obs=previous['observed'];missing=previous['missing'];diff=previous['stages'][-1]
    x,weather,valid,target,weights,keys,lat=prepare_masked_data()
    idx=keys.index(CASE)
    np.testing.assert_allclose(x[idx].numpy(),truth,atol=1e-5)
    train=np.flatnonzero(np.array([int(k[:4]) for k in keys])<=2009)
    physics=SolarPrior.fit(lat,x,valid,keys,train)
    base,_=physics.fields(target[idx:idx+1].to(dev),[CASE],weights.to(dev),valid[idx:idx+1].to(dev))
    yy=torch.from_numpy(truth).to(dev);mm=torch.from_numpy(obs).to(dev)
    # Explicit absolute month index is essential: idx=138, not validation index=18.
    monthly=target[idx].mean().to(dev).expand_as(yy)/300
    inputs=make_spatial_inputs(yy*mm,mm,weather[idx].to(dev),monthly,base[0])
    unets=[];hashes=[]
    for seed in range(8):
        path=ROOT/f'outputs/spatial_unet_ensemble/model_{seed}.pt'
        net=SpatialUNet().to(dev)
        net.load_state_dict(torch.load(path,map_location=dev,weights_only=False));net.eval()
        pr=(net(inputs)*400).clamp(0,400)
        unets.append(torch.where(mm>0,yy,pr).cpu().numpy())
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    unet=np.stack(unets)
    protocol=json.loads((ROOT/'outputs/fused_probabilistic_model/metrics.json').read_text(encoding='utf-8'))['protocol']
    w=protocol['mean_fusion_weight_unet'];scale=protocol['uncertainty_scale']
    um=unet.mean(0,keepdims=True);dm=diff.mean(0,keepdims=True)
    mean=w*um+(1-w)*dm
    fused=mean+scale*np.concatenate([w*(unet-um),(1-w)*(diff-dm)],axis=0)
    methods={'diffusion':diff,'unet':unet,'latest_fusion':fused}
    for s in methods.values():
        assert np.isfinite(s).all()
        np.testing.assert_allclose(s.mean(0)[obs>0],truth[obs>0],atol=.001)
    result={name:score(s,truth,missing) for name,s in methods.items()}
    np.testing.assert_allclose(fused.mean(0),mean[0],atol=.001)
    np.savez_compressed(OUT/'source_data.npz',truth=truth,observed=obs,missing=missing,**methods)
    meta={'case':CASE,'day_shown':12,'missing_values':int(missing.sum()),'weight_unet':w,'spread_scale':scale,
      'selection':'Same fixed July 9-15 central 4x4 gap as previous figure, no performance-based selection',
      'rerun':'Latest saved weights and frozen fusion parameters, corrected calendar-aligned U-Net inference',
      'calendar_index':idx,'legacy_inference_issue':'Old validation-local index 18 selected Jan2000-origin weather/monthly/base (July2001) instead of July2011',
      'calibration_limit':'Fusion parameters inherited from legacy misaligned evaluation; not recalibrated after correction',
      'reference':'NASA POWER product; monthly target derived from complete reference, including hidden values',
      'unet_checkpoint_sha256':hashes,'metrics':result,
      'negative_fused_missing_members_fraction':float((fused[:,missing]<0).mean()),
      'aggregation':'All 112 gap cells equal weight; temporal intervals computed after per-member spatial averaging'}
    (OUT/'metrics.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['method',*next(iter(result.values())).keys()]);writer.writeheader()
        for name,r in result.items():writer.writerow({'method':name,**r})
    return truth,obs,missing,methods,meta

def spatial(ax,field,title,norm,cmap='cividis'):
    cm=mpl.colormaps[cmap].copy();cm.set_bad('#e4e8ed')
    im=ax.imshow(field,origin='lower',interpolation='nearest',aspect='auto',norm=norm,cmap=cm)
    ax.add_patch(Rectangle((2.5,2.5),4,4,fill=False,edgecolor='#D55E00',lw=1.3,ls='--'))
    ax.set(xticks=[],yticks=[],title=title)
    return im

def save(fig,name):
    require_matplotlib_panel_alignment(fig,json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.png',dpi=300)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)

def draw(truth,obs,missing,methods,meta):
    names=['diffusion','unet','latest_fusion'];labels=['扩散最终输出','U-Net 集合','最新融合方案']
    colors=['#8593A5','#AF7D48','#167D8D']
    means={k:v.mean(0) for k,v in methods.items()}
    vmax=float(np.ceil(max(truth[DAY].max(),*[s[DAY].max() for s in means.values()])/50)*50)
    norm=Normalize(0,vmax)
    fig=plt.figure(figsize=(14,8.5))
    gs=fig.add_gridspec(2,4,left=.065,right=.90,bottom=.20,top=.77,hspace=.65,wspace=.50,height_ratios=[1,1.15])
    ax=fig.add_subplot(gs[0,0]);spatial(ax,truth[DAY],'POWER 参考值',norm)
    for j,(name,label) in enumerate(zip(names,labels)):
        ax=fig.add_subplot(gs[0,j+1]);im=spatial(ax,means[name][DAY],label,norm)
    cax=fig.add_axes([.923,.575,.011,.195]);cax.set_label('<colorbar>')
    fig.colorbar(im,cax=cax).set_label('太阳辐射（W/m²）')
    ax=fig.add_subplot(gs[1,:2]);days=np.arange(9,16)
    ax.plot(days,truth[8:15,3:7,3:7].mean((1,2)),'-o',color='#202D42',label='参考值',lw=2,ms=4,zorder=5)
    for name,label,color in zip(names,labels,colors):
        s=methods[name][:,8:15,3:7,3:7].mean((2,3))
        ax.plot(days,s.mean(0),'-o',color=color,label=label,ms=3,lw=1.5)
        if name=='latest_fusion':
            lo,hi=np.quantile(s,[.05,.95],axis=0)
            ax.fill_between(days,lo,hi,color=color,alpha=.13,label='融合 90% 集合区间')
    ax.set(xticks=days,xlabel='2011 年 7 月日期',ylabel='缺测空间块均值（W/m²）',title='同一缺测区：逐日变化与融合预测区间')
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),ncol=3,frameon=False,handlelength=1.3,columnspacing=.8)
    ax=fig.add_subplot(gs[1,2:]);y=truth[missing]
    lim=[float(min(y.min(),*[means[n][missing].min() for n in names]))-10,
         float(max(y.max(),*[means[n][missing].max() for n in names]))+10]
    ax.plot(lim,lim,'--',color='#687383',lw=1,label='完全准确：预测 = 参考')
    for name,label,color in zip(names,labels,colors):
        ax.scatter(y,means[name][missing],s=18,c=color,alpha=.60,label=label,edgecolors='none')
    ax.set(xlim=lim,ylim=lim,xlabel='参考值（W/m²）',ylabel='预测值（W/m²）',title='全部 112 个缺测值：越接近虚线越准确')
    ax.grid(alpha=.12)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),ncol=2,frameon=False,handlelength=1.3)
    fig.text(.065,.94,'最新融合效果  |  同一案例、同一缺测掩码',fontsize=21,weight='bold',color='#202D42')
    fig.text(.065,.88,'2011 年 7 月 9–15 日，中央 4×4 网格缺测；上排固定展示 7 月 12 日。',fontsize=11)
    r=meta['metrics']
    fig.text(.065,.825,'缺测 RMSE（W/m²）：  '+ '     |     '.join(f'{label}  {r[name]["rmse"]:.1f}' for name,label in zip(names,labels)),fontsize=11)
    fig.text(.065,.035,'最新保存权重＋固定融合参数；已纠正 U-Net 推理日期错位，尚未重新校准。月条件来自完整 POWER 参考产品。',fontsize=9,color='#536173')
    save(fig,'latest_matched_comparison')

    fig,axes=plt.subplots(1,3,figsize=(12,4.6),gridspec_kw={'left':.055,'right':.91,'bottom':.20,'top':.70,'wspace':.25})
    errors={n:means[n][DAY]-truth[DAY] for n in names}
    emax=float(np.ceil(max(np.max(abs(e[missing[DAY]])) for e in errors.values())/10)*10)
    for ax,name,label in zip(axes,names,labels):
        im=spatial(ax,np.where(missing[DAY],errors[name],np.nan),label,Normalize(-emax,emax),'RdBu_r')
    cax=fig.add_axes([.933,.20,.012,.50]);cax.set_label('<colorbar>')
    fig.colorbar(im,cax=cax).set_label('预测 − 参考（W/m²）')
    fig.text(.055,.90,'误差放在一起看：颜色越接近白色，偏差越小',fontsize=18,weight='bold',color='#202D42')
    fig.text(.055,.79,'7 月 12 日同一缺测区，共用色标；蓝色偏低，红色偏高，灰色区域不计分。',fontsize=10)
    fig.text(.055,.08,'固定单案例；最新融合采用正确日期重新推理，融合参数沿用旧评估选择值，未重新拟合。',fontsize=9,color='#536173')
    save(fig,'latest_error_comparison')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if '--render-only' in sys.argv:
        data=np.load(OUT/'source_data.npz');meta=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        truth,obs,missing=[data[k] for k in ('truth','observed','missing')]
        methods={k:data[k] for k in ('diffusion','unet','latest_fusion')}
    else:
        truth,obs,missing,methods,meta=compute()
    print(json.dumps(meta['metrics'],indent=2),flush=True)
    draw(truth,obs,missing,methods,meta)

if __name__=='__main__':main()
