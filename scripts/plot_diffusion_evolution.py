"""Actual frozen-model trajectory, with a fixed illustrative missing-data case.

Contract: show how recovered fields change, without assuming accuracy improves.
Image plate + quantitative diagnostics; Python; Chinese screen-readable export.
All gap cells are scored; a fixed middle day supplies spatial illustrations.
"""
from pathlib import Path
import sys
import json
import csv
import hashlib

import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, 'C:/Users/13256/.agents/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
from dmgsr.masked_temporal import prepare_masked_data, MaskedTemporalVelocityNet, sample_masked
from dmgsr.solar_prior import SolarPrior

OUT = ROOT / 'outputs/figures/diffusion_evolution'
MODEL = ROOT / 'outputs/masked_v9_graph/model_11.pt'
CASE = '2011-07'
SEED = 20260911
MEMBERS = 12
DAY = 11  # Day 12: predetermined middle of the 9--15 July gap.
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def acquire():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, features, valid, target, weights, keys, lat = prepare_masked_data()
    i = keys.index(CASE)
    checkpoint = torch.load(MODEL, map_location='cpu', weights_only=False)
    net = MaskedTemporalVelocityNet().to(device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    truth, v, goal = x[i:i+1].to(device), valid[i:i+1].to(device), target[i:i+1].to(device)
    obs = v.clone()
    obs[:, 8:15, 3:7, 3:7] = 0
    physics = SolarPrior(checkpoint['latitude'], checkpoint['template'])
    base, factor = physics.fields(goal, [CASE], weights.to(device), v)
    residual = (truth - base) / factor.clamp_min(1e-4) / checkpoint['residual_scale']
    weather = ((features[i:i+1] - checkpoint['feature_mean'][None,None,:,None,None]) /
               checkpoint['feature_std'][None,None,:,None,None]).to(device)
    initial = []
    def capture_initial(module, args):
        if not initial:
            initial.append(args[0].detach().cpu().numpy().copy())
    hook = net.register_forward_pre_hook(capture_initial)
    final, traces = sample_masked(net, goal, [CASE], v, weather, residual * obs, obs,
        weights.to(device), float(checkpoint['mean_scale']), float(checkpoint['residual_scale']),
        base, factor, seed=SEED, members=MEMBERS, chunk_size=8, return_trace=True)
    hook.remove()
    stages = np.concatenate([torch.stack(traces).numpy(), final.numpy()], axis=0)
    truth, obs, v = [q[0].cpu().numpy() for q in (truth, obs, v)]
    missing = (v > 0) & (obs == 0)
    assert missing.sum() == 112 and int(v[:,0,0].sum()) == 31
    assert np.isfinite(stages).all()
    assert np.allclose(stages[:, :, obs > 0], truth[obs > 0], atol=1e-3)
    np.savez_compressed(OUT / 'source_data.npz', stages=stages, truth=truth,
        observed=obs, valid=v, missing=missing, initial_latent=initial[0], latitude=lat)
    return stages, truth, obs, missing, initial[0]


def metrics(stages, truth, missing):
    rows = []
    for j, s in enumerate(stages):
        a, y = s[:, missing], truth[missing]
        mean = a.mean(0)
        lo, hi = np.quantile(a, [0.05, 0.95], axis=0)
        order = np.sort(a, axis=0)
        coeff = (2*np.arange(1, MEMBERS+1)-MEMBERS-1)[:,None]
        rows.append({'stage': f'step_{8*(j+1)}' if j < 8 else 'final_postprocessed',
            'rmse': float(np.sqrt(np.mean((mean-y)**2))), 'mae': float(np.mean(abs(mean-y))),
            'crps': float(np.mean(np.mean(abs(a-y),axis=0) - (coeff*order).sum(0)/MEMBERS**2)),
            'coverage90': float(np.mean((y>=lo)&(y<=hi))), 'width90': float(np.mean(hi-lo))})
    with (OUT/'stage_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def spatial(ax, field, title, norm, cmap='cividis'):
    cm=mpl.colormaps[cmap].copy(); cm.set_bad('#e4e8ed')
    im=ax.imshow(field, origin='lower', interpolation='nearest', aspect='auto', norm=norm, cmap=cm)
    ax.add_patch(Rectangle((2.5,2.5),4,4,fill=False,ec='#D55E00',lw=1.3,ls='--'))
    ax.set(xticks=[],yticks=[],title=title)
    return im


def save(fig, stem, dpi=300):
    require_matplotlib_panel_alignment(fig,json_out=OUT/f'{stem}.alignment.json',strict=True)
    fig.savefig(OUT/f'{stem}.png',dpi=dpi)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    plt.close(fig)


def draw(stages,truth,obs,missing,initial,rows):
    means=stages.mean(1)
    # Full common range, no percentile clipping or spatial interpolation.
    norm=Normalize(0,float(np.ceil(max(stages[:,:,DAY].max(),truth[DAY].max())/50)*50))
    errors=means[:,DAY]-truth[DAY]
    emax=float(np.ceil(np.max(abs(errors))/20)*20)
    enorm=Normalize(-emax,emax)
    fig=plt.figure(figsize=(15,8))
    grid=fig.add_gridspec(2,6,left=.055,right=.92,bottom=.19,top=.79,hspace=.75,wspace=.62,height_ratios=[1,1.2])
    fields=[np.where(obs[DAY]>0,truth[DAY],np.nan),means[0,DAY],means[2,DAY],means[5,DAY],means[-1,DAY],truth[DAY]]
    titles=['缺测输入','第 8 / 64 步','第 24 / 64 步','第 48 / 64 步','最终输出*','POWER 参考值']
    for j,(field,title) in enumerate(zip(fields,titles)):
        ax=fig.add_subplot(grid[0,j]); im=spatial(ax,field,title,norm)
    cax=fig.add_axes([.935,.575,.010,.215]);cax.set_label('<colorbar-main>')
    fig.colorbar(im,cax=cax).set_label('太阳辐射（W/m²）',fontsize=9)
    ax=fig.add_subplot(grid[1,:2])
    ax.plot(np.arange(8),[r['rmse'] for r in rows[:8]],'-o',color='#247A8A',label='分块去噪＋约束反馈',ms=4)
    ax.plot([7,8],[rows[7]['rmse'],rows[8]['rmse']], '--', color='#D55E00')
    ax.scatter([8],[rows[8]['rmse']],color='#D55E00',s=35,label='最终后处理',zorder=3)
    ax.set(xticks=range(9),xticklabels=['8','16','24','32','40','48','56','64','最终'],
           ylabel='缺测区 RMSE（W/m²）',xlabel='已完成的反向去噪步数',title='误差是否真的下降？（全部 112 个缺测值）')
    ax.grid(axis='y',alpha=.18); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=2,frameon=False)
    ax=fig.add_subplot(grid[1,2:4])
    im=spatial(ax,np.where(missing[DAY],errors[-1],np.nan),'最终偏差：蓝色偏低，红色偏高',enorm,'RdBu_r')
    cax=fig.add_axes([.395,.105,.20,.013]);cax.set_label('<colorbar-error>')
    fig.colorbar(im,cax=cax,orientation='horizontal').set_label('预测 − 参考（W/m²）；灰色区域不计分',fontsize=9)
    ax=fig.add_subplot(grid[1,4:])
    days=np.arange(9,16)
    y=truth[8:15,3:7,3:7].mean((1,2))
    series=stages[-1,:,8:15,3:7,3:7].mean((2,3))
    lo,hi=np.quantile(series,[.05,.95],axis=0)
    ax.fill_between(days,lo,hi,color='#247A8A',alpha=.18,label='90% 集合区间')
    ax.plot(days,y,'-o',color='#252D3A',ms=4,label='参考值')
    ax.plot(days,series.mean(0),'-o',color='#247A8A',ms=4,label='最终均值')
    ax.set(xlabel='2011 年 7 月日期',ylabel='缺测空间块平均辐射（W/m²）',title='逐日变化能否跟上参考值？',xticks=days)
    ax.grid(axis='y',alpha=.18); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=3,frameon=False,columnspacing=.6,handlelength=1.4)
    fig.text(.055,.94,'扩散过程演化  |  变化不等于更准确',fontsize=22,weight='bold',color='#202D42')
    fig.text(.055,.885,'固定案例：2011 年 7 月 9–15 日，中央 4×4 网格缺测；上排展示 7 月 12 日，橙框标记缺测区。',fontsize=11)
    fig.text(.055,.84,'同一色标展示 12 个成员的平均重建；图中每一阶段均来自实际模型运行。',fontsize=10,color='#536173')
    fig.text(.055,.03,'* 最终输出包含月约束与空间后处理。月条件来自完整参考产品；本图为受控案例，不代表独立站点精度或整体性能。',fontsize=10,color='#536173')
    save(fig,'diffusion_evolution_overview')

    framepaths=[]
    for j in range(-1,9):
        f,axes=plt.subplots(1,3,figsize=(12,4.9),gridspec_kw={'left':.055,'right':.92,'bottom':.25,'top':.72,'wspace':.24})
        spatial(axes[0],truth[DAY],'固定参考：2011-07-12',norm)
        if j == -1:
            nmax=float(abs(initial[0,DAY]).max())
            im=spatial(axes[1],initial[0,DAY],'初始含噪隐变量（成员 1）',Normalize(-nmax,nmax),'RdBu_r')
            spatial(axes[2],np.where(obs[DAY]>0,truth[DAY],np.nan),'实际输入：灰色为缺测',norm)
            headline='从噪声开始'; subtitle='中图为无量纲隐变量，不能当作辐射值或计算重建误差。'
            cax=f.add_axes([.94,.25,.012,.47]);cax.set_label('<colorbar>');f.colorbar(im,cax=cax).set_label('标准化隐变量（无量纲）')
        else:
            im=spatial(axes[1],means[j,DAY],'当前重建（12 成员均值）',norm)
            spatial(axes[2],np.where(missing[DAY],errors[j],np.nan),'当前误差（仅缺测区）',enorm,'RdBu_r')
            headline=f'完成 {(j+1)*8} / 64 步' if j<8 else '最终输出：约束与空间后处理后'
            subtitle=f"全部缺测值：RMSE {rows[j]['rmse']:.1f} W/m²   |   MAE {rows[j]['mae']:.1f} W/m²   |   90% 覆盖率 {rows[j]['coverage90']:.1%}"
            cax=f.add_axes([.94,.25,.012,.47]);cax.set_label('<colorbar>');f.colorbar(im,cax=cax).set_label('辐射（W/m²）')
        f.text(.055,.90,headline,fontsize=20,weight='bold',color='#202D42')
        f.text(.055,.80,subtitle,fontsize=11)
        f.text(.055,.135,f'辐射固定范围 0–{norm.vmax:.0f} W/m²；误差固定范围 ±{emax:.0f} W/m²（蓝低 / 红高）。',fontsize=10)
        f.text(.055,.07,'固定单案例；参考产品提供月条件；过程可能反复，误差未必单调下降。',fontsize=10,color='#536173')
        stem=f'frame_{j+1:02d}';save(f,stem);framepaths.append(OUT/f'{stem}.png')
    frames=[Image.open(p).convert('RGB') for p in framepaths]
    frames[0].save(OUT/'diffusion_evolution.gif',save_all=True,append_images=frames[1:],duration=[1800]+[1100]*8+[2600],loop=0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if '--render-only' in sys.argv:
        data=np.load(OUT/'source_data.npz')
        stages,truth,obs,missing,initial=[data[k] for k in ('stages','truth','observed','missing','initial_latent')]
        rows=metrics(stages,truth,missing)
        draw(stages,truth,obs,missing,initial,rows)
        print('RENDERED',flush=True)
        return
    print('Extracting real diffusion trajectory...',flush=True)
    stages,truth,obs,missing,initial=acquire()
    rows=metrics(stages,truth,missing)
    print(json.dumps(rows,indent=2),flush=True)
    metadata={'case':CASE,'case_selection':'Fixed July 2011, central 4x4 gap on days 9-15; no score-based selection',
        'training_seed':11,'sampling_seed':SEED,'members':MEMBERS,'missing_values':int(missing.sum()),
        'checkpoint_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        'source_model':'masked_v9_graph','stage_semantics':'8-step chunk estimates after feedback; separate final postprocessing',
        'reference':'NASA POWER satellite/reanalysis-derived product, not station truth',
        'monthly_condition':'Complete reference monthly 2x2 coarse averages, including held-out values',
        'interval':'Empirical 5th-95th quantiles of 12 members; block-mean intervals aggregate each member before quantiles',
        'spatial_day':12,'metric_scope':'All 112 missing day-grid cells, equal cell weights',
        'display':'Native grid, no interpolation, full common color ranges',
        'limitations':['Single illustrative validation case, not performance benchmark','July avoids known short-month projection issue',
                       'No post-hoc spread calibration or U-Net fusion included'], 'metrics':rows}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    draw(stages,truth,obs,missing,initial,rows)
    print('DONE',flush=True)


if __name__=='__main__':
    main()
