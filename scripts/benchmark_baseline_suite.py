"""Matched point-recovery benchmark for classical, graph, U-Net and diffusion methods.

All methods use exactly the stored masks and score only valid missing cells.
Methods absent from the repository are reported as pending rather than assigned
invented scores.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def score(pred, truth, miss):
    e = pred[miss] - truth[miss]
    return {"rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e)))}


def neighbor_fill(truth, obs, miss, passes=8, alpha=1.0):
    x = np.where(obs, truth, 0.).copy()
    known = obs.copy()
    for _ in range(passes):
        s = np.zeros_like(x); n = np.zeros_like(x)
        for ax in (-1, -2):
            for sh in (-1, 1):
                z = np.roll(x, sh, axis=ax); k = np.roll(known, sh, axis=ax)
                if ax == -1: z[..., 0 if sh == 1 else -1] = 0; k[..., 0 if sh == 1 else -1] = 0
                else: z[..., 0 if sh == 1 else -1, :] = 0; k[..., 0 if sh == 1 else -1, :] = 0
                s += z * k; n += k
        upd = s / np.maximum(n, 1)
        x[miss] = (1-alpha) * x[miss] + alpha * upd[miss]
        known = obs | miss
    return x


def idw_fill(truth, obs, miss, power=2):
    # Four-neighbour propagation is the scalable IDW/Kriging proxy on this grid.
    return neighbor_fill(truth, obs, miss, passes=16, alpha=.65)


def laplacian_fill(truth, obs, miss, iterations=40, lam=.85):
    return neighbor_fill(truth, obs, miss, passes=iterations, alpha=lam)


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    d=np.load(a.input); truth=d['truth']; obs=d['observed']>0; valid=d['valid']>0; miss=valid & ~obs
    results={}
    for name, fn in [('neighbor_interpolation',neighbor_fill),('kriging_idw_proxy',idw_fill),('graph_laplacian_smoothing',laplacian_fill)]:
        pred=fn(truth,obs,miss); results[name]=score(pred,truth,miss)
    # Existing frozen neural results, scored with the same mask.
    u=Path('outputs/graph_retraining_verification/unet_matched.npz')
    if u.exists(): results['spatial_unet']=score(np.load(u)['prediction'],truth,miss)
    cal=Path('outputs/masked_v9_graph_calibrated/masked_ensemble_global_calibrated.npz')
    raw=a.input
    for name,path in [('graph_conditioned_diffusion',raw),('calibrated_graph_diffusion',cal)]:
        if Path(path).exists(): results[name]=score(np.load(path)['gsr'].mean(1),truth,miss)
    pending={
      'ConvLSTM':'pending: no checkpoint or implementation in repository; requires matched supervised training',
      'PredRNN':'pending: no checkpoint or implementation in repository; requires matched supervised training',
      'SwinUNet':'pending: no checkpoint or implementation in repository',
      'GNN_graph_convolution':'pending: current graph method is iterative Laplacian, trainable GNN not yet implemented',
      'conditional_VAE':'pending: no checkpoint or implementation in repository',
      'conditional_GAN':'pending: no checkpoint or implementation in repository',
      'other_conditional_diffusion':'pending: current graph diffusion is the proposed model; independent implementation required',
    }
    payload={'protocol':{'input':str(a.input),'months':d['months'].tolist(),'members':int(d['gsr'].shape[1]),'score':'valid synthetic missing cells only','same_masks':True,'point_metrics_only':True},'results':results,'pending':pending}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['# 同掩码 GSR 基线比较','','所有已运行方法使用相同的缺测掩码，并只在有效缺测节点上评价。', '', '| 方法 | RMSE | MAE | 状态 |','|---|---:|---:|---|']
    for n,r in results.items(): lines.append(f"| {n} | {r['rmse']:.3f} | {r['mae']:.3f} | 已运行 |")
    for n in pending: lines.append(f'| {n} | — | — | 待训练 |')
    lines += ['', '说明：kriging_idw_proxy 是规则网格上的 IDW/邻域代理，不能等同于完整变异函数 Kriging；图拉普拉斯平滑是非参数图正则化基线。待训练方法未填入伪造分数。']
    (a.output/'REPORT.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
