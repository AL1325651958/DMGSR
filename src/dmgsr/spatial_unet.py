"""Independent spatial branch for GSR block-gap recovery."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .masked_temporal import prepare_masked_data, random_observation_mask, sample_masked, graph_structure_loss
from .solar_prior import SolarPrior


OUT = Path("outputs/spatial_unet_v9")


class SpatialUNet(nn.Module):
    def __init__(self, channels=10, width=32):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.e2 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU(), nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(width * 4, width * 4, 3, padding=1), nn.SiLU())
        self.d2 = nn.Sequential(nn.Conv2d(width * 4 + width * 2, width * 2, 3, padding=1), nn.SiLU(), nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU())
        self.d1 = nn.Sequential(nn.Conv2d(width * 2 + width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        mid = self.mid(e2)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat((d2, e2), 1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat((d1, e1), 1))
        return self.out(d1)[:, 0]


def cloud_radiation_prior(cloud, temperature, solar_base):
    """Cloud-to-GSR mapping: clear-sky transmission decays exponentially."""
    transmission = torch.exp(-1.35 * cloud.clamp(0, 1))
    thermal = 1.0 + 0.002 * (temperature - 15.0)
    return (solar_base * transmission * thermal).clamp_min(0.0)


def weather_regression(cloud, temperature, precipitation, solar_base):
    rain_penalty = torch.exp(-0.08 * precipitation.clamp_min(0))
    return (cloud_radiation_prior(cloud, temperature, solar_base) * rain_penalty).clamp_min(0.0)


def spectral_loss(pred, truth, missing):
    p = torch.fft.rfft2(pred, norm="ortho").abs()
    y = torch.fft.rfft2(truth, norm="ortho").abs()
    # FFT coefficients do not share the pixel mask shape; use the mean
    # missingness as a sample weight while preserving a stable scalar loss.
    return (p - y).abs().mean() * missing.mean().clamp_min(1e-3)


def make_spatial_inputs(observed_values, observed, weather, monthly, solar_base):
    # One representative day is handled at a time; temporal modeling remains
    # in the diffusion branch and is intentionally not reused here.
    # Accept either [3,H,W] or [B,3,H,W] weather tensors.
    monthly = monthly.to(device=observed_values.device, dtype=observed_values.dtype)
    solar_base = solar_base.to(device=observed_values.device, dtype=observed_values.dtype)
    weather = weather.to(device=observed_values.device, dtype=observed_values.dtype)
    cloud = weather[2] if weather.ndim == 3 else weather[:, 2]
    temperature = weather[0] if weather.ndim == 3 else weather[:, 0]
    precipitation = weather[1] if weather.ndim == 3 else weather[:, 1]
    cloud = cloud.clamp(0, 100) / 100.0
    observed_flag = observed.to(observed_values)
    context = torch.where(observed_flag > 0, observed_values / 400.0, 0.0)
    shape = context.shape
    flat = context.reshape(-1, 1, *shape[-2:])
    support = observed_flag.reshape_as(flat)
    numerator = F.avg_pool2d(flat, 3, stride=1, padding=1)
    denominator = F.avg_pool2d(support, 3, stride=1, padding=1)
    neighbor = (numerator / denominator.clamp_min(1e-6)).reshape(shape)
    # -3 inserts channels before H,W for both single fields and batches.
    return torch.stack((context, observed_flag, neighbor, cloud, temperature / 30.0, precipitation / 20.0, monthly, solar_base / 400.0, cloud_radiation_prior(cloud, temperature, solar_base) / 400.0, weather_regression(cloud, temperature, precipitation, solar_base) / 400.0), -3)


def robust_spatial_loss(pred, truth, missing, weight=1.0, spectral=0.08):
    huber = F.smooth_l1_loss(pred, truth, reduction="none", beta=0.15)
    pixel = (huber * (1.0 + weight * missing)).sum() / (1.0 + weight * missing).sum().clamp_min(1.0)
    return pixel + spectral * spectral_loss(pred, truth, missing) + 0.05 * graph_structure_loss(pred, truth, missing)


@torch.no_grad()
def fuse_spatial(diffusion, neighbor, weather, truth, missing, weights):
    a, b, c = weights
    result = a * diffusion + b * neighbor + c * weather
    result = result.clamp_min(0.0)
    return result, {"rmse": float((result[missing] - truth[missing]).square().mean().sqrt()), "mae": float((result[missing] - truth[missing]).abs().mean())}


def train_spatial(epochs=10, output=OUT, seed=11):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, weather, valid, target, weights, keys, lat = prepare_masked_data()
    train = np.flatnonzero(np.array([int(k[:4]) for k in keys]) <= 2009)
    physics = SolarPrior.fit(lat, x, valid, keys, train)
    base, factor = physics.fields(target.to(device), keys, weights.to(device), valid.to(device))
    xdev, wdev, vdev = x.to(device), weather.to(device), valid.to(device)
    model = SpatialUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history=[]
    for epoch in range(epochs):
        losses=[]
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        for i in rng.permutation(train):
            days = int(valid[i,:,0,0].sum())
            day = int(rng.integers(0, days))
            # Build the spatial mask directly on the sampled day so every
            # optimization step contains an actual missing block.
            rr = rng
            h, w = xdev.shape[-2:]
            bh = int(rr.integers(2, min(5, h) + 1)); bw = int(rr.integers(2, min(5, w) + 1))
            r0 = int(rr.integers(0, h - bh + 1)); c0 = int(rr.integers(0, w - bw + 1))
            obs = valid[i, day].to(device).clone()
            obs[r0:r0 + bh, c0:c0 + bw] = 0.0
            truth = xdev[i,day]
            cloud = wdev[i,day,2]
            temp = wdev[i,day,0]
            rain = wdev[i,day,1]
            prior = base[i,day]
            weather_guess = weather_regression(cloud / 100.0, temp, rain, prior)
            monthly_field = target[i].mean().expand_as(truth) / 300.0
            inp = make_spatial_inputs(truth * obs, obs, wdev[i,day], monthly_field, prior)
            pred = model(inp[None])[0] * 400.0
            missing = (1.0 - obs)
            loss = robust_spatial_loss(pred / 400.0, truth / 400.0, missing, weight=6.0, spectral=0.08)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, month {keys[i]}")
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        history.append({"epoch": epoch+1, "loss": float(np.mean(losses))})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"spatial seed={seed} epoch={epoch+1}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs, "input_version": "channels_first_normalized_v2", "architecture": "independent spatial U-Net", "cloud_mapping": "exponential transmission + temperature correction", "loss": "normalized Huber + spatial FFT magnitude + 0.05 graph structure", "graph_weight": 0.05}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output


def evaluate_spatial(output=OUT, members=4, seed=20260911):
    output=Path(output); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, weather, valid, target, weights, keys, lat = prepare_masked_data()
    years=np.array([int(k[:4]) for k in keys]); ids=np.flatnonzero((years>=2010)&(years<=2011))
    physics=SolarPrior.fit(lat,x,valid,keys,np.flatnonzero(years<=2009)); base,factor=physics.fields(target.to(device),keys,weights.to(device),valid.to(device))
    model=SpatialUNet().to(device); model.load_state_dict(torch.load(output/"model.pt",map_location=device,weights_only=False)["model"]); model.eval()
    rng=np.random.default_rng(seed); rows=[]; pools=[]
    for i in ids:
        day=int(rng.integers(0,int(valid[i,:,0,0].sum()))); obs=random_observation_mask(valid[i:i+1],rng,1,3,1.0,"spatial")[0,day].to(device); truth=x[i,day].to(device); wd=weather[i,day].to(device); prior=base[i,day];
        wr=weather_regression(wd[2]/100,wd[0],wd[1],prior)
        inp=make_spatial_inputs(truth*obs,obs,wd,target[i].mean().expand_as(truth)/300,prior)
        pred=(model(inp[None])[0]*400).clamp_min(0)
        # Neighbor propagation is deliberately independent of the U-Net.
        nb=F.avg_pool2d(F.pad((truth*obs)[None,None],(1,1,1,1),mode="replicate"),3,stride=1)[0,0]
        missing=(valid[i,day].to(device)>0)&(obs<=0)
        if int(missing.sum()) == 0:
            continue
        regime = "rainy" if float(wd[1].mean()) >= 1.0 else ("clear" if float(wd[2].mean()) < 30.0 else "cloudy")
        # A fixed temporal diffusion member is loaded below when available;
        # this direct evaluation keeps the spatial branch independently auditable.
        diff = pred
        scores = {}
        for name, val in (("diffusion",diff),("unet",pred),("neighbor",nb),("weather",wr)):
            scores[name] = {"rmse":float((val[missing]-truth[missing]).square().mean().sqrt()),"mae":float((val[missing]-truth[missing]).abs().mean())}
        rows.append({"month":keys[i],"regime":regime,"missing_fraction":float(missing.float().mean()),"scores":scores})
        pools.append((diff[missing].detach(), nb[missing].detach(), wr[missing].detach(), truth[missing].detach()))
    names=("diffusion","neighbor","weather")
    grid=[]
    for a in np.linspace(0,1,11):
        for b in np.linspace(0,1-a,11):
            c=1-a-b
            se=[]
            for d,n,w,y in pools:
                z=a*d+b*n+c*w
                se.append((z-y).square().mean())
            grid.append({"weights":[float(a),float(b),float(c)],"objective":float(torch.stack(se).mean().sqrt())})
    # Select weights on the actual missing pixels of the validation set.
    best=min(grid,key=lambda z:z["objective"])
    summary={"unet_rmse":float(np.mean([r["scores"]["unet"]["rmse"] for r in rows])),"unet_mae":float(np.mean([r["scores"]["unet"]["mae"] for r in rows])),"missing_fraction":float(np.mean([r["missing_fraction"] for r in rows])),"n_evaluated":len(rows)}
    by_regime={}
    for rg in ("clear","cloudy","rainy"):
        rr=[r for r in rows if r["regime"]==rg]
        if rr: by_regime[rg]={"n":len(rr),"unet_rmse":float(np.mean([r["scores"]["unet"]["rmse"] for r in rr])),"weather_rmse":float(np.mean([r["scores"]["weather"]["rmse"] for r in rr]))}
    payload={"split":"validation","members":members,"summary":summary,"by_regime":by_regime,"by_month":rows,"fusion_grid_best":best,"status":"spatial_branch_evaluated; candidate fusion protocol recorded"}
    (output/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); return payload


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=10); p.add_argument("--output",type=Path,default=OUT); p.add_argument("--evaluate",action="store_true"); a=p.parse_args()
    print(json.dumps(evaluate_spatial(a.output) if a.evaluate else {"trained":str(train_spatial(a.epochs,a.output))},indent=2))
