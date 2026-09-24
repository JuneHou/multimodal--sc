"""panel_generator — stage 1 of the latent SHC learned by a generator f(l_i, t) (plan 2026-09-11, Reading 1).

Stage 1 of the paper (slide 6: covariate step + local-linear kernel smoothing of one site's own months) gives
l_hat_t for t <= T0. Here it is replaced by a neural network trained ACROSS SITES on the pre-period rows:

    l_hat_{i,t} = f_theta(l_i, e_t),   t = 1 .. T0 of the fit,

with l_i a learned 32-value code per site, e_t a learned 8-value embedding per period, theta shared weights.
Two arms differ only in the output head: "mlp" (Linear 256 -> 980) and "cnn" (Linear 256 -> 32x7x7, transposed
convolution to 32x14x14, 3x3 convolution to 5x14x14, flattened in the cache's (channel, row, col) order).
The generator is never asked for a period after the T0 of the fit it serves; the post-period untreated
component is still reconstructed by stage 2 (blocks, stepwise matching, weights) from the site's own history.

Training, per sensor x fill x fit (T0 = 10 for the effect fit, 8 for the validation at n = 2, 7 at n = 3):
rows = every usable (site, period) with period <= T0, all sites (treated and control), minus 10 % of these
cells held out at random (seed 0) for early stopping; loss = MSE on the 980 outputs + 1e-3 * mean_i ||l_i||^2;
Adam lr 1e-3, weight decay 1e-4 on theta (not on the codes); batch 64; up to 5000 epochs, best epoch kept,
stop after 50 epochs without improvement of the held-out cells' MSE; seed 0. Every value is in CFG.

Outputs of train(): L_hat per site (T0 x 980, the generator's prediction for every period <= T0, including
held-out cells and months whose composite was unusable), the codes, the training curve and a description
row (epochs, best epoch, training and held-out MSE, held-out R^2 per channel, code norm, parameters, device,
seconds).
"""
import os, sys, time, copy

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_SHARED = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "ts_SCM_ASCM")   # panel_lib, panel_monthly, panel_align live there
if _SHARED not in _sys.path:
    _sys.path.insert(0, _SHARED)
import panel_lib as pl
import panel_monthly as pm

NCH, NPARCEL, D = 5, 196, 980
FITS = {"effect": 10, "val_n2": 8, "val_n3": 7}          # name -> T0 of the fit the generator serves
ARMS = ("mlp", "cnn")
CFG = dict(code_dim=32, period_dim=8, hidden=(256, 256), cnn_channels=32, lr=1e-3, weight_decay=1e-4,
           code_penalty=1e-3, batch=64, max_epochs=5000, patience=50, heldout_frac=0.1, seed=0)

_torch = None


def _lazy_torch(device=None):
    """Import torch after choosing the GPU (pl.pick_gpu must run before the first torch import)."""
    global _torch
    if _torch is None:
        if device != "cpu":
            try:
                pl.pick_gpu()
            except Exception:
                pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        _torch = torch
    return _torch


# ----------------------------------------------------------------- training cells
def cells(panel, sensor, T0):
    """Every usable (site, period) of the sensor with period <= T0, all sites, in a fixed order."""
    return [(s, q) for s in panel.sites(sensor) for q in range(1, T0 + 1) if panel.L(s, sensor, q) is not None]


def split_cells(cell_list, frac=CFG["heldout_frac"], seed=CFG["seed"]):
    """(training cells, held-out cells): a random 10 % of the cells, drawn once with the seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cell_list))
    k = int(round(frac * len(cell_list)))
    held = set(idx[:k].tolist())
    train = [c for i, c in enumerate(cell_list) if i not in held]
    heldout = [c for i, c in enumerate(cell_list) if i in held]
    return train, heldout


# ----------------------------------------------------------------- the network
def make_generator(n_sites, T0, arm, cfg=CFG, device="cpu"):
    torch = _lazy_torch(device)
    nn = torch.nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            self.codes = nn.Embedding(n_sites, cfg["code_dim"])
            self.periods = nn.Embedding(T0, cfg["period_dim"])
            nn.init.normal_(self.codes.weight, std=0.1)
            nn.init.normal_(self.periods.weight, std=0.1)
            h1, h2 = cfg["hidden"]
            self.trunk = nn.Sequential(nn.Linear(cfg["code_dim"] + cfg["period_dim"], h1), nn.ReLU(),
                                       nn.Linear(h1, h2), nn.ReLU())
            C = cfg["cnn_channels"]
            if arm == "mlp":
                self.head = nn.Linear(h2, D)
            elif arm == "cnn":
                self.to_grid = nn.Sequential(nn.Linear(h2, C * 7 * 7), nn.ReLU())
                self.up = nn.Sequential(nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                                        nn.Conv2d(C, NCH, kernel_size=3, padding=1))
                self.C = C
            else:
                raise ValueError(arm)

        def grid(self, site_idx, period_idx):
            """(B, 5, 14, 14) for the cnn arm; (B, 5, 14, 14) view of the mlp output otherwise."""
            z = self.trunk(torch.cat([self.codes(site_idx), self.periods(period_idx)], dim=1))
            if self.arm == "mlp":
                return self.head(z).view(-1, NCH, 14, 14)
            g = self.to_grid(z).view(-1, self.C, 7, 7)
            return self.up(g)

        def forward(self, site_idx, period_idx):
            return self.grid(site_idx, period_idx).reshape(-1, D)     # (channel, row, col) order = the cache's ravel

    return Generator().to(device)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------- training
def train(panel, sensor, arm, T0, cfg=CFG, device=None, log=print, tag=""):
    """Train f for one sensor and one fit (T0). Returns a dict with L_hat {site: (T0, D)}, codes {site: (code_dim,)},
    curve [(epoch, train_mse, held_mse)], desc (one description row), held_cells, train_cells."""
    torch = _lazy_torch(device)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    t_start = time.time()
    sites = panel.sites(sensor)
    site_index = {s: i for i, s in enumerate(sites)}
    all_cells = cells(panel, sensor, T0)
    tr_cells, ho_cells = split_cells(all_cells, cfg["heldout_frac"], cfg["seed"])
    n_filled = sum(1 for s in sites for q in range(1, T0 + 1) if panel.L(s, sensor, q) is None)

    def tensors(cl):
        S = torch.tensor([site_index[s] for s, _ in cl], dtype=torch.long, device=device)
        Q = torch.tensor([q - 1 for _, q in cl], dtype=torch.long, device=device)
        Y = torch.tensor(np.stack([panel.L(s, sensor, q) for s, q in cl]).astype(np.float32), device=device)
        return S, Q, Y
    S_tr, Q_tr, Y_tr = tensors(tr_cells)
    S_ho, Q_ho, Y_ho = tensors(ho_cells)

    model = make_generator(len(sites), T0, arm, cfg, device)
    code_params = list(model.codes.parameters())
    other_params = [p for n_, p in model.named_parameters() if not n_.startswith("codes.")]
    opt = torch.optim.Adam([{"params": other_params, "weight_decay": cfg["weight_decay"]},
                            {"params": code_params, "weight_decay": 0.0}], lr=cfg["lr"])
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    n_tr = len(tr_cells)
    best = {"held": np.inf, "epoch": -1, "state": None, "train": np.nan}
    curve, since_best, stop = [], 0, "max_epochs"
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(device)
        tot, cnt = 0.0, 0
        for b in range(0, n_tr, cfg["batch"]):
            idx = perm[b:b + cfg["batch"]]
            pred = model(S_tr[idx], Q_tr[idx])
            mse = torch.mean((pred - Y_tr[idx]) ** 2)
            pen = cfg["code_penalty"] * torch.mean(torch.sum(model.codes.weight ** 2, dim=1))
            loss = mse + pen
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(mse.item()) * len(idx); cnt += len(idx)
        model.eval()
        with torch.no_grad():
            held = float(torch.mean((model(S_ho, Q_ho) - Y_ho) ** 2).item()) if len(ho_cells) else np.nan
        tr_mse = tot / max(cnt, 1)
        curve.append((epoch, tr_mse, held))
        if held < best["held"] - 1e-12:
            best = {"held": held, "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "train": tr_mse}
            since_best = 0
        else:
            since_best += 1
            if since_best >= cfg["patience"]:
                stop = "patience"; break
    model.load_state_dict(best["state"]); model.eval()
    # predictions for every site and every period <= T0 (held-out and unusable months included)
    with torch.no_grad():
        S_all = torch.arange(len(sites), device=device).repeat_interleave(T0)
        Q_all = torch.arange(T0, device=device).repeat(len(sites))
        P = model(S_all, Q_all).cpu().numpy().astype(np.float64).reshape(len(sites), T0, D)
        pred_ho = model(S_ho, Q_ho).cpu().numpy() if len(ho_cells) else np.zeros((0, D))
        codes = model.codes.weight.detach().cpu().numpy()
    L_hat = {s: P[i] for s, i in site_index.items()}
    # description: held-out MSE and R^2 per channel (against the held-out cells' own mean per coordinate)
    Yho = Y_ho.cpu().numpy()
    res = (Yho - pred_ho) ** 2
    tot_ss = (Yho - Yho.mean(0, keepdims=True)) ** 2
    r2_ch = 1.0 - res.reshape(-1, NCH, NPARCEL).sum((0, 2)) / tot_ss.reshape(-1, NCH, NPARCEL).sum((0, 2))
    mse_ch = res.reshape(-1, NCH, NPARCEL).mean((0, 2))
    desc = {"sensor": sensor, "arm": arm, "fit": f"T0 = P{T0:02d}", "T0": T0, "n_sites": len(sites),
            "n_train": len(tr_cells), "n_held": len(ho_cells), "n_filled": n_filled, "params": n_params(model),
            "epochs_run": len(curve), "best_epoch": best["epoch"], "stop": stop,
            "train_mse": best["train"], "held_mse": best["held"],
            "r2_all": float(1.0 - res.sum() / tot_ss.sum()),
            **{f"r2_ch{c + 1}": float(v) for c, v in enumerate(r2_ch)},
            **{f"held_mse_ch{c + 1}": float(v) for c, v in enumerate(mse_ch)},
            "code_norm_mean": float(np.mean(np.linalg.norm(codes, axis=1))),
            "device": str(device), "seconds": round(time.time() - t_start, 1),
            **{f"cfg_{k}": (str(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}
    if log:
        log(f"  generator {tag}{sensor} {arm} T0=P{T0:02d}: {len(tr_cells)} rows, {len(ho_cells)} held out, "
            f"best epoch {best['epoch']}/{len(curve)} ({stop}), train MSE {best['train']:.5f}, held-out MSE {best['held']:.5f}, "
            f"R2 {desc['r2_all']:.3f}, {desc['seconds']} s on {device}")
    return {"sensor": sensor, "arm": arm, "T0": T0, "L_hat": L_hat, "codes": {s: codes[i] for s, i in site_index.items()},
            "curve": curve, "desc": desc, "held_cells": ho_cells, "train_cells": tr_cells}


def build(panel, fill, arm, cfg=CFG, device=None, log=print):
    """Train the three generators (T0 = 10, 8, 7) for both sensors of one cache.
    Returns {sensor: {T0: fit}}."""
    fits = {}
    for sensor in pm.SENSORS:
        fits[sensor] = {}
        for name, T0 in FITS.items():
            fits[sensor][T0] = train(panel, sensor, arm, T0, cfg, device, log, tag=f"{fill} ")
    return fits


def describe(fits_by_fill):
    """fits_by_fill: {fill: {sensor: {T0: fit}}} -> one row per fill x sensor x fit."""
    rows = []
    for fill, by_sensor in fits_by_fill.items():
        for sensor, by_T0 in by_sensor.items():
            for T0, f in by_T0.items():
                rows.append({"fill": fill, **f["desc"]})
    return pd.DataFrame(rows)


def curves(fits_by_fill):
    rows = []
    for fill, by_sensor in fits_by_fill.items():
        for sensor, by_T0 in by_sensor.items():
            for T0, f in by_T0.items():
                for epoch, tr, ho in f["curve"]:
                    rows.append({"fill": fill, "sensor": sensor, "arm": f["arm"], "T0": T0, "epoch": epoch,
                                 "train_mse": tr, "held_mse": ho})
    return pd.DataFrame(rows)


def save_lhat(fits_by_fill, path):
    """npz: key 'fill|sensor|T0|site' -> (T0, D) float32 predictions; 'codes|fill|sensor|T0|site' -> code."""
    out = {}
    for fill, by_sensor in fits_by_fill.items():
        for sensor, by_T0 in by_sensor.items():
            for T0, f in by_T0.items():
                for s, L in f["L_hat"].items():
                    out[f"{fill}|{sensor}|{T0}|{s}"] = L.astype(np.float32)
                    out[f"codes|{fill}|{sensor}|{T0}|{s}"] = f["codes"][s].astype(np.float32)
    np.savez_compressed(path, **out)


def load_lhat(path):
    z = np.load(path)
    L, codes = {}, {}
    for k in z.files:
        parts = k.split("|")
        if parts[0] == "codes":
            codes[(parts[1], parts[2], int(parts[3]), parts[4])] = z[k]
        else:
            L[(parts[0], parts[1], int(parts[2]), parts[3])] = z[k].astype(np.float64)
    return L, codes
