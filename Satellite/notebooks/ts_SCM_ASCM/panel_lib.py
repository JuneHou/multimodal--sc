"""panel_lib — shared functional module for the ts_SCM_ASCM biweekly-panel pipeline.

Extracted verbatim (2026-08-20) from the executed notebooks 01–04 of this folder so any
experiment can reuse the pipeline stages — kNN donor selection, SCM/ASCM weight fitting
(`panel_scm.py` / `panel_ascm.py`), decode-vs-ground-truth validation, effects, figures —
by swapping only the embedding fed in (a ``{(site, sensor, period_id): latent}`` dict).
Numbers produced from the original `latents_biweekly.npz` must reproduce the notebook
02/03 CSVs exactly (parity gate in notebook 05).

Import layout: numpy/pandas parts import at module load; everything needing torch /
terratorch sits behind ``pick_gpu()`` + ``load_tokenizers()`` / ``load_generator()`` so
the GPU is chosen before CUDA initializes.

Data gotcha carried from notebook 01: the biweekly tifs are CHANNEL-FIRST (C,101,101) —
``read_chip_biweekly`` transposes and asserts; anything reading these files must too.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------- fixed knobs
ROOT = Path("/data/wang/junh/githubs/latent-synthetic-control")
BIW = ROOT / "Satellite" / "data" / "biweekly_datasets"
TS = ROOT / "Satellite" / "notebooks" / "ts_SCM_ASCM"
LATD = ROOT / "Satellite" / "data" / "embeddings_tok_panel"

SENSORS = ("sentinel1", "sentinel2")
S1_BANDS = ["VV", "VH", "VV_minus_VH"]
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]
BANDS = {"sentinel1": S1_BANDS, "sentinel2": S2_BANDS}

TRAIN8 = list(range(1, 9))            # fit window w8 = P01–P08
TRAIN9 = list(range(1, 10))           # fit window w9 = P01–P09
POST = list(range(11, 21))            # effects P11–P20
J = 5                                 # donors per treated site
LAMBDAS = np.logspace(-4, 4, 81)      # ASCM ridge grid (collaborator notebook 15's)
TIMESTEPS, SEED = 50, 0               # tokenizer decode knobs
GEN_TIMESTEPS = 10                    # TerraMind generation timesteps (IBM examples)


def set_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.titlesize": 12,
    })


def pick_gpu():
    """Set CUDA_VISIBLE_DEVICES to the least-used GPU. Call BEFORE any torch import."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        free = sorted((int(u), int(i)) for i, u in
                      (l.split(", ") for l in q.strip().splitlines()))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    return os.environ["CUDA_VISIBLE_DEVICES"]


# ---------------------------------------------------------------- chip reading
def read_chip(path):
    a = tifffile.imread(path).astype(np.float32)
    a[~np.isfinite(a)] = np.nan
    return a


def read_chip_biweekly(path, sensor):
    """Biweekly tifs are CHANNEL-FIRST (C,101,101) — rasterio-written planar, unlike
    the finals chips' (101,101,C). Transpose to (H,W,C) so the verbatim `prep` /
    `feats_from_raw` band slicing is correct. Band order (writer's SENSOR_BANDS):
    S1 VV,VH,VV_minus_VH; S2 B2,B3,B4,B8,B11,B12,NDVI,NDWI."""
    a = read_chip(path)
    assert a.ndim == 3 and a.shape[0] == len(BANDS[sensor]), (path, a.shape)
    return np.moveaxis(a, 0, -1)


def fill_nan(x):
    for c in range(x.shape[0]):
        b = x[c]; m = np.isnan(b)
        if m.any():
            b[m] = np.nanmean(b) if not np.isnan(b).all() else 0.0
    return x


def feats_from_raw(sensor, chip):
    return {b: float(np.nanmean(chip[..., i])) for i, b in enumerate(BANDS[sensor])}


# ---------------------------------------------------------------- panel index
def load_roster():
    roster = pd.read_csv(ROOT / "Satellite" / "data" / "daily_datasets" /
                         "selected_site_sample.csv")
    assert len(roster) == 60 and roster["site_id"].is_unique
    return roster


def build_panel_index(roster=None, expected_files=2322):
    """One row per site × sensor × period (2,400). Local tif paths are CONSTRUCTED from
    the metadata columns (never the inventory's Dropbox `output_file` strings) and
    asserted to match the inventory's `composite_created` flag exactly."""
    if roster is None:
        roster = load_roster()
    inv = pd.read_csv(BIW / "biweekly_image_quality.csv")
    assert len(inv) == 2400, len(inv)
    idx = inv[["site_id", "group", "sensor", "period", "period_number", "period_id",
               "period_start", "period_end", "composite_created",
               "valid_pixel_fraction"]].copy()
    idx = idx.merge(roster[["site_id", "matched_treatment_site_id", "control_rank"]],
                    on="site_id", how="left", validate="many_to_one")
    idx["seq"] = idx["period_number"] + np.where(idx["period"] == "after", 10, 0)
    idx = idx.rename(columns={"period": "half"})
    idx["tif"] = [str(BIW / r.sensor / r.group / r.half /
                      f"{r.site_id}_{r.period_id}_{r.period_start}_{r.period_end}"
                      f"_biweekly_{r.sensor}.tif")
                  for r in idx.itertuples()]
    idx["file_exists"] = idx["tif"].map(lambda p: Path(p).exists())
    assert (idx["file_exists"] == (idx["composite_created"] == 1)).all(), \
        "constructed paths do not match inventory composite_created"
    if expected_files is not None:
        assert int(idx["file_exists"].sum()) == expected_files
    return idx


def load_latents(npz_path):
    """npz → {(site, sensor, period_id): float32 (5,14,14)}."""
    z = np.load(npz_path)
    return {tuple(k.split("|")): z[k] for k in z.files}


def save_latents(lat, npz_path, manifest=None):
    np.savez_compressed(npz_path, **{"|".join(k): np.asarray(v, dtype=np.float32)
                                     for k, v in lat.items()})
    if manifest is not None:
        Path(npz_path).with_name(Path(npz_path).stem + "_manifest.json").write_text(
            json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- Panel
class Panel:
    """Latents + index + roster + pooled scaler; every accessor the fits need.

    ``lat``: {(site, sensor, period_id): array (5,14,14) or flat} — any embedding
    source. Fits use float64 flattened copies; decode uses the float32 originals.
    Scaler: per sensor × dim, pooled over ALL sites, P01–P08 only, ddof=1, SD=0→1
    (identical construction to notebooks 02/03/04).
    """

    def __init__(self, lat, idx, roster):
        self.idx, self.roster = idx, roster
        self.treatments = sorted(roster.query("group == 'treatment'")["site_id"])
        # 50 sites with biweekly imagery — not the matching table's 260. kNN
        # iterates this list; 210 snapshot-era controls have no latent (README).
        self.controls = sorted(roster.query("group == 'counterfactual'")["site_id"])
        self.lat32 = {k: np.asarray(v, dtype=np.float32) for k, v in lat.items()}
        self.flat = {k: v.ravel().astype(np.float64) for k, v in self.lat32.items()}
        self.D = len(next(iter(self.flat.values())))
        seq_of = idx.set_index(["site_id", "sensor", "period_id"])["seq"].to_dict()
        self.pid_of_seq = {v: k[2] for k, v in seq_of.items()}
        self.vfrac = idx.set_index(["site_id", "sensor", "period_id"])[
            "valid_pixel_fraction"].to_dict()
        # observed-on-disk flag for the ground-truth side: generated latents (arm B)
        # have no raw image, so truth readers must check this, never have().
        # (panel_latent_index.csv has no file_exists column; there has_latent == it.)
        fcol = "file_exists" if "file_exists" in idx.columns else "has_latent"
        self.fexists = idx.set_index(["site_id", "sensor", "period_id"])[
            fcol].to_dict()
        self.meta = idx.set_index(["site_id", "sensor", "period_id"]).to_dict("index")
        self.scaler = {}
        for sensor in SENSORS:
            X = np.stack([self.flat[(s, sensor, self.pid_of_seq[q])]
                          for s in roster["site_id"] for q in TRAIN8
                          if (s, sensor, self.pid_of_seq[q]) in self.flat])
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=1)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            self.scaler[sensor] = (mu, sd)

    @classmethod
    def from_npz(cls, npz_path, idx_csv=None, roster=None):
        if roster is None:
            roster = load_roster()
        idx = pd.read_csv(idx_csv or TS / "panel_latent_index.csv")
        return cls(load_latents(npz_path), idx, roster)

    # -- accessors (verbatim semantics from notebooks 02/03) --
    def L(self, site, sensor, seq):
        """flattened float64 latent or None; seq 1..20."""
        return self.flat.get((site, sensor, self.pid_of_seq[seq]))

    def ok(self, site, sensor, seq, arm="main"):
        v = self.flat.get((site, sensor, self.pid_of_seq[seq]))
        if v is None:
            return False
        if arm == "quality40":
            f = self.vfrac.get((site, sensor, self.pid_of_seq[seq]), np.nan)
            return bool(np.isfinite(f) and f >= 0.40)
        return True

    def Z(self, site, sensor, seq):
        v = self.L(site, sensor, seq)
        if v is None:
            return None
        mu, sd = self.scaler[sensor]
        return (v - mu) / sd

    def usable(self, tid, sensor, seqs, dl, arm="main"):
        return [q for q in seqs
                if self.ok(tid, sensor, q, arm)
                and all(self.ok(c, sensor, q, arm) for c in dl)]

    def design(self, tid, sensor, seqs, dl):
        """flattened z-scored design: y (T*D,), X (T*D, J)."""
        y = np.stack([self.Z(tid, sensor, q) for q in seqs]).reshape(-1)
        X = np.stack([np.stack([self.Z(c, sensor, q) for q in seqs]).reshape(-1)
                      for c in dl]).T
        return y, X

    def rmse_of(self, w, tid, sensor, seqs, dl):
        """RMSPE in P01–P08 SD units: √mean((z − Xw)²) over periods × dims.
        On held-out seqs this is holdout RMSPE (ADH's RMSPE, z-scored). After
        that scaling, RMSPE of the pooled mean is 1 by construction."""
        y, X = self.design(tid, sensor, seqs, dl)
        r = y - X @ w
        return float(np.sqrt(np.mean(r * r)))

    def tif_path(self, site, sensor, seq):
        r = self.meta[(site, sensor, self.pid_of_seq[seq])]
        return (BIW / sensor / r["group"] / r["half"] /
                f"{site}_{self.pid_of_seq[seq]}_{r['period_start']}_{r['period_end']}"
                f"_biweekly_{sensor}.tif")

    def have(self, site, sensor, seq):
        return (site, sensor, self.pid_of_seq[seq]) in self.flat

    def observed(self, site, sensor, seq):
        """True only if the RAW image exists on disk (the ground-truth side)."""
        return bool(self.fexists.get((site, sensor, self.pid_of_seq[seq]), False))


# ---------------------------------------------------------------- donor selection
def knn_donors(panel, arms=("main",), J_=J):
    """5 nearest controls per treated site × sensor: distance = mean over shared
    available P01–P08 periods of the Euclidean distance between (unscaled) flattened
    latents. Search set = panel.controls (the 50 sites with biweekly imagery), not
    the snapshot-era 260: 210 of those have no P01–P20 chip, hence no latent (see
    this folder's README, "Limitation: kNN cannot search the 260-control snapshot
    pool"). main arm requires ≥4 shared periods per candidate; the quality40
    sensitivity arm relaxes to ≥2 and RECORDS not-estimable site × sensor cells.
    Returns (donor DataFrame, {arm: {(sensor, tid): [5 ids] or None}}, not_estimable).
    """
    cov5 = {t: sorted(panel.roster.query(
        "matched_treatment_site_id == @t and control_rank <= 5")["site_id"])
        for t in panel.treatments}
    don_rows, donors, not_estimable = [], {}, []
    for arm in arms:
        donors[arm] = {}
        min_shared = 4 if arm == "main" else 2
        for sensor in SENSORS:
            for tid in panel.treatments:
                scores = []
                for c in panel.controls:
                    sh = [q for q in TRAIN8 if panel.ok(tid, sensor, q, arm)
                          and panel.ok(c, sensor, q, arm)]
                    if len(sh) < min_shared:
                        continue
                    d = np.mean([np.linalg.norm(panel.L(tid, sensor, q) -
                                                panel.L(c, sensor, q)) for q in sh])
                    scores.append((d, c, len(sh)))
                scores.sort()
                if arm == "main":
                    assert len(scores) >= J_, (arm, sensor, tid, len(scores))
                if len(scores) < J_:
                    donors[arm][(sensor, tid)] = None
                    not_estimable.append({"arm": arm, "sensor": sensor,
                                          "treatment_site_id": tid,
                                          "n_candidates": len(scores)})
                    continue
                donors[arm][(sensor, tid)] = [c for _, c, _ in scores[:J_]]
                for rank, (d, c, nsh) in enumerate(scores[:J_], start=1):
                    don_rows.append({"arm": arm, "sensor": sensor,
                                     "treatment_site_id": tid,
                                     "counterfactual_site_id": c, "rank": rank,
                                     "distance": float(d), "n_shared_periods": nsh})
    don = pd.DataFrame(don_rows)
    don.attrs["cov5"] = cov5
    return don, donors, not_estimable


def donor_overlap(don, panel, arm="main"):
    """Per-sensor overlap of kNN donors with the site's 5 covariate-matched controls."""
    cov5 = don.attrs["cov5"]
    out = {}
    for sensor in SENSORS:
        sub = don.query("arm == @arm and sensor == @sensor")
        ov = [len(set(sub.query("treatment_site_id == @t")
                      ["counterfactual_site_id"]) & set(cov5[t]))
              for t in panel.treatments]
        out[sensor] = ov
    return out


# ---------------------------------------------------------------- validation flags
def flags(train, valid):
    """Notebook-14 ratio (≤1.5 good / ≤2.0 caution / else poor) plus whether
    holdout RMSPE beats the mean predictor (valid < 1 in SD units). After
    P01–P08 z-scoring, RMSPE of the pooled mean is 1 by construction. CSV
    column `flag_level` keeps the pass/FAIL label (not a named '1-SD test')."""
    ratio = valid / train if train > 0 else np.inf
    fr = "good" if ratio <= 1.5 else ("caution" if ratio <= 2.0 else "poor")
    return ratio, fr, ("pass" if valid < 1.0 else "FAIL")


# ================================================================ torch-side ====
class Tok:
    """TerraMind v1 tokenizers (frozen) + the verbatim embed_DiD/04 prep/encode/decode/
    feature machinery. Build via load_tokenizers() AFTER pick_gpu()."""

    def __init__(self):
        import torch
        import torch.nn.functional as F
        from terratorch.registry import FULL_MODEL_REGISTRY
        from terratorch.models.backbones.terramind.model.terramind_register import (
            PRETRAINED_BANDS, v1_pretraining_mean, v1_pretraining_std)
        self.torch, self.F = torch, F
        self.device = torch.device("cuda:0")
        self.tok = {
            "sentinel1": FULL_MODEL_REGISTRY.build(
                "terramind_v1_tokenizer_s1grd", pretrained=True
            ).to(self.device).eval(),
            "sentinel2": FULL_MODEL_REGISTRY.build(
                "terramind_v1_tokenizer_s2l2a", pretrained=True
            ).to(self.device).eval()}
        for m_ in self.tok.values():
            for p_ in m_.parameters():
                p_.requires_grad_(False)
        self.M = {"sentinel1": np.array(v1_pretraining_mean["tok_sen1grd@224"],
                                        dtype=np.float32),
                  "sentinel2": np.array(v1_pretraining_mean["tok_sen2l2a@224"],
                                        dtype=np.float32)}
        self.SD = {"sentinel1": np.array(v1_pretraining_std["tok_sen1grd@224"],
                                         dtype=np.float32),
                   "sentinel2": np.array(v1_pretraining_std["tok_sen2l2a@224"],
                                         dtype=np.float32)}
        self.ALL12 = PRETRAINED_BANDS["untok_sen2l2a@224"]
        self.OUR6 = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
        self.IDX12 = [self.ALL12.index(b) for b in self.OUR6]

    def prep(self, sensor, chip):
        """chip (H,W,C) raw native units -> standardized tokenizer input (C_tok,224,224).
        S1: VV,VH dB (first 2 channels). S2: first 6 reflectance channels ×10⁴ into the
        12-band slots IDX12, missing slots at the pretraining mean."""
        torch, F = self.torch, self.F
        if sensor == "sentinel1":
            x = torch.from_numpy(fill_nan(np.ascontiguousarray(
                chip[..., :2].transpose(2, 0, 1))))[None]
            x = F.interpolate(x, size=(224, 224), mode="bilinear",
                              align_corners=False)
            m, s = self.M[sensor], self.SD[sensor]
            x = (x - torch.tensor(m)[None, :, None, None]) / \
                torch.tensor(s)[None, :, None, None]
            return x[0]
        x6 = torch.from_numpy(fill_nan(np.ascontiguousarray(
            chip[..., :6].transpose(2, 0, 1))))[None] * 10_000.0
        x6 = F.interpolate(x6, size=(224, 224), mode="bilinear", align_corners=False)
        x12 = torch.zeros(1, 12, 224, 224)
        m, s = self.M["sentinel2"], self.SD["sentinel2"]
        for ci, jx in enumerate(self.IDX12):
            x12[0, jx] = (x6[0, ci] - m[jx]) / s[jx]
        return x12[0]

    def prep_s2_dn224(self, x6_dn):
        """Generated S2 chip, 6 project bands ALREADY in DN units (reflectance ×10⁴)
        at 224×224 (torch or numpy (6,224,224)) -> standardized 12-band input.
        Skips the 101→224 resize and the ×10⁴ (generation output is already DN@224)."""
        torch = self.torch
        x6 = torch.as_tensor(np.asarray(x6_dn, dtype=np.float32))
        assert x6.shape == (6, 224, 224), tuple(x6.shape)
        x12 = torch.zeros(12, 224, 224)
        m, s = self.M["sentinel2"], self.SD["sentinel2"]
        for ci, jx in enumerate(self.IDX12):
            x12[jx] = (x6[ci] - m[jx]) / s[jx]
        return x12

    def prep_s1_db224(self, x2_db):
        """Generated S1 chip, VV,VH ALREADY in dB at 224×224 ((2,224,224)) ->
        standardized tokenizer input. Skips the 101→224 resize (generation output
        is already dB@224)."""
        torch = self.torch
        x = torch.as_tensor(np.asarray(x2_db, dtype=np.float32))
        assert x.shape == (2, 224, 224), tuple(x.shape)
        m, s = self.M["sentinel1"], self.SD["sentinel1"]
        return (x - torch.tensor(m)[:, None, None]) / torch.tensor(s)[:, None, None]

    def encode_batch(self, sensor, tensors):
        torch = self.torch
        with torch.no_grad():
            q, _, _ = self.tok[sensor].encode(torch.stack(tensors).to(self.device))
        return q.cpu()

    def decode_batch(self, sensor, quants_t, batch=16):
        torch = self.torch
        outs = []
        with torch.no_grad():
            for s0 in range(0, len(quants_t), batch):
                gen = torch.Generator().manual_seed(SEED)
                outs.append(self.tok[sensor].decode_quant(
                    quants_t[s0:s0 + batch].to(self.device), timesteps=TIMESTEPS,
                    generator=gen, image_size=(224, 224)).cpu())
        return torch.cat(outs)

    def feats_from_decoded(self, sensor, dec):
        m, s = self.M[sensor], self.SD[sensor]
        if sensor == "sentinel1":
            vv = dec[0] * s[0] + m[0]
            vh = dec[1] * s[1] + m[1]
            return {"VV": float(vv.mean()), "VH": float(vh.mean()),
                    "VV_minus_VH": float((vv - vh).mean())}
        refl = {b: (dec[jx] * s[jx] + m[jx]) / 10_000.0
                for b, jx in zip(self.OUR6, self.IDX12)}
        ndvi = (refl["NIR_BROAD"] - refl["RED"]) / (refl["NIR_BROAD"] + refl["RED"])
        ndwi = (refl["GREEN"] - refl["NIR_BROAD"]) / (refl["GREEN"] + refl["NIR_BROAD"])
        out = {b2: float(refl[b].mean()) for b2, b in
               zip(["B2", "B3", "B4", "B8", "B11", "B12"], self.OUR6)}
        out["NDVI"] = float(ndvi.mean())
        out["NDWI"] = float(ndwi.mean())
        return out

    def native_bands(self, sensor, dec):
        """decoded standardized (C_tok,224,224) -> native-unit band stack at 101×101."""
        torch, F = self.torch, self.F
        m, s = self.M[sensor], self.SD[sensor]
        if sensor == "sentinel1":
            vv = dec[0] * s[0] + m[0]; vh = dec[1] * s[1] + m[1]
            stack = torch.stack([vv, vh, vv - vh])
        else:
            refl = {b: (dec[jx] * s[jx] + m[jx]) / 10_000.0
                    for b, jx in zip(self.OUR6, self.IDX12)}
            ndvi = (refl["NIR_BROAD"] - refl["RED"]) / (refl["NIR_BROAD"] + refl["RED"])
            ndwi = (refl["GREEN"] - refl["NIR_BROAD"]) / \
                   (refl["GREEN"] + refl["NIR_BROAD"])
            stack = torch.stack([refl["BLUE"], refl["GREEN"], refl["RED"],
                                 refl["NIR_BROAD"], refl["SWIR_1"], refl["SWIR_2"],
                                 ndvi, ndwi])
        return F.interpolate(stack[None], size=(101, 101), mode="bilinear",
                             align_corners=False)[0].numpy()


def load_tokenizers():
    return Tok()


class Gen:
    """TerraMind v1 generation model wrapper — EXACTLY the IBM repo recipe
    (github.com/IBM/terramind, notebooks/terramind_generation.ipynb):
    FULL_MODEL_REGISTRY.build('terramind_v1_<variant>_generate', modalities=[inp],
    output_modalities=[out], pretrained=True, standardize=True) and
    model(input, timesteps=10) on raw physical-unit tensors [B,C,224,224]
    (S1 VV,VH in dB; S2 12-band DN). Output destandardized to physical units.
    Only additions (Jun-approved 2026-08-20): inputs bilinearly resized 101→224,
    and torch.manual_seed before each call for reproducibility."""

    def __init__(self, variant="terramind_v1_large_generate",
                 inp="S1GRD", out="S2L2A"):
        import torch
        from terratorch.registry import FULL_MODEL_REGISTRY
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.inp, self.out = inp, out
        self.model = FULL_MODEL_REGISTRY.build(
            variant, modalities=[inp], output_modalities=[out],
            pretrained=True, standardize=True).to(self.device).eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)

    def generate(self, x, timesteps=GEN_TIMESTEPS, seed=SEED):
        """x: torch [B,C,224,224] raw physical units. Returns [B,C_out,224,224]
        physical units (destandardized by standardize=True).
        The TerraMind sampler draws its seed from Python's `random` module
        (terramind_generation.py forward), so BOTH random and torch are seeded."""
        import random as _random
        torch = self.torch
        _random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            gen = self.model(x.to(self.device), timesteps=timesteps)
        if isinstance(gen, dict):
            assert len(gen) == 1, list(gen)
            gen = next(iter(gen.values()))
        return gen.cpu()


def load_generator(variant="terramind_v1_large_generate", inp="S1GRD", out="S2L2A"):
    return Gen(variant=variant, inp=inp, out=out)


# ================================================================ decode-vs-truth
def weights_dict(wdf, donors, arm="main"):
    """weights DataFrame -> {(sensor, tid, fit_window): w (J,) float32} in donor order."""
    out = {}
    for (sensor, tid, win), g in wdf.query("arm == @arm").groupby(
            ["sensor", "treatment_site_id", "fit_window"]):
        order = donors[arm][(sensor, tid)]
        out[(sensor, tid, win)] = g.set_index("donor").loc[order, "weight"].to_numpy(
            dtype=np.float32)
    return out


def synth_latent(panel, tokb, donors_arm, W_method, sensor, tid, seq):
    """weighted RAW latent tensor for period seq; w8 for P09 & post, w9 for P10.
    None if any donor lacks that period's latent."""
    torch = tokb.torch
    dl = donors_arm[(sensor, tid)]
    if not all(panel.have(c, sensor, seq) for c in dl):
        return None
    win = "P01-09" if seq == 10 else "P01-08"
    w = W_method[(sensor, tid, win)]
    stack = torch.stack([torch.from_numpy(
        panel.lat32[(c, sensor, panel.pid_of_seq[seq])]) for c in dl])
    return torch.tensordot(torch.from_numpy(w), stack, dims=1)


def decode_all(panel, tokb, donors_arm, W, sensors=SENSORS, post=POST):
    """Decode synthetic counterfactuals (P09, P10, post periods; methods in W) plus the
    reconstruction floor (treated P09/P10 latents decoded back). Returns dec_of:
    {(method, sensor, tid, seq): decoded (C_tok,224,224)}."""
    torch = tokb.torch
    jobs, dec_of = [], {}
    for sensor in sensors:
        for tid in panel.treatments:
            for method in W:
                for seq in [9, 10] + list(post):
                    sl = synth_latent(panel, tokb, donors_arm, W[method],
                                      sensor, tid, seq)
                    if sl is not None:
                        jobs.append((method, sensor, tid, seq, sl))
            for seq in (9, 10):     # floor: observed latent decoded straight back
                if panel.have(tid, sensor, seq):
                    jobs.append(("floor", sensor, tid, seq, torch.from_numpy(
                        panel.lat32[(tid, sensor, panel.pid_of_seq[seq])])))
    print("decode jobs:", len(jobs), "| by kind:",
          pd.Series([j[0] for j in jobs]).value_counts().to_dict())
    for sensor in sensors:
        sj = [j for j in jobs if j[1] == sensor]
        if not sj:
            continue
        dec = tokb.decode_batch(sensor, torch.stack([j[4] for j in sj]))
        for j_, d_ in zip(sj, dec):
            dec_of[(j_[0], sensor, j_[2], j_[3])] = d_
        print(f"{sensor}: {len(sj)} decoded")
    return dec_of


def decode_validation_table(panel, tokb, donors_arm, W, dec_of, sensors=SENSORS):
    """Three-space P09/P10 validation vs observed ground truth, next to the floor.
    Ground truth ALWAYS from the raw observed chip (nan-aware features; image RMSE over
    its valid pixels only) — generated/filled pixels never enter the truth side."""
    rows = []
    for sensor in sensors:
        mu, sd = panel.scaler[sensor]
        for tid in panel.treatments:
            for seq in (9, 10):
                if not (panel.have(tid, sensor, seq)
                        and panel.observed(tid, sensor, seq)):
                    continue
                obs_lat = panel.flat[(tid, sensor, panel.pid_of_seq[seq])]
                chip = read_chip_biweekly(panel.tif_path(tid, sensor, seq), sensor)
                f_obs = feats_from_raw(sensor, chip)
                valid = np.isfinite(chip)
                period = f"P{seq:02d}"
                for method in list(W) + ["floor"]:
                    d = dec_of.get((method, sensor, tid, seq))
                    if d is None:
                        continue
                    if method != "floor":
                        sl = synth_latent(panel, tokb, donors_arm, W[method],
                                          sensor, tid, seq).numpy().ravel()
                        rows.append({"sensor": sensor, "treatment_site_id": tid,
                                     "period": period, "method": method,
                                     "space": "latent", "band": "all",
                                     "value": float(np.sqrt(np.mean(
                                         ((sl - obs_lat) / sd) ** 2)))})
                    f_dec = tokb.feats_from_decoded(sensor, d)
                    for b in BANDS[sensor]:
                        rows.append({"sensor": sensor, "treatment_site_id": tid,
                                     "period": period, "method": method,
                                     "space": "feature", "band": b,
                                     "value": f_dec[b] - f_obs[b]})
                    nb_ = tokb.native_bands(sensor, d)
                    for bi, b in enumerate(BANDS[sensor]):
                        m_ = valid[..., bi]
                        if m_.any():
                            e = nb_[bi][m_] - chip[..., bi][m_]
                            rows.append({"sensor": sensor,
                                         "treatment_site_id": tid,
                                         "period": period, "method": method,
                                         "space": "image", "band": b,
                                         "value": float(np.sqrt(np.mean(e * e)))})
    return pd.DataFrame(rows)


def effect_features_table(panel, tokb, dec_of, methods=("scm", "ascm"),
                          sensors=SENSORS, post=POST):
    """P11–P20 effects in feature units: observed raw features − decoded synthetic."""
    rows = []
    for sensor in sensors:
        for tid in panel.treatments:
            for seq in post:
                period = f"P{seq}"
                f_obs = None
                if panel.observed(tid, sensor, seq):
                    f_obs = feats_from_raw(sensor, read_chip_biweekly(
                        panel.tif_path(tid, sensor, seq), sensor))
                for method in methods:
                    d = dec_of.get((method, sensor, tid, seq))
                    f_syn = tokb.feats_from_decoded(sensor, d) if d is not None else None
                    for b in BANDS[sensor]:
                        o = f_obs[b] if f_obs else np.nan
                        s_ = f_syn[b] if f_syn else np.nan
                        rows.append({"sensor": sensor, "treatment_site_id": tid,
                                     "period": period, "method": method, "band": b,
                                     "observed": o, "synthetic_decoded": s_,
                                     "effect": o - s_})
    return pd.DataFrame(rows)


# ================================================================ figures
def plot_validation_bars(val, panel, path, title_note="SCM", sensors=SENSORS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, sensor in zip(axes, sensors):
        sub = val.query("arm == 'main' and sensor == @sensor")
        x = np.arange(len(panel.treatments))
        fj = sub.query("scheme == 'frozen_joint'").set_index(
            "treatment_site_id").loc[panel.treatments, "valid_rmse"]
        e9 = sub.query("scheme == 'expanding' and eval_period == 'P09'").set_index(
            "treatment_site_id").loc[panel.treatments, "valid_rmse"]
        e10 = sub.query("scheme == 'expanding' and eval_period == 'P10'").set_index(
            "treatment_site_id").loc[panel.treatments, "valid_rmse"]
        ax.bar(x - 0.27, fj, 0.25, label="frozen joint P09+P10 (notebook-14 scheme)")
        ax.bar(x, e9, 0.25, label="expanding: P09")
        ax.bar(x + 0.27, e10, 0.25, label="expanding: P10")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.text(0.02, 1.02, "mean baseline", transform=ax.get_yaxis_transform(),
                fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([t[-2:] for t in panel.treatments])
        ax.set_xlabel("treatment site")
        ax.set_title(f"{sensor} — {title_note} holdout RMSPE")
        ax.set_ylabel("holdout RMSPE (SD units)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_effect_trajectories(traj, panel, path, title_note="SCM", sensors=SENSORS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax, sensor in zip(axes, sensors):
        for tid in panel.treatments:
            s = traj[(sensor, tid)]
            ax.plot(list(s.keys()), list(s.values()), marker="o", ms=2.5, lw=0.9,
                    label=tid[-2:])
        ax.axvline(10.5, color="k", lw=1.0, ls="--")
        ax.text(10.6, 0.95, "hurricane", transform=ax.get_xaxis_transform(),
                fontsize=7)
        ax.set_title(f"{sensor} — latent gap (treated − {title_note} synthetic), "
                     "w8 frozen")
        ax.set_ylabel("gap norm (pooled-SD units)")
    axes[1].set_xlabel("period (1–10 pre, 11–20 post)")
    axes[0].legend(ncol=10, frameon=False, fontsize=6, title="site",
                   title_fontsize=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_scm_vs_ascm(cmp_, path, sensors=SENSORS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharex=False)
    for ax, sensor in zip(axes, sensors):
        t = cmp_.query("arm == 'main' and scheme == 'frozen_joint' and "
                       "sensor == @sensor")
        lim = max(t["valid_rmse_scm"].max(), t["valid_rmse_ascm"].max()) * 1.08
        ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls="--")
        ax.scatter(t["valid_rmse_scm"], t["valid_rmse_ascm"], s=28)
        for _, r in t.iterrows():
            ax.annotate(r["treatment_site_id"][-2:],
                        (r["valid_rmse_scm"], r["valid_rmse_ascm"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("SCM holdout RMSPE (SD units)")
        ax.set_ylabel("ASCM holdout RMSPE (SD units)")
        ax.set_title(f"{sensor} — below the line = ASCM better")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_lambda(lam_tab, path, sensors=SENSORS):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    lt = lam_tab.query("arm == 'main'")
    for sensor, marker in zip(sensors, "os"):
        s = lt.query("sensor == @sensor")
        ax.scatter(np.arange(len(s)), s["lambda"], marker=marker, s=24, label=sensor)
    ax.set_yscale("log")
    ax.axhline(LAMBDAS[0], color="k", lw=0.6, ls=":")
    ax.axhline(LAMBDAS[-1], color="k", lw=0.6, ls=":")
    ax.set_ylabel("chosen lambda (log)")
    ax.set_xlabel("fit (site x window)")
    ax.set_title("ASCM ridge penalty chosen by leave-one-period-out CV (main arm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_decode_features(dv, path, methods=("scm", "ascm"), sensors=SENSORS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, sensor in zip(axes, sensors):
        bands = BANDS[sensor]
        x = np.arange(len(bands))
        sub = (dv.query("space == 'feature' and sensor == @sensor")
               .assign(a=lambda t: t["value"].abs())
               .pivot_table(index="band", columns="method", values="a").loc[bands])
        offs = np.linspace(-0.27, 0.27, len(methods) + 1)
        for o, method in zip(offs, methods):
            ax.bar(x + o, sub[method], 0.25, label=f"{method.upper()} synthetic")
        ax.bar(x + offs[-1], sub["floor"], 0.25, label="reconstruction floor",
               color="0.55")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=30, ha="right")
        ax.set_title(f"{sensor} — held-out P09/P10, decoded vs observed features")
        ax.set_ylabel("mean |error| (native units)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_site_trajectories(panel, tokb, dec_of, sensor, band, path,
                           methods=("scm", "ascm")):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharex=True)
    for ax, tid in zip(axes.ravel(), panel.treatments):
        xs, obs = [], []
        for seq in range(1, 21):
            if panel.observed(tid, sensor, seq):
                f = feats_from_raw(sensor, read_chip_biweekly(
                    panel.tif_path(tid, sensor, seq), sensor))
                xs.append(seq)
                obs.append(f[band])
        ax.plot(xs, obs, color="k", marker="o", ms=2.5, lw=1.0, label="observed")
        for method, color in zip(methods, ("C0", "C1")):
            xs2 = [s for s in range(9, 21)
                   if dec_of.get((method, sensor, tid, s)) is not None]
            ys2 = [tokb.feats_from_decoded(sensor,
                                           dec_of[(method, sensor, tid, s)])[band]
                   for s in xs2]
            ax.plot(xs2, ys2, color=color, marker="s", ms=2.5, lw=1.0,
                    label=f"{method.upper()} synthetic (decoded)")
        ax.axvline(10.5, color="k", lw=0.8, ls="--")
        ax.set_title(tid, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=6)
    for ax in axes[1]:
        ax.set_xlabel("period")
    fig.suptitle(f"{sensor} {band}: observed vs decoded synthetic counterfactual "
                 "(dashed line = hurricane; P09–P10 = held-out validation)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig
