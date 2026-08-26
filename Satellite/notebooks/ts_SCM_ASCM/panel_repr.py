"""panel_repr — pooled outcome representations of the TerraMind latent, in the
structure Okano & Kurisu's FSC code expects.

Motivation (measured, see `Satellite/Docs/8-27-update.md`): the raw 980-d latent is a
position-indexed grid — coordinate (c, r, col) is "detector c of the parcel at row r,
column col OF THIS CHIP". Matching it across sites matches one site's parking lot to
another's forest. On clean chips the token is a strong LOCAL descriptor of its own parcel
(within-chip R² 0.65–0.82 vs NDVI/NDWI/NDBI), which is exactly why position-matching is
the wrong comparison and why pooling — discarding the spatial index — is the fix.

Three pooled representations, each an isometric embedding into a Hilbert space so that
Okano–Kurisu's FSC applies (their examples 2 and 3):

    chip_mean            5  channel means                      (baseline, not FSC)
    quantile_functions 500  per-channel quantile function on their `grids`, concatenated
                            over the 5 channels — 2-Wasserstein / their mortality.R case
    gram_vector         15  upper triangle incl. diagonal of A Aᵀ/196 — their `covvec`
                            convention exactly (service.R L92-95), Frobenius / their
                            example 3
    combined           515  quantile ⊕ gram, each block scaled to unit mean square

What each answers, concretely: the quantile arm says *how much of each thing is present*
(land-cover composition); the Gram arm says *what fires together at the same parcel*
(surface co-occurrence — a marsh and a forest-beside-a-lake have the same histogram but
different Gram). Neither says *where*.

ADAPTATION, flagged: their code carries one scalar-valued function per unit-period; we
have 5 latent channels. Concatenating the 5 quantile functions is the direct sum of
Hilbert embeddings (a valid isometry under the product metric √Σ_c d_c²), and `combined`
is a second direct sum. Both are OUR extension of their setup, not something their
scripts do.
"""
import numpy as np

from panel_lib import SENSORS, TRAIN8

# Quantile grid. Their mortality.R uses seq(0.01, 0.99, length=100) for ONE function;
# we concatenate 5 channels, so 100 points would give M = 500 and their augmented
# estimator costs O((M·T_0)²) — measured `cross_val_covmat`: 6.5 s at M=45, 33 s at
# M=100, 218 s at M=500. Plain `FSCM` is free at any M, so the operating point was
# chosen by measuring where the FSC result converges (notebook grid-resolution gate):
#
#     pts/channel   M    mean holdout ratio_own_mean (FSC)   max |weight diff| vs 100pt
#          5        25              0.9991                          0.220
#          9        45              0.9601                          0.136
#         20       100              0.9575                          0.044
#        100       500              0.9510                          –
#
# 20 points/channel is converged (0.9575 vs 0.9510) and affordable; the 100-point grid
# is kept as GRIDS_MORTALITY for the FSC-only fidelity check.
GRIDS_MORTALITY = np.linspace(0.01, 0.99, 100)     # their grid, mortality.R L20
GRIDS_Q = np.linspace(0.01, 0.99, 20)
# Upper triangle INCLUDING diagonal, in R's COLUMN-MAJOR order — the order
# `m[upper.tri(m, diag = TRUE)]` produces in service.R L92-95. numpy's np.triu_indices is
# ROW-major; using it here while fsc_bridge.R reassembles with upper.tri() silently
# scrambled the matrix (it round-trips consistently, so plain FSC/aFSC fits were
# unaffected — but the nearPD projection was then "repairing" a matrix that had only been
# mis-assembled). Build it column-major so both sides agree and the convention is theirs.
_iu = [(r, c) for c in range(5) for r in range(c + 1)]
IU5 = (np.array([r for r, _ in _iu]), np.array([c for _, c in _iu]))
NPARCEL = 196

REPRS = ("chip_mean", "quantile", "gram", "combined")


def dim_of(name, grids=GRIDS_Q):
    """Outcome dimension M. The quantile/combined arms depend on the grid resolution,
    which is a knob: their code costs O((M·T_0)²) in the AUGMENTED estimator (measured:
    `cross_val_covmat` takes 6.5 s at M=45 and 218 s at M=500), while plain `FSCM` is
    free at any M. See the notebooks' grid-resolution gate."""
    nq = 5 * len(grids)
    return {"chip_mean": 5, "quantile": nq, "gram": len(IU5[0]),
            "combined": nq + len(IU5[0])}[name]


DIMS = {n: dim_of(n) for n in REPRS}


def matched_donors(panel):
    """{treated site: its 5 covariate-matched controls} — the collaborator's
    notebook-14 matched design (`control_rank <= 5`), NOT embedding kNN."""
    r = panel.roster
    out = {}
    for t in panel.treatments:
        d = sorted(r.query("matched_treatment_site_id == @t and control_rank <= 5")
                   ["site_id"])
        assert len(d) == 5, (t, len(d))
        out[t] = d
    return out


# ---------------------------------------------------------------- representations
def _A(panel, site, sensor, seq):
    """(5, 196) view of a chip's latent, or None."""
    v = panel.L(site, sensor, seq)
    return None if v is None else v.reshape(5, NPARCEL)


def chip_mean(A):
    """5 channel means over the 196 parcels. The baseline: §7 of the 8-27 doc found
    this already recovers chip-mean NDVI at R² 0.917, so richer summaries must beat it."""
    return A.mean(axis=1)


def quantile_functions(A, grids=GRIDS_Q):
    """Per channel, the quantile function evaluated on `grids`, concatenated.

    Ψ(μ) = Fμ⁻¹ is the isometric embedding of a 1-D distribution into L²([0,1]) under
    the 2-Wasserstein metric (their example 2), so L² distance between these vectors IS
    W₂ distance between the parcel-value distributions, and a simplex-weighted average
    is the Wasserstein barycenter.

    Note the FSQ lattice has only 8/8/8/6/5 levels, so each quantile function is a step
    function with at most 8 jumps — asking for more than ~8 grid points adds resolution
    the encoder does not have. We keep their 100-point grid anyway for fidelity.
    """
    return np.concatenate([np.quantile(A[c], grids) for c in range(A.shape[0])])


def gram_vector(A):
    """Upper triangle incl. diagonal of A Aᵀ / 196 — their `covvec` (service.R L92-95).

    G[a,b] = mean over the 196 parcels of (detector a x detector b): do detectors a and b
    light up on the SAME parcels? SPSD matrices form a closed convex cone, so a
    simplex-weighted average is still a valid covariance (their example 3); augmented
    weights can leave the cone and are projected back with Matrix::nearPD.
    """
    return (A @ A.T / A.shape[1])[IU5]


def gram_to_matrix(v, n=5):
    """Inverse of `gram_vector` — their reconstruction, service.R L92-95:
    fill the upper triangle (COLUMN-major, as R does), add the transpose, subtract half
    the doubled diagonal."""
    m = np.zeros((n, n))
    iu = ([r for c in range(n) for r in range(c + 1)],
          [c for c in range(n) for r in range(c + 1)])
    m[iu] = v
    m = m + m.T
    return m - np.diag(np.diag(m) / 2.0)


def _repr_raw(A, name, grids=GRIDS_Q):
    if name == "chip_mean":
        return chip_mean(A)
    if name == "quantile":
        return quantile_functions(A, grids)
    if name == "gram":
        return gram_vector(A)
    if name == "combined":
        return np.concatenate([quantile_functions(A, grids), gram_vector(A)])
    raise ValueError(name)


# ---------------------------------------------------------------- panel assembly
def complete_periods(panel, sensor, groups, periods=range(1, 21)):
    """Periods for which EVERY unit of EVERY group has a latent. Their func_vals_list
    requires a complete rectangular panel; this is how we get one."""
    return [q for q in periods
            if all(panel.have(s, sensor, q) for g in groups for s in g)]


def complete_groups(panel, sensor, groups, need=range(1, 11)):
    """Groups whose every unit has every period in `need` (default P01-P10, the fit +
    holdout window). Returns (kept, dropped) so exclusions are recorded, never silent."""
    kept, dropped = [], []
    for g in groups:
        (kept if all(panel.have(s, sensor, q) for s in g for q in need)
         else dropped).append(g)
    return kept, dropped


def build_groups(panel):
    """[[treated, c1..c5], ...] — treated FIRST, matching their index-1 convention
    (main_functions.R L41-45 makes unit 1 the treated `d_vec`)."""
    don = matched_donors(panel)
    return [[t] + don[t] for t in panel.treatments]


def build_block(panel, sensor, name, groups, periods, grids=GRIDS_Q,
                block_scale=None):
    """Outcome block of shape (n_groups, n_units, n_periods, M), C-order, float64 —
    the array `fsc_bridge.R` reads and reshapes into their `func_vals_list`.

    `block_scale` (combined arm only): (s_quantile, s_gram) divisors so neither block
    dominates the L² objective. Computed ONCE over the whole block, so the map stays a
    linear isometry rather than a per-chip renormalisation.
    """
    M = dim_of(name, grids)
    out = np.zeros((len(groups), len(groups[0]), len(periods), M), dtype=np.float64)
    for gi, g in enumerate(groups):
        for ui, site in enumerate(g):
            for ti, q in enumerate(periods):
                A = _A(panel, site, sensor, q)
                assert A is not None, (site, sensor, q)
                out[gi, ui, ti] = _repr_raw(A, name, grids)
    if name == "combined":
        nq = 5 * len(grids)
        if block_scale is None:
            sq = float(np.sqrt((out[..., :nq] ** 2).mean()))
            sg = float(np.sqrt((out[..., nq:] ** 2).mean()))
            block_scale = (sq or 1.0, sg or 1.0)
        out[..., :nq] /= block_scale[0]
        out[..., nq:] /= block_scale[1]
    return out, block_scale


def describe(panel, sensor, name, groups, periods):
    """One-line provenance string for the notebook."""
    return (f"{name}: M={DIMS[name]}  groups={len(groups)}  units/group={len(groups[0])}"
            f"  periods={len(periods)} {periods}  sensor={sensor}")
