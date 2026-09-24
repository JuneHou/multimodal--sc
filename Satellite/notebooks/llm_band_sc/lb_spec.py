"""lb_spec — the SHC design specification a model fills in. There are no defaults anywhere in this module's
validation: a field the caller leaves out is an error that names the field, never a silent fall-back.

Class 1: settings her code already takes (applied by rebinding her module names, lb_her_run.class1_overrides).
Class 2: choices her code makes one way with no branch; the alternatives are new code in lb_fit.
HER_SPEC: her settings, for tests and for lb_shc.run_all only. lb_fit / lb_sandbox / lb_spec_prompts may not
import it (gate G6) — the fit code must never know what "her value" is.

    validate_spec(spec, sensor)      -> normalized copy, or SpecError listing every violation with its field
    class1_only(spec)                -> the same for arm A, refusing any Class-2 key as a protocol violation
    schema_text(class2=True)         -> the field list for a prompt: names, allowed values, meaning; no value filled
"""
import json

import lb_lib as lb

N_TRAIN = 9                                  # P01-P09; the training span is not a field


class SpecError(ValueError):
    pass


# name -> (type, allowed / predicate, one-line meaning for the prompt)
CLASS1 = {
    "m": ("int", lambda v: v >= 1,
          "integer >= 1: length in months of the matching window and of every historical block"),
    "n": ("int", lambda v: v >= 1,
          "integer >= 1: how many months after each historical block that block is taken to predict; the first of them is used"),
    "kernel_bandwidth": ("float", lambda v: v > 0,
                         "number > 0: Gaussian bandwidth, in months, of the local-linear smoother that estimates each band's latent trajectory"),
    "zeta": ("float", lambda v: v >= 0,
             "number >= 0: in stepwise selection a further block is kept only if it lowers the matching MSE by more than this"),
    "beta_trend_degree": ("int", lambda v: v in (0, 1, 2),
                          "0, 1 or 2: degree of the time trend partialled out when the annual sine/cosine seasonal terms are estimated"),
}
CLASS2 = {
    "weight_constraint": ("enum", ("simplex", "nonneg", "unconstrained"),
                          "simplex = weights >= 0 summing to 1; nonneg = weights >= 0, any sum; unconstrained = any real weights"),
    "intercept": ("enum", ("none", "fitted"),
                  "none = no constant term; fitted = a constant is fitted alongside the block weights and added to the prediction"),
    "band_weighting": ("enum", ("equal", "inverse_variance", "subset"),
                       "equal = every band's rows enter the joint fit with weight 1; inverse_variance = each band's rows scaled by 1/SD of that band over P01-P09 in the fitting space; subset = fit on the bands listed in band_subset only (all bands are still predicted)"),
    "fit_space": ("enum", ("latent_std", "raw"),
                  "latent_std = weights fitted on each band's smoothed, deseasonalized, standardized trajectory; raw = weights fitted on the observed band values"),
    "readd_seasonal": ("bool", None,
                       "false = predict as the weighted combination of the observed values that followed each kept block; true = predict in the latent space, undo the standardization, and add the target month's seasonal term"),
    "selection": ("enum", ("greedy_forward", "exhaustive"),
                  "greedy_forward = stepwise: best single block first, then add one at a time while the MSE falls by more than zeta; exhaustive = every non-empty subset of the blocks, lowest MSE wins (zeta unused)"),
}
OPTIONAL = {"band_subset": ("list", None, "list of band names, required if and only if band_weighting is subset")}
REQUIRED = list(CLASS1) + list(CLASS2)
CROSS_FIELD_RULES = [
    "m + n must be at most 9, so that at least one historical block exists (blocks = 9 - m - n + 1).",
    "readd_seasonal = true requires fit_space = latent_std (there is no seasonal term to add back in raw space).",
    "intercept = fitted requires fit_space = raw or readd_seasonal = true (a constant fitted in standardized latent units has no meaning added to an observed-value prediction).",
    "weight_constraint = nonneg or unconstrained requires fit_space = raw or readd_seasonal = true (weights fitted in standardized latent space keep the observed values' scale only when they sum to one).",
    "band_subset is given if and only if band_weighting = subset, and lists band names of this sensor without repeats.",
    "No field may refer to P10. The fit sees P01-P09 only.",
]

HER_SPEC = {"m": 4, "n": 1, "kernel_bandwidth": 2.0, "zeta": 1e-4, "beta_trend_degree": 1,
            "weight_constraint": "simplex", "intercept": "none", "band_weighting": "equal",
            "fit_space": "latent_std", "readd_seasonal": False, "selection": "greedy_forward"}


def _typed(field, kind, allowed, v, errors):
    """Return the normalized value, or None after appending an error. bool is never accepted as a number."""
    if kind == "int":
        if isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, float) and v.is_integer())):
            errors.append(f"field '{field}': expected an integer, got {v!r}"); return None
        v = int(v)
        if not allowed(v):
            errors.append(f"field '{field}': {v} is outside the allowed range"); return None
        return v
    if kind == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"field '{field}': expected a number, got {v!r}"); return None
        v = float(v)
        if not allowed(v):
            errors.append(f"field '{field}': {v} is outside the allowed range"); return None
        return v
    if kind == "enum":
        if v not in allowed:
            errors.append(f"field '{field}': {v!r} is not one of {list(allowed)}"); return None
        return v
    if kind == "bool":
        if not isinstance(v, bool):
            errors.append(f"field '{field}': expected true or false, got {v!r}"); return None
        return v
    raise AssertionError(kind)


def validate_spec(spec, sensor, n_train=N_TRAIN):
    """Every violation is collected and raised together, each naming its field. Returns a normalized copy."""
    if not isinstance(spec, dict):
        raise SpecError(f"spec must be a JSON object, got {type(spec).__name__}")
    errors, out = [], {}
    missing = [f for f in REQUIRED if f not in spec]
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    unknown = [k for k in spec if k not in REQUIRED and k not in OPTIONAL]
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    for field, (kind, allowed, _) in {**CLASS1, **CLASS2}.items():
        if field in spec:
            v = _typed(field, kind, allowed, spec[field], errors)
            if v is not None:
                out[field] = v
    bands = [lb.disp(b) for b in lb.BANDS[sensor]]
    if out.get("band_weighting") == "subset":
        sub = spec.get("band_subset")
        if not isinstance(sub, list) or not sub:
            errors.append("field 'band_subset': required as a non-empty list when band_weighting is subset")
        else:
            bad = [b for b in sub if b not in bands]
            if bad:
                errors.append(f"field 'band_subset': {bad} are not bands of {sensor} ({bands})")
            if len(set(sub)) != len(sub):
                errors.append("field 'band_subset': repeated band")
            out["band_subset"] = list(sub)
    elif "band_subset" in spec and "band_weighting" in out and spec["band_subset"] not in (None, []):
        # A null or empty band_subset with a non-subset weighting counts as "not given": the reply format asks for
        # every listed key, so models send band_subset: null (2026-09-23, 54 of 108 arm-B first attempts).
        errors.append("field 'band_subset': given, but band_weighting is not subset")
    if "m" in out and "n" in out and out["m"] + out["n"] > n_train:
        errors.append(f"fields 'm','n': m + n = {out['m'] + out['n']} exceeds {n_train}; no historical block would exist")
    if out.get("readd_seasonal") is True and out.get("fit_space") == "raw":
        errors.append("fields 'readd_seasonal','fit_space': readd_seasonal requires fit_space = latent_std")
    if out.get("intercept") == "fitted" and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'intercept','fit_space','readd_seasonal': intercept = fitted requires fit_space = raw or readd_seasonal = true")
    if out.get("weight_constraint") in ("nonneg", "unconstrained") and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'weight_constraint','fit_space','readd_seasonal': nonneg or unconstrained weights require fit_space = raw or readd_seasonal = true")
    for k, v in spec.items():                                  # the guard: nothing may name P10
        if "P10" in json.dumps(v) or "P10" in str(k):
            errors.append(f"field '{k}': refers to P10, which is withheld")
    if errors:
        raise SpecError("; ".join(errors))
    return out


def class1_only(spec, sensor, n_train=N_TRAIN):
    """Arm A: the model supplies Class 1 only. A Class-2 key is a protocol violation, not something to trim."""
    if not isinstance(spec, dict):
        raise SpecError(f"spec must be a JSON object, got {type(spec).__name__}")
    present = [k for k in spec if k in CLASS2 or k in OPTIONAL]
    if present:
        raise SpecError("protocol violation: arm A takes Class-1 fields only, but got " + ", ".join(present))
    errors, out = [], {}
    missing = [f for f in CLASS1 if f not in spec]
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    unknown = [k for k in spec if k not in CLASS1]
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    for field, (kind, allowed, _) in CLASS1.items():
        if field in spec:
            v = _typed(field, kind, allowed, spec[field], errors)
            if v is not None:
                out[field] = v
    if "m" in out and "n" in out and out["m"] + out["n"] > n_train:
        errors.append(f"fields 'm','n': m + n = {out['m'] + out['n']} exceeds {n_train}; no historical block would exist")
    for k, v in spec.items():
        if "P10" in json.dumps(v) or "P10" in str(k):
            errors.append(f"field '{k}': refers to P10, which is withheld")
    if errors:
        raise SpecError("; ".join(errors))
    return out


def spec_fingerprint(spec):
    import hashlib
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def schema_text(class2=True):
    """The field list a prompt shows: name, allowed values, meaning. Nothing is filled in."""
    lines = ["Class 1 (settings of the method):"]
    lines += [f"  {f}: {meaning}" for f, (_, _, meaning) in CLASS1.items()]
    if class2:
        lines.append("Class 2 (choices of the method):")
        lines += [f"  {f}: one of {list(allowed)}. {meaning}" if kind == "enum" else f"  {f}: true or false. {meaning}"
                  for f, (kind, allowed, meaning) in CLASS2.items()]
        lines.append(f"  band_subset: {OPTIONAL['band_subset'][2]}")
        lines.append("Rules that apply across fields:")
        lines += [f"  - {r}" for r in CROSS_FIELD_RULES]
    else:
        lines.append("Rules that apply across fields:")
        lines += [f"  - {CROSS_FIELD_RULES[0]}", f"  - {CROSS_FIELD_RULES[-1]}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(schema_text())
    print("\nHER_SPEC validates:", validate_spec(HER_SPEC, "sentinel1") == HER_SPEC)
