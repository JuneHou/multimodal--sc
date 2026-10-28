"""lb_spec_daily — the daily SHC design specification a model fills in (plan ~/.claude/plans/simple-skills-daily.md,
rows C1 to C3). The monthly lb_spec.py with each monthly element in the form notebook 14 gives it. There are no
defaults anywhere in this module's validation: a field the caller leaves out is an error that names the field.

Class 1: settings notebook 14 already takes, without its numerical tolerances and on/off switches (applied by
rebinding her names, lb_fit_daily.class1_binds). m* and each band's bandwidth are chosen by her rules.
Class 2: choices her code makes one way with no branch; the alternatives are new code in lb_fit_daily.
HER_SPEC: her settings, for tests only. lb_fit_daily may not import it: the fit code must never know her values.

    validate_spec(spec, sensor, n_given, to_predict)   -> normalized copy, or SpecError listing every violation
    class1_only(spec, sensor, n_given, to_predict)     -> the same for shc-configure, refusing any Class-2 key
    schema_text(class2=True)                           -> the field list: names, allowed values, meaning
`n_given` is the number of visits given; `to_predict` lists the period_id and date of every visit to predict.
"""
import hashlib
import json

import lb_lib as lb


class SpecError(ValueError):
    pass


def _bw_list(v):
    return isinstance(v, list) and len(v) > 0 and all(
        not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0 for x in v)


# name -> (type, allowed / predicate, one-line meaning for the skill text)
CLASS1 = {
    "time_bandwidth_candidates": ("bwlist", _bw_list,
                                  "non-empty list of numbers > 0: the Gaussian bandwidths, in days, among which leave-one-out cross-validation picks each band's bandwidth for the local-linear smoother that estimates the band's latent trajectory"),
    "m_min": ("int", lambda v: v >= 1,
              "integer >= 1: the shortest matching window tried, in visits"),
    "m_step": ("int", lambda v: v >= 1,
               "integer >= 1: the step between the matching window lengths tried, in visits"),
    "m_max": ("int_or_null", lambda v: v >= 1,
              "integer >= 1, or null: the longest matching window tried, in visits; null = the longest that leaves at least one historical block"),
    "max_donors": ("int", lambda v: v >= 1,
                   "integer >= 1: the most historical blocks used; when more exist, this many are kept, spread evenly over the history"),
    "nu": ("float", lambda v: v >= 0,
           "number >= 0: the matching window length is the longest m tried whose fit on all blocks has R <= nu, where R is the mean over bands of the band's mean squared matching error divided by the variance of its values over the visits given; if no m passes, the longest m tried"),
    "zeta": ("float", lambda v: v >= 0,
             "number >= 0: in stepwise selection a further block is kept only if it lowers R by more than this"),
}
CLASS2 = {
    "weight_constraint": ("enum", ("simplex", "nonneg", "unconstrained"),
                          "simplex = weights >= 0 summing to 1; nonneg = weights >= 0, any sum; unconstrained = any real weights"),
    "intercept": ("enum", ("none", "fitted"),
                  "none = no constant term; fitted = a constant is fitted alongside the block weights and added to the prediction"),
    "band_weighting": ("enum", ("equal", "inverse_variance", "subset"),
                       "equal = every band's rows enter the joint fit with weight 1; inverse_variance = each band's rows scaled by 1/SD of that band over the visits given in the fitting space; subset = fit on the bands listed in band_subset only (all bands are still predicted)"),
    "fit_space": ("enum", ("latent_std", "raw"),
                  "latent_std = weights fitted on each band's smoothed, standardized trajectory; raw = weights fitted on the observed band values"),
    "readd_seasonal": ("bool", None,
                       "false = predict as the weighted combination of the observed values that followed each kept block; true = predict in the latent space, undo the standardization, and add the seasonal term, which is zero for this method"),
    "selection": ("enum", ("greedy_forward", "exhaustive"),
                  "greedy_forward = stepwise: the block with the lowest matching MSE first, then add one at a time the block that lowers R most, while R falls by more than zeta; exhaustive = every non-empty subset of the blocks, lowest R wins (zeta unused)"),
}
OPTIONAL = {"band_subset": ("list", None, "list of band names, required if and only if band_weighting is subset")}
REQUIRED = list(CLASS1) + list(CLASS2)
CROSS_FIELD_RULES = [
    "m_min must be at most m_max.",
    "m_max plus the number of visits to predict must be at most the number of visits given, so that at least one historical block exists; with m_max null the same holds for m_min.",
    "readd_seasonal = true requires fit_space = latent_std (there is no seasonal term to add back in raw space).",
    "intercept = fitted requires fit_space = raw or readd_seasonal = true (a constant fitted in standardized latent units has no meaning added to an observed-value prediction).",
    "weight_constraint = nonneg or unconstrained requires fit_space = raw or readd_seasonal = true (weights fitted in standardized latent space keep the observed values' scale only when they sum to one).",
    "band_subset is given if and only if band_weighting = subset, and lists band names of this sensor without repeats.",
    "No field may refer to a visit to predict. The fit sees the visits given only.",
]
CONFIGURE_RULES = [0, 1, 6]                  # the rules that involve Class-1 fields only

HER_SPEC = {"time_bandwidth_candidates": [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0], "m_min": 10, "m_step": 5,
            "m_max": None, "max_donors": 60, "nu": 0.1, "zeta": 0.001,
            "weight_constraint": "simplex", "intercept": "none", "band_weighting": "equal",
            "fit_space": "latent_std", "readd_seasonal": False, "selection": "greedy_forward"}


def _typed(field, kind, allowed, v, errors):
    """Return the normalized value, or None after appending an error. bool is never accepted as a number."""
    if kind == "int_or_null":
        if v is None:
            return None
        kind = "int"
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
    if kind == "bwlist":
        if not allowed(v):
            errors.append(f"field '{field}': expected a non-empty list of numbers > 0, got {v!r}"); return None
        return [float(x) for x in v]
    if kind == "enum":
        if v not in allowed:
            errors.append(f"field '{field}': {v!r} is not one of {list(allowed)}"); return None
        return v
    if kind == "bool":
        if not isinstance(v, bool):
            errors.append(f"field '{field}': expected true or false, got {v!r}"); return None
        return v
    raise AssertionError(kind)


def _class1(spec, errors, out):
    for field, (kind, allowed, _) in CLASS1.items():
        if field in spec:
            v = _typed(field, kind, allowed, spec[field], errors)
            if v is not None or (kind == "int_or_null" and spec[field] is None):
                out[field] = v


def _class1_rules(out, n_given, to_predict, errors):
    n = len(to_predict)
    if isinstance(out.get("m_min"), int) and isinstance(out.get("m_max"), int) and out["m_min"] > out["m_max"]:
        errors.append(f"fields 'm_min','m_max': m_min = {out['m_min']} exceeds m_max = {out['m_max']}")
    if isinstance(out.get("m_max"), int) and out["m_max"] + n > n_given:
        errors.append(f"fields 'm_max': m_max + {n} visits to predict = {out['m_max'] + n} exceeds the {n_given} "
                      f"visits given; no historical block would exist")
    if "m_max" in out and out["m_max"] is None and isinstance(out.get("m_min"), int) and out["m_min"] + n > n_given:
        errors.append(f"fields 'm_min','m_max': m_max is null and m_min + {n} visits to predict = {out['m_min'] + n} "
                      f"exceeds the {n_given} visits given; no historical block would exist")


def _guard(spec, to_predict, errors):
    """Nothing may name a visit to predict, by period_id or by date."""
    names = [str(x) for pair in to_predict for x in pair]
    for k, v in spec.items():
        text = json.dumps(v) + " " + str(k)
        hit = [x for x in names if x in text]
        if hit:
            errors.append(f"field '{k}': refers to {hit[0]}, a visit to predict, which is withheld")


def validate_spec(spec, sensor, n_given, to_predict):
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
    _class1(spec, errors, out)
    for field, (kind, allowed, _) in CLASS2.items():
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
        errors.append("field 'band_subset': given, but band_weighting is not subset")
    _class1_rules(out, n_given, to_predict, errors)
    if out.get("readd_seasonal") is True and out.get("fit_space") == "raw":
        errors.append("fields 'readd_seasonal','fit_space': readd_seasonal requires fit_space = latent_std")
    if out.get("intercept") == "fitted" and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'intercept','fit_space','readd_seasonal': intercept = fitted requires fit_space = raw or readd_seasonal = true")
    if out.get("weight_constraint") in ("nonneg", "unconstrained") and out.get("fit_space") == "latent_std" and out.get("readd_seasonal") is False:
        errors.append("fields 'weight_constraint','fit_space','readd_seasonal': nonneg or unconstrained weights require fit_space = raw or readd_seasonal = true")
    _guard(spec, to_predict, errors)
    if errors:
        raise SpecError("; ".join(errors))
    return out


def class1_only(spec, sensor, n_given, to_predict):
    """shc-configure: the model supplies Class 1 only. A Class-2 key is a protocol violation, not something to trim."""
    if not isinstance(spec, dict):
        raise SpecError(f"spec must be a JSON object, got {type(spec).__name__}")
    present = [k for k in spec if k in CLASS2 or k in OPTIONAL]
    if present:
        raise SpecError("protocol violation: shc-configure takes Class-1 fields only, but got " + ", ".join(present))
    errors, out = [], {}
    missing = [f for f in CLASS1 if f not in spec]
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    unknown = [k for k in spec if k not in CLASS1]
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    _class1(spec, errors, out)
    _class1_rules(out, n_given, to_predict, errors)
    _guard(spec, to_predict, errors)
    if errors:
        raise SpecError("; ".join(errors))
    return out


def spec_fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def schema_text(class2=True):
    """The field list a skill shows: name, allowed values, meaning. Nothing is filled in."""
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
        lines += [f"  - {CROSS_FIELD_RULES[i]}" for i in CONFIGURE_RULES]
    return "\n".join(lines)


if __name__ == "__main__":
    print(schema_text())
    tp = [("P243", "2024-09-29")]
    print("\nHER_SPEC validates:", validate_spec(HER_SPEC, "sentinel1", 242, tp) == HER_SPEC)
