"""scs_spec — the settings and choices a model writes, checked against a method's Spec (scs_method.Spec).

There are no defaults: a missing field is an error naming the field, never a fall-back. Every violation is collected
and reported together, each naming its field.

    validate(S, given, ctx)         design: settings and choices -> normalized copy, or SpecError
    settings_only(S, given, ctx)    configure: settings only; a choice field is a protocol violation, not trimmed
    fill_choices(S, settings)       configure: the settings with every choice at the reference code's value

`ctx` is what the method's rules need about the case (for SHC monthly the sensor; for SHC daily the sensor, the number
of visits given and the visits to predict). The rules themselves are the method's Check functions, run in order.
"""


class SpecError(ValueError):
    pass


KIND_LABEL = {"int": "<int>", "float": "<float>", "int_or_null": "<int or null>", "positive_list": "<list of numbers>",
              "enum": "<one of the listed values>", "bool": "<true or false>", "name_list": "<list of names>",
              "predictor_list": '<list of {"variable": <name>, "periods": <list of periods>, "op": <name>}>',
              "positive_list_or_null": "<list of numbers, or null>",
              "option": '<an object: "option" and the values that option carries>'}


def _whole_periods(v):
    return (isinstance(v, list) and len(v) > 0 and all(not isinstance(x, bool) and isinstance(x, (int, float))
                                                       and float(x).is_integer() and x >= 1 for x in v))


def typed(field, kind, allowed, v, errors):
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
    if kind == "positive_list":
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
    if kind == "positive_list_or_null":
        if v is None:
            return None
        return typed(field, "positive_list", allowed, v, errors)
    if kind == "name_list":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errors.append(f"field '{field}': expected a list of names, got {v!r}"); return None
        bad = [x for x in v if x not in allowed]
        if bad:
            errors.append(f"field '{field}': {bad} are not among {list(allowed)}"); return None
        if len(set(v)) != len(v):
            errors.append(f"field '{field}': a name is repeated"); return None
        return list(v)
    if kind == "predictor_list":
        if not isinstance(v, list):
            errors.append(f"field '{field}': expected a list of predictor entries, got {v!r}"); return None
        out = []
        for i, e in enumerate(v):
            if not isinstance(e, dict) or set(e) != {"variable", "periods", "op"}:
                errors.append(f"field '{field}': entry {i} must be an object with exactly \"variable\", \"periods\" "
                              f"and \"op\", got {e!r}"); return None
            if e["variable"] not in allowed["variables"]:
                errors.append(f"field '{field}': entry {i} variable {e['variable']!r} is not one of "
                              f"{list(allowed['variables'])}"); return None
            if not _whole_periods(e["periods"]):
                errors.append(f"field '{field}': entry {i} periods must be a non-empty list of whole periods, "
                              f"got {e['periods']!r}"); return None
            if e["op"] not in allowed["ops"]:
                errors.append(f"field '{field}': entry {i} op {e['op']!r} is not one of {list(allowed['ops'])}")
                return None
            out.append({"variable": e["variable"], "periods": [int(x) for x in e["periods"]], "op": e["op"]})
        return out
    if kind == "option":
        if not isinstance(v, dict) or "option" not in v:
            errors.append(f"field '{field}': expected an object with \"option\" and the values that option carries, "
                          f"got {v!r}"); return None
        o = v["option"]
        if not isinstance(o, str) or o not in allowed:
            errors.append(f"field '{field}': option {o!r} is not one of {list(allowed)}"); return None
        vals = allowed[o]["values"]
        got = [k for k in v if k != "option"]
        if set(got) != set(vals):
            errors.append(f"field '{field}': option {o!r} carries " + (", ".join(f"'{k}'" for k in vals) or "no values")
                          + ", got " + (", ".join(f"'{k}'" for k in got) or "none")); return None
        out = {"option": o}
        for k, f in vals.items():
            if f.kind is None:                          # typed by the method's checks
                out[k] = v[k]
                continue
            x = typed(f"{field}.{k}", f.kind, f.allowed, v[k], errors)
            if x is None and not (f.kind in ("int_or_null", "positive_list_or_null") and v[k] is None):
                return None
            out[k] = x
        return out
    raise AssertionError(kind)


def _type_fields(fields, given, errors, out):
    for name, f in fields.items():
        if name in given:
            v = typed(name, f.kind, f.allowed, given[name], errors)
            if v is not None or (f.kind in ("int_or_null", "positive_list_or_null") and given[name] is None):
                out[name] = v


def validate(S, given, ctx):
    if not isinstance(given, dict):
        raise SpecError(f"spec must be a JSON object, got {type(given).__name__}")
    errors, out = [], {}
    required = list(S.settings) + list(S.choices)
    missing = [f for f in required if f not in given]
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    unknown = [k for k in given if k not in required and k not in S.optional]
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    _type_fields({**S.settings, **S.choices}, given, errors, out)
    for c in S.checks:
        c.fn(given, out, ctx, errors)
    if errors:
        raise SpecError("; ".join(errors))
    return out


def settings_only(S, given, ctx):
    if not isinstance(given, dict):
        raise SpecError(f"spec must be a JSON object, got {type(given).__name__}")
    present = [k for k in given if k in S.choices or k in S.optional]
    if present:
        raise SpecError(f"protocol violation: {S.configure_label} takes Class-1 fields only, but got " + ", ".join(present))
    errors, out = [], {}
    missing = [f for f in S.settings if f not in given]
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    unknown = [k for k in given if k not in S.settings]
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    _type_fields(S.settings, given, errors, out)
    for c in S.checks:
        if c.configure:
            c.fn(given, out, ctx, errors)
    if errors:
        raise SpecError("; ".join(errors))
    return out


def fill_choices(S, settings):
    return {**settings, **{k: S.reference[k] for k in S.choices}}


def reference_choices(S, spec):
    """True when every choice of a validated spec equals the reference code's value."""
    return all(spec[k] == S.reference[k] and type(spec[k]) is type(S.reference[k]) for k in S.choices)
