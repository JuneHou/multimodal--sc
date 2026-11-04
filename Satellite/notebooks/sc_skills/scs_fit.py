"""scs_fit — one fit under a validated spec: the copied reference code when every choice equals the reference code's
value, the method's native engine otherwise (design, step 2).

    run(pkg, ctx, spec, engine="auto")
        engine = "auto"       the copied code when the choices are the reference code's, else the native engine
                 "reference"  force the copied code (error if any choice differs)
                 "native"     force the native engine even on the reference choices (test only)

Nothing here knows a method's settings as values: the dispatch compares the spec's choices with the package's
reference values, which is what routing to the copied code requires.
"""
import scs_spec


def run(pkg, ctx, spec, engine="auto"):
    assert engine in ("auto", "reference", "native"), engine
    missing = [k for k in pkg.spec.choices if k not in spec]
    if missing:
        raise ValueError(f"spec is missing choice field(s) {missing}")
    at_reference = scs_spec.reference_choices(pkg.spec, spec)
    if engine == "reference" and not at_reference:
        raise ValueError("engine='reference' needs every choice at the reference code's value")
    if at_reference and engine != "native":
        return pkg.engine.reference(ctx, spec)
    return pkg.engine.native(ctx, spec, engine == "native")
