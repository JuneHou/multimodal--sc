# Field checks of the settings file, the same rules as the package's checks (method.py and the pipeline's scs_spec.py).
# check_settings(given, mode) returns the problems found, each naming its field; an empty vector means the file is
# accepted. mode "configure" takes the settings only; mode "design" takes the settings and the choices.

SCM_VARIABLES <- c("Z1", "Z2", "Y_observed")
SCM_OPS <- c("mean", "median", "sd", "var", "min", "max", "sum")
SCM_DONORS <- sprintf("donor_%02d", 1:29)
SCM_OPTIMX <- c("BFGS", "CG", "Nelder-Mead", "L-BFGS-B", "nlm", "nlminb", "lbfgsb3c", "Rcgmin", "Rtnmin", "Rvmmin",
                "snewton", "snewtonm", "spg", "ucminf", "newuoa", "bobyqa", "uobyqa", "nmkb", "hjkb", "hjn", "lbfgs",
                "subplex", "ncg", "nvm", "mla", "slsqp", "tnewt", "anms", "pracmanm", "nlnm", "snewtm", "All")

SCM_SETTINGS <- list(
  special_predictors = list(kind = "predictor_list"),
  predictors = list(kind = "name_list", allowed = SCM_VARIABLES),
  predictors_op = list(kind = "enum", allowed = SCM_OPS),
  time_predictors_prior = list(kind = "positive_list"),
  time_optimize_ssr = list(kind = "positive_list"),
  controls_identifier = list(kind = "name_list", allowed = SCM_DONORS),
  custom_v = list(kind = "positive_list_or_null"),
  optimxmethod = list(kind = "name_list", allowed = SCM_OPTIMX),
  genoud = list(kind = "bool"),
  quadopt = list(kind = "enum", allowed = c("ipop")),
  margin_ipop = list(kind = "float"),
  sigf_ipop = list(kind = "int"),
  bound_ipop = list(kind = "float"))
SCM_CHOICES <- list(
  weights = list(kind = "enum", allowed = c("convex", "extrapolation")),
  interpolation_penalty = list(kind = "enum", allowed = c("none", "distance")))
SCM_OPTIONAL <- c("penalty_lambda")
SCM_SYNTH_CHOICES <- list(weights = "convex", interpolation_penalty = "none")     # what Synth does in its body

is_array <- function(v) is.list(v) && is.null(names(v))
is_number <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
is_whole <- function(v) is_number(v) && v == round(v)
shown <- function(v) if (is.null(v)) "null" else as.character(jsonlite::toJSON(v, auto_unbox = TRUE, null = "null"))

positive_list_ok <- function(v) is_array(v) && length(v) > 0 && all(vapply(v, function(x) is_number(x) && x > 0, TRUE))

check_value <- function(field, spec, v) {
  kind <- spec$kind
  if (kind == "int") {
    if (!is_whole(v)) return(sprintf("field '%s': expected an integer, got %s", field, shown(v)))
    if (!(v > 0)) return(sprintf("field '%s': %s is outside the allowed range", field, shown(v)))
  } else if (kind == "float") {
    if (!is_number(v)) return(sprintf("field '%s': expected a number, got %s", field, shown(v)))
    if (!(v > 0)) return(sprintf("field '%s': %s is outside the allowed range", field, shown(v)))
  } else if (kind == "bool") {
    if (!(is.logical(v) && length(v) == 1 && !is.na(v)))
      return(sprintf("field '%s': expected true or false, got %s", field, shown(v)))
  } else if (kind == "enum") {
    if (!(is.character(v) && length(v) == 1 && v %in% spec$allowed))
      return(sprintf("field '%s': %s is not one of %s", field, shown(v), shown(spec$allowed)))
  } else if (kind == "positive_list" || (kind == "positive_list_or_null" && !is.null(v))) {
    if (!positive_list_ok(v))
      return(sprintf("field '%s': expected a non-empty list of numbers > 0, got %s", field, shown(v)))
  } else if (kind == "name_list") {
    if (!is_array(v) || !all(vapply(v, function(x) is.character(x) && length(x) == 1, TRUE)))
      return(sprintf("field '%s': expected a list of names, got %s", field, shown(v)))
    bad <- setdiff(unlist(v), spec$allowed)
    if (length(bad)) return(sprintf("field '%s': %s are not among %s", field, shown(bad), shown(spec$allowed)))
    if (anyDuplicated(unlist(v))) return(sprintf("field '%s': a name is repeated", field))
  } else if (kind == "predictor_list") {
    if (!is_array(v)) return(sprintf("field '%s': expected a list of predictor entries, got %s", field, shown(v)))
    for (i in seq_along(v)) {
      e <- v[[i]]
      if (!(is.list(e) && !is.null(names(e)) && setequal(names(e), c("variable", "periods", "op")) && length(e) == 3))
        return(sprintf("field '%s': entry %d must be an object with exactly \"variable\", \"periods\" and \"op\", got %s",
                       field, i - 1, shown(e)))
      if (!(is.character(e$variable) && length(e$variable) == 1 && e$variable %in% SCM_VARIABLES))
        return(sprintf("field '%s': entry %d variable %s is not one of %s", field, i - 1, shown(e$variable),
                       shown(SCM_VARIABLES)))
      p <- e$periods
      if (!(is_array(p) && length(p) > 0 && all(vapply(p, function(x) is_whole(x) && x >= 1, TRUE))))
        return(sprintf("field '%s': entry %d periods must be a non-empty list of whole periods, got %s",
                       field, i - 1, shown(p)))
      if (!(is.character(e$op) && length(e$op) == 1 && e$op %in% SCM_OPS))
        return(sprintf("field '%s': entry %d op %s is not one of %s", field, i - 1, shown(e$op), shown(SCM_OPS)))
    }
  }
  NULL
}

check_settings <- function(given, mode) {
  if (!(is.list(given) && !is.null(names(given)))) return("spec must be a JSON object")
  fields <- if (mode == "design") c(SCM_SETTINGS, SCM_CHOICES) else SCM_SETTINGS
  if (mode == "configure") {
    present <- names(given)[names(given) %in% c(names(SCM_CHOICES), SCM_OPTIONAL)]
    if (length(present))
      return(paste0("protocol violation: scm-configure takes Class-1 fields only, but got ",
                    paste(present, collapse = ", ")))
  }
  errors <- character(0)
  missing <- setdiff(names(fields), names(given))
  if (length(missing)) errors <- c(errors, paste0("missing required field(s): ", paste(missing, collapse = ", ")))
  known <- if (mode == "design") c(names(fields), SCM_OPTIONAL) else names(fields)
  unknown <- setdiff(names(given), known)
  if (length(unknown)) errors <- c(errors, paste0("unknown field(s): ", paste(unknown, collapse = ", ")))
  ok <- character(0)
  for (f in names(fields)) {
    if (!(f %in% names(given))) next
    e <- check_value(f, fields[[f]], given[[f]])
    if (is.null(e)) ok <- c(ok, f) else errors <- c(errors, e)
  }
  # rule: every period in time_predictors_prior and time_optimize_ssr is a whole period number
  for (f in c("time_predictors_prior", "time_optimize_ssr")) {
    if (f %in% ok && !all(vapply(given[[f]], function(x) x == round(x), TRUE)))
      errors <- c(errors, sprintf("field '%s': every period must be a whole period number", f))
  }
  if (mode == "design") {
    pen <- if ("interpolation_penalty" %in% ok) given$interpolation_penalty else NULL
    # rule: penalty_lambda is given, a number of 0 or above, exactly when interpolation_penalty is distance
    if (identical(pen, "distance")) {
      if (!("penalty_lambda" %in% names(given))) {
        errors <- c(errors, "field 'penalty_lambda': required when interpolation_penalty is distance")
      } else {
        lam <- given$penalty_lambda
        if (!(is_number(lam) && lam >= 0))
          errors <- c(errors, sprintf("field 'penalty_lambda': expected a number of 0 or above, got %s", shown(lam)))
      }
    } else if ("penalty_lambda" %in% names(given) && !is.null(pen)) {
      errors <- c(errors, "field 'penalty_lambda': given, but interpolation_penalty is not distance")
    }
    # rule: interpolation_penalty = distance requires weights = convex
    if (identical(pen, "distance") && "weights" %in% ok && identical(given$weights, "extrapolation"))
      errors <- c(errors, paste0("fields 'interpolation_penalty','weights': interpolation_penalty = distance ",
                                 "requires weights = convex"))
  }
  errors
}
