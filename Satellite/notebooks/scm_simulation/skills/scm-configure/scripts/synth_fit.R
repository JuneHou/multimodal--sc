# Synth 1.1-10, run unchanged at a spec: dataprep() then synth(), and the fit record.
# Shared by scripts/fit.R in both skills and by engine.R (the package's reference engine).
#
# The arguments that describe the data file are filled the same way for every spec:
#   dependent "Y_observed", unit.variable a column of unit numbers added in memory (treated 1, donor_jj jj + 1),
#   time.variable "period", treatment.identifier "treated", unit.names.variable "unit_id",
#   time.plot every period of the file, verbose FALSE.
suppressPackageStartupMessages({library(jsonlite); library(Synth)})

scm_data <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  donors <- sort(unique(d$unit_id[d$unit_id != "treated"]))
  d$unit_num <- match(d$unit_id, c("treated", donors))
  d
}

scm_dataprep <- function(d, s) {
  special <- lapply(s$special_predictors, function(e) list(e$variable, as.numeric(unlist(e$periods)), e$op))
  dataprep(foo = d,
           predictors = if (length(s$predictors)) unlist(s$predictors) else NULL,
           predictors.op = s$predictors_op,
           special.predictors = if (length(special)) special else NULL,
           dependent = "Y_observed", unit.variable = "unit_num", time.variable = "period",
           treatment.identifier = "treated", controls.identifier = unlist(s$controls_identifier),
           time.predictors.prior = as.numeric(unlist(s$time_predictors_prior)),
           time.optimize.ssr = as.numeric(unlist(s$time_optimize_ssr)),
           time.plot = sort(unique(d$period)), unit.names.variable = "unit_id")
}

scm_custom_v <- function(s) if (is.null(s$custom_v)) NULL else as.numeric(unlist(s$custom_v))

fit_synth <- function(d, s) {
  dp <- scm_dataprep(d, s)
  so <- synth(data.prep.obj = dp, custom.v = scm_custom_v(s), optimxmethod = unlist(s$optimxmethod),
              genoud = s$genoud, quadopt = s$quadopt, Margin.ipop = s$margin_ipop, Sigf.ipop = s$sigf_ipop,
              Bound.ipop = s$bound_ipop)
  scm_record(d, dp, so, s, "synth")
}

# The fit record: donors and weights, V and its predictors, the losses, and the synthetic path and gaps over every
# period of the file, computed as Synth's help computes them (Y0plot %*% solution.w, Y1plot minus that). Vectors are
# written as JSON arrays even when they hold one value.
scm_record <- function(d, dp, so, s, engine) {
  w <- as.numeric(so$solution.w)
  path <- as.numeric(dp$Y0plot %*% so$solution.w)
  gaps <- as.numeric(dp$Y1plot) - path
  periods <- as.character(sort(unique(d$period)))
  names_by_num <- setNames(d$unit_id, d$unit_num)
  list(engine_used = engine,
       donor_ids = as.list(unname(names_by_num[rownames(so$solution.w)])),
       weights = as.list(w),
       predictor_names = as.list(names(so$solution.v)),
       v = as.list(as.numeric(so$solution.v)),
       fit_mse = as.numeric(so$loss.v),
       loss_w = as.numeric(so$loss.w),
       predictions = setNames(as.list(path), periods),
       gaps = setNames(as.list(gaps), periods),
       spec = s)
}

scm_json <- function(x) toJSON(x, auto_unbox = TRUE, digits = NA, null = "null")
