# scm-four-design: the four design choices turned into Synth's arguments, and the fit (plan
# use-nature-skills-analyze-wiggly-coral.md, "How the fit script turns the spec into Synth's arguments").
#   four_to_synth(g)        spec.json (checked) -> the translated spec: Synth's arguments, outcome_combinations and the
#                           two choices weights and interpolation_penalty (with penalty_lambda under distance)
#   four_fit(d, s)          the fit of a translated spec: each outcome combination added to the panel as a column and
#                           appended as a special predictor, then Synth when the weights are convex and the penalty is
#                           none, the native engine (scm_native.R) otherwise. The record keeps s as given here.
# Every fit uses all 29 donors and Synth's default optimizers and solver settings (synth.R lines 8 to 13).
# Needs synth_fit.R and scm_native.R sourced first.

FOUR_ALL_DONORS <- sprintf("donor_%02d", 1:29)

four_to_synth <- function(g) {
  p <- g$predictors; v <- g$v
  last <- if (identical(v$option, "validation")) as.integer(v$training_end) else 40L
  s <- list(
    predictors = list("Z1", "Z2"), predictors_op = "mean",
    time_predictors_prior = lapply(unlist(p$covariate_periods), as.numeric),
    special_predictors = if (identical(p$option, "every_outcome"))
      lapply(seq_len(last), function(t) list(variable = "Y_observed", periods = list(t), op = "mean")) else list(),
    outcome_combinations = if (identical(p$option, "summaries")) p$summaries else list(),
    controls_identifier = as.list(FOUR_ALL_DONORS),
    custom_v = if (identical(v$option, "judgement")) lapply(unlist(v$values), as.numeric) else NULL,
    time_optimize_ssr = as.list(if (identical(v$option, "validation")) (last + 1L):40L else 1:40),
    optimxmethod = list("Nelder-Mead", "BFGS"), genoud = FALSE, quadopt = "ipop",
    margin_ipop = 0.0005, sigf_ipop = 5, bound_ipop = 10,
    weights = g$weights$option, interpolation_penalty = g$penalty$option)
  if (identical(g$penalty$option, "distance")) s$penalty_lambda <- as.numeric(g$penalty$lambda)
  s
}

# Each combination m becomes the column outcome_combination_m: for each unit, the sum of the weights times its
# outcome in their periods, constant over the periods, and a special predictor averaged over the combination's first
# period. A spec with no outcome_combinations is returned unchanged.
four_add_combinations <- function(d, s) {
  comb <- s$outcome_combinations
  if (!length(comb)) return(list(d = d, s = s))
  for (m in seq_along(comb)) {
    w <- comb[[m]]
    per <- as.integer(names(w)); k <- as.numeric(unlist(w))
    col <- sprintf("outcome_combination_%d", m)
    val <- vapply(split(d, d$unit_id), function(u) sum(k * u$Y_observed[match(per, u$period)]), 0)
    d[[col]] <- unname(val[d$unit_id])
    s$special_predictors <- c(s$special_predictors, list(list(variable = col, periods = list(min(per)), op = "mean")))
  }
  list(d = d, s = s)
}

four_synth_choices <- function(s) {
  w <- if (is.null(s$weights)) "convex" else s$weights
  pen <- if (is.null(s$interpolation_penalty)) "none" else s$interpolation_penalty
  identical(w, "convex") && identical(pen, "none")
}

four_fit <- function(d, s) {
  a <- four_add_combinations(d, s)
  rec <- if (four_synth_choices(s)) fit_synth(a$d, a$s) else fit_native(a$d, a$s)
  rec$spec <- s
  rec
}
