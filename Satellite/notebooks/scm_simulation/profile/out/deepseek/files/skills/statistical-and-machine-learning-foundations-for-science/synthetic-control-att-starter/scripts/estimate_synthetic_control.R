#!/usr/bin/env Rscript
#
# estimate_synthetic_control.R
#
# Estimate the counterfactual outcome of a single treated unit for the treated
# periods of a panel and the average treatment effect on the treated (ATT) using
# the synthetic control method of Abadie, Diamond and Hainmueller (2010) through
# the Synth R package.
#
# The treated unit is identified by its `unit_id` value (default "treated"); all
# other units are untreated donors. Predictors (default `Z1`, `Z2`) and the
# pre-treatment outcome series are used to build a weighted combination of the
# donors that best reproduces the treated unit before treatment. The donor
# combination is then used as the counterfactual in the treated periods.
#
# Outputs `answer.json` with fields:
#   counterfactual: {"<period>": <number>, ...}
#   att:            <number>
#   rationale:      "<text>"
# and, optionally, a richer diagnostics JSON via --summary-out.

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}

suppressWarnings(suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("This script requires the 'jsonlite' R package. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("Synth", quietly = TRUE)) {
    stop("This script requires the 'Synth' R package (>= 1.1-10). Install with install.packages('Synth').")
  }
}))

fail <- function(...) {
  message("ERROR: ", paste0(..., collapse = ""))
  quit(status = 1L, save = "no")
}

print_usage <- function() {
  cat(paste(
    "Usage: Rscript estimate_synthetic_control.R [options]",
    "",
    "Required:",
    "  --panel PATH            Panel CSV with columns unit_id,period,treatment_group,Z1,Z2,Y_observed",
    "",
    "Options:",
    "  --out PATH              answer.json path (default: ./answer.json)",
    "  --summary-out PATH      Optional diagnostics JSON path",
    "  --treated-unit NAME     unit_id value of the treated unit (default: treated)",
    "  --treated-periods LIST  Comma-separated treated periods, e.g. 1989,1990,1991.",
    "                          Default: periods where the treated unit's treatment_group == 'treated'.",
    "  --pre-period-start N    First pre-treatment period to use for fitting (default: earliest period)",
    "  --pre-period-end N      Last pre-treatment period to use for fitting (default: period before first treated period)",
    "  --predictors LIST       Comma-separated predictor columns (default: Z1,Z2)",
    "  --lag-periods LIST      Comma-separated pre-periods added as lagged-outcome predictors",
    "  --custom-v LIST         Comma-separated predictor weights (bypasses V optimization)",
    "  --optimxmethod LIST     optimx methods (default: Nelder-Mead,BFGS)",
    "  --genoud                Use rgenoud global pre-search before local optimization",
    "  --allow-fallback        If Synth fails, fall back to a plain donor mean and flag it in rationale",
    "  --describe-panel        Print a panel summary JSON and exit without fitting",
    "  -h, --help              Show this help",
    "",
    "Example:",
    "  Rscript estimate_synthetic_control.R \\",
    "    --panel ./tobacco_panel.csv \\",
    "    --treated-periods 1989,1990,1991,1992 \\",
    "    --out ./answer.json --summary-out ./sc_summary.json",
    sep = "\n"
  ))
}

parse_args <- function(argv) {
  opts <- list(
    panel = NULL,
    out = "answer.json",
    summary_out = NULL,
    treated_unit = "treated",
    treated_periods = NULL,
    pre_start = NA_real_,
    pre_end = NA_real_,
    predictors = c("Z1", "Z2"),
    lag_periods = NULL,
    custom_v = NULL,
    optimxmethod = c("Nelder-Mead", "BFGS"),
    genoud = FALSE,
    allow_fallback = FALSE,
    describe = FALSE
  )
  i <- 1L
  n <- length(argv)
  split_list <- function(x) trimws(strsplit(x, ",")[[1]])
  while (i <= n) {
    key <- argv[[i]]
    need <- function() {
      if (i + 1L > n) fail("Missing value for argument ", key)
      argv[[i + 1L]]
    }
    if (key == "--panel") { opts$panel <- need(); i <- i + 2L }
    else if (key == "--out") { opts$out <- need(); i <- i + 2L }
    else if (key == "--summary-out") { opts$summary_out <- need(); i <- i + 2L }
    else if (key == "--treated-unit") { opts$treated_unit <- need(); i <- i + 2L }
    else if (key == "--treated-periods") { opts$treated_periods <- as.numeric(split_list(need())); i <- i + 2L }
    else if (key == "--pre-period-start") { opts$pre_start <- as.numeric(need()); i <- i + 2L }
    else if (key == "--pre-period-end") { opts$pre_end <- as.numeric(need()); i <- i + 2L }
    else if (key == "--predictors") { opts$predictors <- split_list(need()); i <- i + 2L }
    else if (key == "--lag-periods") { opts$lag_periods <- as.numeric(split_list(need())); i <- i + 2L }
    else if (key == "--custom-v") { opts$custom_v <- as.numeric(split_list(need())); i <- i + 2L }
    else if (key == "--optimxmethod") { opts$optimxmethod <- split_list(need()); i <- i + 2L }
    else if (key == "--genoud") { opts$genoud <- TRUE; i <- i + 1L }
    else if (key == "--allow-fallback") { opts$allow_fallback <- TRUE; i <- i + 1L }
    else if (key == "--describe-panel") { opts$describe <- TRUE; i <- i + 1L }
    else if (key %in% c("-h", "--help")) { print_usage(); quit(status = 0L) }
    else fail("Unknown argument: ", key)
  }
  opts
}

discover_panel <- function(path) {
  if (!is.null(path)) {
    return(path)
  }
  # Convenience: if exactly one *_panel.csv exists in the working directory, use it.
  candidates <- list.files(pattern = "_panel\\.csv$", full.names = FALSE)
  if (length(candidates) == 1L) {
    message("Auto-discovered panel: ", candidates[[1]])
    return(candidates[[1]])
  }
  if (length(candidates) == 0L) {
    fail("--panel not given and no *_panel.csv found in the working directory. ",
         "Pass --panel PATH.")
  }
  fail("--panel not given and multiple *_panel.csv files found: ",
       paste(candidates, collapse = ", "), ". Pass --panel PATH to disambiguate.")
}

read_panel <- function(path) {
  path <- discover_panel(path)
  if (!file.exists(path)) fail("Panel file not found: ", path)
  df <- tryCatch(
    utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) fail("Could not read panel CSV: ", conditionMessage(e))
  )
  required <- c("unit_id", "period", "treatment_group", "Z1", "Z2", "Y_observed")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0L) {
    fail("Panel is missing required columns: ", paste(missing, collapse = ", "),
         ". Found columns: ", paste(names(df), collapse = ", "))
  }
  if (nrow(df) == 0L) fail("Panel has zero rows.")
  df$period <- suppressWarnings(as.numeric(df$period))
  df$Y_observed <- suppressWarnings(as.numeric(df$Y_observed))
  if (any(is.na(df$period))) fail("Panel 'period' column has non-numeric values.")
  if (any(is.na(df$Y_observed))) fail("Panel 'Y_observed' column has non-numeric values.")
  df
}

panel_summary <- function(df, treated_unit) {
  units <- unique(df$unit_id)
  if (!(treated_unit %in% units)) {
    fail("Treated unit '", treated_unit, "' not found. Available unit_id values: ",
         paste(head(units, 20), collapse = ", "))
  }
  donors <- setdiff(units, treated_unit)
  treated_rows <- df[df$unit_id == treated_unit, ]
  list(
    panel_rows = nrow(df),
    unit_count = length(units),
    donor_count = length(donors),
    treated_unit = treated_unit,
    donors = donors,
    period_min = min(df$period),
    period_max = max(df$period),
    period_count = length(unique(df$period)),
    treated_group_periods = sort(unique(treated_rows$period[treated_rows$treatment_group == "treated"])),
    null_Y_rows = sum(is.na(df$Y_observed))
  )
}

coerce_treated_periods <- function(df, opts) {
  if (!is.null(opts$treated_periods)) {
    tp <- sort(unique(as.numeric(opts$treated_periods)))
  } else {
    # Auto-detect: periods where the treated unit row is flagged 'treated' while
    # at least one donor in the same period is not, i.e. a genuine treated-period
    # contrast. A unit-level treatment_group label produces no contrast.
    auto <- sort(unique(df$period[df$unit_id == opts$treated_unit & df$treatment_group == "treated"]))
    if (length(auto) > 0L) {
      donors <- setdiff(unique(df$unit_id), opts$treated_unit)
      auto <- auto[vapply(auto, function(p) {
        any(df$unit_id %in% donors & df$period == p & df$treatment_group != "treated")
      }, logical(1))]
    }
    tp <- auto
    if (length(tp) == 0L) {
      fail("Could not infer treated periods (no period has treatment_group contrast). ",
           "Pass --treated-periods explicitly, e.g. --treated-periods 1989,1990,1991.")
    }
  }
  tp <- sort(unique(as.numeric(tp)))
  available <- unique(df$period[df$unit_id == opts$treated_unit])
  missing <- setdiff(tp, available)
  if (length(missing) > 0L) {
    fail("Treated periods not present for the treated unit: ", paste(missing, collapse = ", "))
  }
  tp
}

resolve_common_panel <- function(df, treated_unit) {
  # Keep only periods where every unit has an observation so the matrices align.
  donors <- setdiff(unique(df$unit_id), treated_unit)
  keep_units <- c(treated_unit, donors)
  df <- df[df$unit_id %in% keep_units, ]
  tab <- table(df$unit_id, df$period)
  complete_periods <- as.numeric(colnames(tab)[apply(tab, 2, function(col) all(col >= 1L))])
  if (length(complete_periods) == 0L) fail("No period has observations for every unit.")
  df[df$period %in% complete_periods, ]
}

build_predictor_value <- function(df, col, pre_periods) {
  sub <- df[df$period %in% pre_periods, c("unit_id", col)]
  agg <- tapply(sub[[col]], sub$unit_id, function(x) mean(as.numeric(x), na.rm = TRUE))
  agg
}

fallback_donor_mean <- function(df, treated_unit, donors, treated_periods) {
  cf <- vapply(treated_periods, function(p) {
    vals <- df$Y_observed[df$unit_id %in% donors & df$period == p]
    mean(vals, na.rm = TRUE)
  }, numeric(1))
  names(cf) <- as.character(treated_periods)
  cf
}

write_json <- function(payload, path) {
  text <- jsonlite::toJSON(payload, pretty = TRUE, auto_unbox = TRUE, na = "null")
  if (!is.null(path)) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    writeLines(text, path, useBytes = TRUE)
  }
  invisible(text)
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(argv)
  if (opts$describe && is.null(opts$panel)) {
    fail("--describe-panel requires --panel.")
  }
  df <- read_panel(opts$panel)

  if (opts$describe) {
    payload <- panel_summary(df, opts$treated_unit)
    payload$columns <- names(df)
    cat(jsonlite::toJSON(payload, pretty = TRUE, auto_unbox = TRUE), "\n")
    return(invisible(NULL))
  }

  info <- panel_summary(df, opts$treated_unit)
  if (info$donor_count < 2L) {
    fail("Synthetic control needs at least 2 donor units; found ", info$donor_count, ".")
  }
  donors <- info$donors
  treated_unit <- opts$treated_unit
  treated_periods <- coerce_treated_periods(df, opts)

  first_treated <- min(treated_periods)
  pre_end <- if (is.na(opts$pre_end)) first_treated - 1 else opts$pre_end
  pre_start <- if (is.na(opts$pre_start)) info$period_min else opts$pre_start
  if (pre_end < pre_start) fail("Empty pre-treatment window (", pre_start, "-", pre_end, ").")

  df <- resolve_common_panel(df, treated_unit)
  all_periods <- sort(unique(df$period))
  pre_periods <- all_periods[all_periods >= pre_start & all_periods <= pre_end]
  if (length(pre_periods) < 2L) {
    fail("Need at least 2 pre-treatment periods to fit; found ", length(pre_periods),
         " in [", pre_start, ",", pre_end, "]. Adjust --pre-period-start/--pre-period-end.")
  }
  missing_tp <- setdiff(treated_periods, all_periods)
  if (length(missing_tp) > 0L) {
    fail("Treated periods absent after alignment: ", paste(missing_tp, collapse = ", "))
  }

  predictors <- opts$predictors
  for (p in predictors) {
    if (!(p %in% names(df))) fail("Predictor column not found: ", p)
  }

  fit_error <- NULL
  counterfactual <- NULL
  solution_w <- NULL
  solution_v <- NULL
  loss_v <- NA_real_
  method_used <- "synth"

  synth_fit <- tryCatch({
    df$.unit_num <- match(df$unit_id, c(treated_unit, donors))
    special <- NULL
    if (!is.null(opts$lag_periods)) {
      special <- lapply(opts$lag_periods, function(lp) list("Y_observed", lp, "mean"))
    }
    dp <- Synth::dataprep(
      foo = df,
      predictors = predictors,
      predictors.op = "mean",
      special.predictors = special,
      dependent = "Y_observed",
      unit.variable = ".unit_num",
      time.variable = "period",
      treatment.identifier = treated_unit,
      controls.identifier = donors,
      time.predictors.prior = pre_periods,
      time.optimize.ssr = pre_periods,
      time.plot = all_periods,
      unit.names.variable = "unit_id"
    )
    syn <- Synth::synth(
      dp,
      custom.v = opts$custom_v,
      optimxmethod = opts$optimxmethod,
      genoud = opts$genoud,
      verbose = FALSE
    )
    list(dp = dp, syn = syn)
  }, error = function(e) {
    fit_error <<- conditionMessage(e)
    NULL
  })

  if (!is.null(synth_fit)) {
    dp <- synth_fit$dp
    syn <- synth_fit$syn
    solution_w <- syn$solution.w
    solution_v <- as.numeric(syn$solution.v)
    loss_v <- as.numeric(syn$loss.v)
    synthetic_path <- dp$Y0plot %*% solution_w
    rownames(synthetic_path) <- rownames(dp$Y0plot)
    counterfactual <- vapply(treated_periods, function(p) {
      as.numeric(synthetic_path[as.character(p), 1])
    }, numeric(1))
    names(counterfactual) <- as.character(treated_periods)
  } else if (opts$allow_fallback) {
    method_used <- "donor-mean-fallback"
    counterfactual <- fallback_donor_mean(df, treated_unit, donors, treated_periods)
  } else {
    fail("Synth fitting failed: ", fit_error,
         "  Re-run with --allow-fallback for a donor-mean counterfactual, or inspect the panel.")
  }

  observed <- vapply(treated_periods, function(p) {
    df$Y_observed[df$unit_id == treated_unit & df$period == p][1]
  }, numeric(1))
  names(observed) <- as.character(treated_periods)
  gaps <- observed - counterfactual
  att <- mean(gaps)

  cf_json <- as.list(counterfactual)
  names(cf_json) <- names(counterfactual)

  weight_note <- ""
  if (!is.null(solution_w)) {
    w <- as.numeric(solution_w)
    names(w) <- rownames(solution_w)
    top <- sort(w, decreasing = TRUE)
    top <- top[top > 1e-4]
    weight_note <- paste0(
      " Donor weights (positive): ",
      paste(sprintf("%s=%.3f", names(top), top), collapse = ", "), "."
    )
  }

  rationale <- sprintf(
    paste0(
      "Synthetic control (Abadie, Diamond & Hainmueller 2010) fit with the Synth R package ",
      "(v%s). Treated unit '%s'; %d donor units; predictors [%s]; pre-treatment window %s-%s ",
      "(%d periods); pre-treatment MSPE=%.4f. Counterfactual for %s is the donor-weighted ",
      "trajectory; ATT = mean(observed - synthetic) over %d treated period(s) = %.4f.%s"
    ),
    as.character(utils::packageVersion("Synth")),
    treated_unit,
    length(donors),
    paste(predictors, collapse = ", "),
    pre_start, pre_end, length(pre_periods),
    loss_v,
    paste(treated_periods, collapse = ", "),
    length(treated_periods),
    att,
    weight_note
  )
  if (method_used == "donor-mean-fallback") {
    rationale <- paste0(rationale, " FALLBACK USED (", fit_error, "): counterfactual is the donor mean.")
  }

  answer <- list(
    counterfactual = cf_json,
    att = unname(att),
    rationale = rationale
  )
  write_json(answer, opts$out)

  if (!is.null(opts$summary_out)) {
    summary_payload <- list(
      method = method_used,
      treated_unit = treated_unit,
      donor_count = length(donors),
      donors = donors,
      predictors = predictors,
      lag_periods = opts$lag_periods,
      pre_period_start = pre_start,
      pre_period_end = pre_end,
      pre_period_count = length(pre_periods),
      treated_periods = treated_periods,
      observed = as.list(observed),
      counterfactual = cf_json,
      gaps = as.list(gaps),
      att = unname(att),
      pre_treatment_mspe = loss_v,
      solution_v = solution_v,
      solution_w = if (!is.null(solution_w)) {
        w <- as.numeric(solution_w); names(w) <- rownames(solution_w); as.list(w)
      } else NULL,
      synth_error = fit_error,
      answer_path = normalizePath(opts$out, mustWork = FALSE)
    )
    write_json(summary_payload, opts$summary_out)
  }

  cat(jsonlite::toJSON(
    list(status = "ok", method = method_used, att = unname(att),
         n_treated_periods = length(treated_periods), answer = normalizePath(opts$out, mustWork = FALSE)),
    auto_unbox = TRUE
  ), "\n")
  invisible(NULL)
}

main()
