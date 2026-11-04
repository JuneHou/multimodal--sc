#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

suppressWarnings(suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("This script requires the 'jsonlite' R package.")
  }
}))

split_csv_values <- function(value) {
  if (is.null(value) || !nzchar(value)) {
    return(character())
  }
  pieces <- trimws(unlist(strsplit(value, ",", fixed = TRUE)))
  pieces[nzchar(pieces)]
}

write_json <- function(payload, out_path = NULL) {
  text <- jsonlite::toJSON(payload, pretty = TRUE, auto_unbox = TRUE, null = "null")
  if (is.null(out_path) || identical(out_path, "-")) {
    cat(text, "\n")
    return(invisible(NULL))
  }
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(text, out_path, useBytes = TRUE)
}

stop_with <- function(message, status = 1L) {
  cat(message, "\n", file = stderr())
  quit(save = "no", status = status)
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(
    panel = NULL,
    treated_periods = character(),
    treated_unit_id = "treated",
    outcome_column = "Y_observed",
    predictor_columns = c("Z1", "Z2"),
    out = NULL,
    diagnostics_out = NULL,
    describe_panel = FALSE,
    min_pre_periods = 3L
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--panel") {
      parsed$panel <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--treated-periods") {
      parsed$treated_periods <- c(parsed$treated_periods, split_csv_values(args[[i + 1L]]))
      i <- i + 2L
    } else if (key == "--treated-period") {
      parsed$treated_periods <- c(parsed$treated_periods, trimws(args[[i + 1L]]))
      i <- i + 2L
    } else if (key == "--treated-unit-id") {
      parsed$treated_unit_id <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--outcome-column") {
      parsed$outcome_column <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--predictor-columns") {
      parsed$predictor_columns <- split_csv_values(args[[i + 1L]])
      i <- i + 2L
    } else if (key == "--out") {
      parsed$out <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--diagnostics-out") {
      parsed$diagnostics_out <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--describe-panel") {
      parsed$describe_panel <- TRUE
      i <- i + 1L
    } else if (key == "--min-pre-periods") {
      parsed$min_pre_periods <- as.integer(args[[i + 1L]])
      i <- i + 2L
    } else if (key %in% c("-h", "--help")) {
      cat(
        paste(
          "Usage: Rscript run_synth_single_treated_unit_scm.R [options]",
          "",
          "Estimate single-treated-unit synthetic-control counterfactuals from a local panel CSV.",
          "",
          "Options:",
          "  --panel PATH                 Optional explicit panel path. If omitted, auto-detect one *_panel.csv in cwd.",
          "  --treated-periods LIST       Comma-separated treated periods, for example 2006,2007,2008",
          "  --treated-period VALUE       Repeatable single treated period argument",
          "  --treated-unit-id VALUE      Treated unit identifier (default: treated)",
          "  --outcome-column NAME        Outcome column name (default: Y_observed)",
          "  --predictor-columns LIST     Comma-separated predictor columns (default: Z1,Z2)",
          "  --out PATH                   Output answer JSON path (default: answer.json)",
          "  --diagnostics-out PATH       Optional diagnostics JSON path",
          "  --describe-panel             Emit a panel summary instead of fitting Synth",
          "  --min-pre-periods N          Minimum number of pre-treatment periods required (default: 3)",
          "",
          "Examples:",
          "  Rscript run_synth_single_treated_unit_scm.R --panel toy_panel.csv --treated-periods 2006,2007,2008 --out answer.json",
          "  Rscript run_synth_single_treated_unit_scm.R --treated-period 1989 --treated-period 1990 --describe-panel",
          sep = "\n"
        )
      )
      quit(save = "no", status = 0L)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }
  }

  parsed$treated_periods <- unique(parsed$treated_periods[nzchar(parsed$treated_periods)])
  parsed$predictor_columns <- unique(parsed$predictor_columns[nzchar(parsed$predictor_columns)])
  parsed
}

locate_panel <- function(explicit_path = NULL) {
  if (!is.null(explicit_path)) {
    if (!file.exists(explicit_path)) {
      stop(sprintf("Panel file does not exist: %s", explicit_path))
    }
    return(normalizePath(explicit_path))
  }
  candidates <- sort(list.files(path = getwd(), pattern = "_panel\\.csv$", full.names = TRUE))
  if (length(candidates) != 1L) {
    stop(sprintf(
      "Expected exactly one '*_panel.csv' file in the current directory, found %d.",
      length(candidates)
    ))
  }
  normalizePath(candidates[[1L]])
}

coerce_numeric_column <- function(table, column_name) {
  raw_values <- table[[column_name]]
  numeric_values <- suppressWarnings(as.numeric(raw_values))
  if (any(is.na(numeric_values) & !is.na(raw_values) & nzchar(as.character(raw_values)))) {
    stop(sprintf("Column '%s' must be numeric for Synth.", column_name))
  }
  table[[column_name]] <- numeric_values
  table
}

period_metadata <- function(period_values) {
  period_labels <- as.character(period_values)
  unique_labels <- unique(period_labels)
  numeric_candidate <- suppressWarnings(as.numeric(unique_labels))
  numeric_ordering <- all(!is.na(numeric_candidate))
  if (numeric_ordering) {
    ordered_labels <- unique_labels[order(numeric_candidate)]
  } else {
    ordered_labels <- unique_labels
  }
  mapping <- setNames(seq_along(ordered_labels), ordered_labels)
  list(
    labels = period_labels,
    ordered_labels = ordered_labels,
    index = unname(mapping[period_labels]),
    map = mapping,
    numeric_ordering = numeric_ordering
  )
}

validate_balance <- function(table, period_labels, unit_ids) {
  global_periods <- sort(unique(period_labels))
  duplicate_rows <- table[duplicated(table[c("unit_id", "period_label")]), c("unit_id", "period_label"), drop = FALSE]
  balanced <- TRUE
  problems <- character()

  if (nrow(duplicate_rows) > 0L) {
    balanced <- FALSE
    problems <- c(problems, "duplicate unit_id/period rows")
  }

  for (unit_id in unit_ids) {
    unit_periods <- sort(unique(table$period_label[table$unit_id == unit_id]))
    if (!identical(unit_periods, global_periods)) {
      balanced <- FALSE
      problems <- c(problems, sprintf("unit %s has periods [%s]", unit_id, paste(unit_periods, collapse = ", ")))
    }
  }

  list(
    balanced = balanced,
    duplicate_rows = duplicate_rows,
    global_periods = global_periods,
    problems = unique(problems)
  )
}

prepare_panel <- function(panel_path, args) {
  required_columns <- c("unit_id", "period", "treatment_group", args$predictor_columns, args$outcome_column)
  panel <- utils::read.csv(panel_path, stringsAsFactors = FALSE, check.names = FALSE)

  if (nrow(panel) == 0L) {
    stop("Panel file is empty.")
  }

  missing_columns <- setdiff(required_columns, names(panel))
  if (length(missing_columns) > 0L) {
    stop(sprintf("Panel is missing required columns: %s", paste(missing_columns, collapse = ", ")))
  }

  panel$unit_id <- as.character(panel$unit_id)
  panel$period_label <- as.character(panel$period)

  period_info <- period_metadata(panel$period)
  panel$period_index <- period_info$index

  panel <- coerce_numeric_column(panel, args$outcome_column)
  for (column_name in args$predictor_columns) {
    panel <- coerce_numeric_column(panel, column_name)
  }

  if (!(args$treated_unit_id %in% panel$unit_id)) {
    stop(sprintf("Treated unit '%s' is not present in the panel.", args$treated_unit_id))
  }

  unit_ids <- sort(unique(panel$unit_id))
  donor_units <- unit_ids[unit_ids != args$treated_unit_id]
  if (length(donor_units) < 2L) {
    stop("Synthetic control requires at least two donor units.")
  }

  missing_values <- colSums(is.na(panel[, required_columns, drop = FALSE]))
  if (any(missing_values > 0L)) {
    bad_columns <- names(missing_values)[missing_values > 0L]
    stop(sprintf("Required columns contain missing values: %s", paste(bad_columns, collapse = ", ")))
  }

  balance_info <- validate_balance(panel, panel$period_label, unit_ids)

  if (!balance_info$balanced) {
    stop(sprintf("Panel must be balanced. Problems: %s", paste(balance_info$problems, collapse = "; ")))
  }

  requested_periods <- unique(as.character(args$treated_periods))
  if (length(requested_periods) == 0L) {
    stop("At least one treated period must be supplied with --treated-periods or --treated-period.")
  }

  unknown_periods <- setdiff(requested_periods, period_info$ordered_labels)
  if (length(unknown_periods) > 0L) {
    stop(sprintf("Requested treated periods are not present in the panel: %s", paste(unknown_periods, collapse = ", ")))
  }

  target_indices <- unname(period_info$map[requested_periods])
  order_idx <- order(target_indices)
  target_indices <- target_indices[order_idx]
  requested_periods <- requested_periods[order_idx]

  pre_indices <- seq_len(min(target_indices) - 1L)
  if (length(pre_indices) < args$min_pre_periods) {
    stop(sprintf(
      "Need at least %d pre-treatment periods before the earliest requested treated period; found %d.",
      args$min_pre_periods,
      length(pre_indices)
    ))
  }

  unit_levels <- c(args$treated_unit_id, sort(donor_units))
  panel$unit_num <- match(panel$unit_id, unit_levels)

  list(
    panel = panel,
    panel_path = panel_path,
    predictors = args$predictor_columns,
    outcome_column = args$outcome_column,
    treated_unit_id = args$treated_unit_id,
    donor_units = sort(donor_units),
    unit_levels = unit_levels,
    period_info = period_info,
    requested_periods = requested_periods,
    target_indices = target_indices,
    pre_indices = pre_indices,
    plot_indices = seq_len(max(target_indices)),
    balance_info = balance_info
  )
}

describe_panel <- function(prepared) {
  list(
    panel_file = prepared$panel_path,
    row_count = nrow(prepared$panel),
    unit_count = length(prepared$unit_levels),
    donor_count = length(prepared$donor_units),
    donor_units = prepared$donor_units,
    period_count = length(prepared$period_info$ordered_labels),
    available_periods = prepared$period_info$ordered_labels,
    requested_treated_periods = prepared$requested_periods,
    inferred_pre_periods = prepared$period_info$ordered_labels[prepared$pre_indices],
    balanced_panel = prepared$balance_info$balanced,
    duplicate_unit_period_rows = if (nrow(prepared$balance_info$duplicate_rows) == 0L) {
      character()
    } else {
      apply(prepared$balance_info$duplicate_rows, 1L, function(row) paste(row, collapse = "::"))
    },
    numeric_period_ordering = prepared$period_info$numeric_ordering,
    treated_unit_present = prepared$treated_unit_id %in% prepared$panel$unit_id
  )
}

ensure_synth <- function() {
  if (!requireNamespace("Synth", quietly = TRUE)) {
    stop("This script requires the 'Synth' R package. Install it before fitting the model.")
  }
}

make_special_predictors <- function(outcome_column, pre_indices) {
  lapply(pre_indices, function(index_value) {
    list(outcome_column, index_value, "mean")
  })
}

ordered_map_list <- function(keys, values, digits = 6L) {
  rounded <- round(as.numeric(values), digits = digits)
  output <- vector("list", length(keys))
  names(output) <- as.character(keys)
  for (idx in seq_along(keys)) {
    output[[idx]] <- rounded[[idx]]
  }
  output
}

top_donor_rows <- function(weight_vector, donor_units, top_n = 3L) {
  weights <- as.numeric(weight_vector)
  names(weights) <- donor_units
  ordered_names <- names(sort(weights, decreasing = TRUE))
  ordered_names <- ordered_names[seq_len(min(top_n, length(ordered_names)))]
  lapply(ordered_names, function(unit_id) {
    list(unit_id = unit_id, weight = round(unname(weights[[unit_id]]), 6))
  })
}

fit_scm <- function(prepared) {
  ensure_synth()

  dataprep_out <- Synth::dataprep(
    foo = prepared$panel,
    predictors = prepared$predictors,
    predictors.op = "mean",
    dependent = prepared$outcome_column,
    unit.variable = "unit_num",
    time.variable = "period_index",
    treatment.identifier = 1,
    controls.identifier = seq.int(2L, length(prepared$unit_levels)),
    special.predictors = make_special_predictors(prepared$outcome_column, prepared$pre_indices),
    time.predictors.prior = prepared$pre_indices,
    time.optimize.ssr = prepared$pre_indices,
    time.plot = prepared$plot_indices
  )

  synth_out <- Synth::synth(dataprep_out, verbose = FALSE)

  plot_labels <- prepared$period_info$ordered_labels[prepared$plot_indices]
  synthetic_values <- as.numeric(dataprep_out$Y0plot %*% synth_out$solution.w)
  actual_values <- as.numeric(dataprep_out$Y1plot)
  names(synthetic_values) <- plot_labels
  names(actual_values) <- plot_labels

  requested_labels <- prepared$requested_periods
  requested_counterfactual <- synthetic_values[requested_labels]
  requested_observed <- actual_values[requested_labels]
  requested_gaps <- requested_observed - requested_counterfactual

  pre_labels <- prepared$period_info$ordered_labels[prepared$pre_indices]
  pre_rmspe <- sqrt(mean((actual_values[pre_labels] - synthetic_values[pre_labels]) ^ 2))

  donor_weights <- as.numeric(synth_out$solution.w)
  names(donor_weights) <- prepared$donor_units
  predictor_weights <- as.numeric(synth_out$solution.v)
  names(predictor_weights) <- rownames(dataprep_out$X1)

  comparison_rows <- lapply(plot_labels, function(label) {
    list(
      period = label,
      observed = round(unname(actual_values[[label]]), 6),
      counterfactual = round(unname(synthetic_values[[label]]), 6),
      gap = round(unname(actual_values[[label]] - synthetic_values[[label]]), 6),
      phase = if (label %in% pre_labels) "pre" else if (label %in% requested_labels) "target" else "other"
    )
  })

  att <- mean(requested_gaps)
  top_donors <- top_donor_rows(donor_weights, prepared$donor_units, top_n = 3L)
  top_donor_text <- paste(
    vapply(top_donors, function(row) sprintf("%s=%.3f", row$unit_id, row$weight), character(1)),
    collapse = ", "
  )

  rationale <- sprintf(
    paste(
      "Estimated with Synth %s using %d donor units, predictors %s averaged over pre-treatment periods,",
      "and lagged %s for each pre-treatment period. Pre-treatment RMSPE was %.4f.",
      "Largest donor weights: %s. ATT is observed minus counterfactual over periods %s."
    ),
    as.character(utils::packageVersion("Synth")),
    length(prepared$donor_units),
    paste(prepared$predictors, collapse = ", "),
    prepared$outcome_column,
    pre_rmspe,
    top_donor_text,
    paste(requested_labels, collapse = ", ")
  )

  answer <- list(
    counterfactual = ordered_map_list(requested_labels, requested_counterfactual),
    att = round(unname(att), 6),
    rationale = rationale
  )

  diagnostics <- list(
    panel_file = prepared$panel_path,
    treated_unit_id = prepared$treated_unit_id,
    donor_units = prepared$donor_units,
    predictors = prepared$predictors,
    outcome_column = prepared$outcome_column,
    pre_periods = pre_labels,
    target_periods = requested_labels,
    pre_rmspe = round(unname(pre_rmspe), 6),
    att = round(unname(att), 6),
    donor_weights = lapply(prepared$donor_units, function(unit_id) {
      list(unit_id = unit_id, weight = round(unname(donor_weights[[unit_id]]), 6))
    }),
    predictor_weights = lapply(names(predictor_weights), function(name) {
      list(predictor = name, weight = round(unname(predictor_weights[[name]]), 6))
    }),
    top_donors = top_donors,
    loss_v = round(unname(as.numeric(synth_out$loss.v)), 6),
    loss_w = round(unname(as.numeric(synth_out$loss.w)), 6),
    comparison_rows = comparison_rows
  )

  list(answer = answer, diagnostics = diagnostics)
}

main <- function() {
  args <- parse_args()
  panel_path <- locate_panel(args$panel)
  prepared <- prepare_panel(panel_path, args)

  if (args$describe_panel) {
    write_json(describe_panel(prepared), args$out)
    return(invisible(NULL))
  }

  fitted <- fit_scm(prepared)
  write_json(fitted$answer, args$out %||% "answer.json")
  if (!is.null(args$diagnostics_out)) {
    write_json(fitted$diagnostics, args$diagnostics_out)
  }
  invisible(NULL)
}

tryCatch(
  main(),
  error = function(error) {
    stop_with(conditionMessage(error), status = 1L)
  }
)
