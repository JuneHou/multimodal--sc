#!/usr/bin/env Rscript
# Fit the synthetic control method with the Synth R package, version 1.1-10, unchanged, at the settings in a file:
#   Rscript scripts/fit.R settings.json
# The arguments of dataprep() that describe the data file are filled here, the same for every file:
#   dependent "Y_observed", unit.variable unit numbers (treated 1, donor_jj jj + 1), time.variable "period",
#   treatment.identifier "treated", unit.names.variable "unit_id", time.plot every period of the file, verbose FALSE.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "scm_fields.R"))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
args <- commandArgs(trailingOnly = TRUE)
spec_file <- if (length(args)) args[1] else "settings.json"
given <- tryCatch(fromJSON(spec_file, simplifyVector = FALSE), error = function(e) {
  cat("refused: ", spec_file, " is not readable JSON (", conditionMessage(e), ")\n", sep = "")
  quit(status = 1)
})
errors <- check_settings(given, "configure")
if (length(errors)) {
  cat("refused: ", paste(errors, collapse = "; "), "\n", sep = "")
  quit(status = 1)
}
s <- c(given, SCM_SYNTH_CHOICES)          # the two choices this skill fixes, at what Synth does
panel <- list.files(".", pattern = "_panel[.]csv$")
if (length(panel) != 1) stop("expected one *_panel.csv file in the working directory")
d <- scm_data(panel)
synth_choices <- identical(s$weights, "convex") && identical(s$interpolation_penalty, "none")
rec <- if (synth_choices) fit_synth(d, s) else fit_native(d, s)
rec$file <- basename(spec_file)
log <- Sys.getenv("FIT_LOG")                         # the record of this fit, for the run log
if (nzchar(log)) cat(scm_json(rec), "\n", file = log, append = TRUE, sep = "")

cat(if (synth_choices) "fit: Synth 1.1-10 dataprep() and synth(), unchanged\n" else
      sprintf("fit: Synth's code with weights = %s, interpolation_penalty = %s\n", s$weights, s$interpolation_penalty))
cat(sprintf("predictors and their weights in V (%d):\n", length(rec$v)))
for (i in seq_along(rec$v)) cat(sprintf("  %-28s %.6f\n", rec$predictor_names[[i]], rec$v[[i]]))
cat(sprintf("donor weights (%d donors in the pool):\n", length(rec$weights)))
for (i in seq_along(rec$weights)) cat(sprintf("  %-10s %.6f\n", rec$donor_ids[[i]], rec$weights[[i]]))
cat(sprintf("pre-treatment fit: mean squared prediction error over time_optimize_ssr (%d periods) %.6f\n",
            length(s$time_optimize_ssr), rec$fit_mse))
cat("period, treated unit observed, synthetic control, gap (observed minus synthetic control):\n")
for (p in names(rec$predictions))
  cat(sprintf("  %3s %12.6f %12.6f %12.6f\n", p, rec$predictions[[p]] + rec$gaps[[p]], rec$predictions[[p]],
              rec$gaps[[p]]))
