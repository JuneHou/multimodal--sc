# The engines of the scm-four-design package, run outside the box by four_design/method.py and checks.py:
#   Rscript four_design/engine.R synth  <spec.json> <panel.csv>   the translated spec through Synth unchanged
#   Rscript four_design/engine.R native <spec.json> <panel.csv>   the translated spec through scm_native.R
#   Rscript four_design/engine.R auto   <spec.json> <panel.csv>   Synth or scm_native.R by the fit script's rule
# <spec.json> is a spec.json of scm-four-design, the four design choices. Prints the fit record as one JSON line, the
# record fit.R writes in the skill.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "..", "synth_fit.R"))
source(file.path(here, "..", "scm_native.R"))
source(file.path(here, "four_translate.R"))
args <- commandArgs(trailingOnly = TRUE)
g <- fromJSON(args[2], simplifyVector = FALSE)
s <- four_to_synth(g)
d <- scm_data(args[3])
a <- four_add_combinations(d, s)
rec <- switch(args[1], synth = fit_synth(a$d, a$s), native = fit_native(a$d, a$s), auto = four_fit(d, s),
              stop("engine must be synth, native or auto"))
rec$spec <- s
rec$design <- g
cat(scm_json(rec), "\n")
