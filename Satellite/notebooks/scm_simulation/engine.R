# The package's engines, run outside the box by method.py:
#   Rscript engine.R synth  <spec.json> <panel.csv>    Synth unchanged (synth_fit.R)
#   Rscript engine.R native <spec.json> <panel.csv>    the copy with the design choices (scm_native.R)
# Prints the fit record as one JSON line, the same record fit.R prints in the skills.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
args <- commandArgs(trailingOnly = TRUE)
s <- fromJSON(args[2], simplifyVector = FALSE)
d <- scm_data(args[3])
rec <- if (args[1] == "synth") fit_synth(d, s) else fit_native(d, s)
cat(scm_json(rec), "\n")
