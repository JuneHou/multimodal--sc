# R trial of the Synth package on one case folder of the simulated data (no model involved).
# Runs dataprep() and synth() of Synth 1.1-10 unchanged at the two reference ways and prints one JSON line:
#   {"weights": {"readme": {...}, "adh": {...}}, "prediction": {"readme": {...}, "adh": {...}}, "v": ..., "loss_v": ...}
# The prediction is the synthetic path Y0plot %*% solution.w for periods 41 to 60, as Synth's help computes it.
#   README way: Z1 and Z2 at period 40 and Y_observed at each of periods 1 to 40, V equal (custom.v = 42 ones).
#   ADH way:    Z1 and Z2 averaged over periods 23 to 40, Y_observed at 12, 23 and 40, V searched over periods 1 to 40.
# The arguments that describe the file are filled the same way as fit.R fills them (plan departure 1).
suppressPackageStartupMessages({library(jsonlite); library(Synth)})

f <- list.files(".", pattern = "_panel\\.csv$")
if (length(f) != 1) stop("expected exactly one *_panel.csv in the working directory")
d <- read.csv(f, stringsAsFactors = FALSE)
donors <- sort(unique(d$unit_id[d$unit_id != "treated"]))
d$unit_num <- match(d$unit_id, c("treated", donors))          # treated 1, donor_jj jj + 1
periods <- sort(unique(d$period))
post <- periods[periods >= 41]

run <- function(predictors, prior, special, ssr, custom_v) {
  dp <- dataprep(foo = d, predictors = predictors, predictors.op = "mean", special.predictors = special,
                 dependent = "Y_observed", unit.variable = "unit_num", time.variable = "period",
                 treatment.identifier = "treated", controls.identifier = donors,
                 time.predictors.prior = prior, time.optimize.ssr = ssr, time.plot = periods,
                 unit.names.variable = "unit_id")
  so <- synth(data.prep.obj = dp, custom.v = custom_v)
  w <- as.numeric(so$solution.w)
  path <- as.numeric(dp$Y0plot %*% so$solution.w)
  names(path) <- periods
  list(weights = setNames(as.list(w), donors),
       prediction = as.list(path[as.character(post)]),
       v = as.numeric(so$solution.v), loss_v = as.numeric(so$loss.v))
}

readme_special <- c(list(list("Z1", 40, "mean"), list("Z2", 40, "mean")),
                    lapply(1:40, function(t) list("Y_observed", t, "mean")))
readme <- run(predictors = NULL, prior = 1:40, special = readme_special, ssr = 1:40,
              custom_v = rep(1, length(readme_special)))

adh_special <- lapply(c(12, 23, 40), function(t) list("Y_observed", t, "mean"))
adh <- run(predictors = c("Z1", "Z2"), prior = 23:40, special = adh_special, ssr = 1:40, custom_v = NULL)

cat(toJSON(list(file = f,
                weights = list(readme = readme$weights, adh = adh$weights),
                prediction = list(readme = readme$prediction, adh = adh$prediction),
                v = list(readme = readme$v, adh = adh$v),
                loss_v = list(readme = readme$loss_v, adh = adh$loss_v)),
           auto_unbox = TRUE, digits = NA), "\n")
