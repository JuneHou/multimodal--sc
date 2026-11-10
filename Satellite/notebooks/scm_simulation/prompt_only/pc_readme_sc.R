# The README Section 3 synthetic control on each shipped panel, copied from data/simulation/README.md (Jun "copy from
# readme", 2026-11-07), as the fifth rule of the step analysis. Never a column of the results tables.
#   Rscript prompt_only/pc_readme_sc.R      from notebooks/scm_simulation; writes prompt_only/results/readme_sc_paths.csv
#                                           (sigma, dgp, period 1..60, y_syn) and readme_sc_weights.csv (sigma, dgp,
#                                           donor, weight)
#
# Step 1 (README lines 142 to 168)  predictors: Z1 and Z2 at the last pre-treatment period T0 = 40 and the outcome at
#        each of periods 1 to 40, 42 predictors, X1 for the treated unit and X0 for the 29 donors
# Step 2 (lines 172 to 192)  each predictor divided by its SD across the 30 units, no centring; V = I_K, not estimated
# Step 3 (lines 194 to 200)  W = argmin (X1 - X0 W)' V (X1 - X0 W), w_j >= 0, sum w_j = 1
# Step 4 (lines 202 to 212)  counterfactual sum_j w_j Y_jt; ATT = mean over 41..60 of observed minus counterfactual
# Section 5 (lines 330 to 335)  fitted with the R Synth package, custom.v = rep(1, 42), quadopt = "ipop"
suppressMessages(library(Synth))
T0 <- 40; PRE <- as.character(1:T0); ALL <- as.character(1:60)
dir.create("prompt_only/results", showWarnings = FALSE)
paths <- list(); weights <- list()
for (sigma in c("1.0", "0.5", "0.2")) {
  for (g in c("dgp1", "dgp2")) {
    d <- read.csv(file.path("../../data/simulation/simulation_data", paste0("sigma_", sigma), paste0(g, "_panel.csv")),
                  stringsAsFactors = FALSE)
    donors <- sort(setdiff(unique(d$unit_id), "treated")); units <- c("treated", donors)
    Y <- tapply(d$Y_observed, list(d$period, d$unit_id), function(x) x[1])[ALL, units]
    z <- d[d$period == T0, ]; Z <- as.matrix(z[match(units, z$unit_id), c("Z1", "Z2")])
    raw <- cbind(t(Y[PRE, ]), Z)                                    # 30 units x 42 predictors
    sds <- apply(raw, 2, function(x) sqrt(mean((x - mean(x))^2)))  # SD across the 30 units
    std <- sweep(raw, 2, sds, "/"); rownames(std) <- units
    X1 <- matrix(std["treated", ], ncol = 1); X0 <- t(std[donors, , drop = FALSE])
    Z1 <- matrix(Y[PRE, "treated"], ncol = 1); Z0 <- Y[PRE, donors, drop = FALSE]
    invisible(utils::capture.output(suppressWarnings(
      fit <- synth(X1 = X1, X0 = X0, Z1 = Z1, Z0 = Z0, custom.v = rep(1, nrow(X1)), quadopt = "ipop"))))
    w <- as.numeric(fit$solution.w)
    paths[[length(paths) + 1]] <- data.frame(sigma = sigma, dgp = g, period = 1:60, y_syn = as.numeric(Y[, donors] %*% w))
    weights[[length(weights) + 1]] <- data.frame(sigma = sigma, dgp = g, donor = donors, weight = w)
  }
}
write.csv(do.call(rbind, paths), "prompt_only/results/readme_sc_paths.csv", row.names = FALSE)
write.csv(do.call(rbind, weights), "prompt_only/results/readme_sc_weights.csv", row.names = FALSE)
cat("wrote prompt_only/results/readme_sc_paths.csv and readme_sc_weights.csv\n")
