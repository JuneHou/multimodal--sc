# The collaborator's SCM (data/simulation/README.md Section 3) computed from her formulas, with her checks
# (Section 4.1, 4.2, 4.3), on one panel. No Synth call; quadprog's active-set solver gives the exact minimum.
#   Rscript her_section3.R <panel.csv> <ground_truth.csv> <dgp>      prints one JSON line
#
# Step 1  X1 and X0: Z1 and Z2 at T0 = 40 and Y_observed at each of periods 1..40 (42 predictors), treated and 29 donors
# Step 2  each predictor divided by its standard deviation over the treated unit and the 29 donors (Synth's scaling, the
#         paper's code); V = I_42
# Step 3  W = argmin (X1 - X0 W)' V (X1 - X0 W), w_j >= 0, sum w_j = 1, solved exactly; a weight is 0 exactly when its
#         constraint is active
# Step 4  counterfactual sum_j w_j Y_jt for 41..60; ATT = mean over 41..60 of (Y_1t - counterfactual)
# 4.1     counterfactual RMSE against Y0_true; ATT error = ATT - mean tau_true; relative = |ATT error| / |mean tau_true|
# 4.2     pre-treatment RMSPE over 1..40; covariate balance sqrt(1/2 sum_r (Z_1r,T0 - sum_j w_j Z_jr,T0)^2)
# 4.3     leave one period out: for s in 1..40, W(-s) = argmin over the other 39 pre-treatment periods of
#         sum_t (Y_1t - sum_j w_j Y_jt)^2 under the Section 3 constraints; CV-RMSE = sqrt(mean e_s^2);
#         block: the same fit on 1..35, RMSE over 36..40 (her "say 36-40");
#         leave one donor out: for each donor with w_j > 0, Steps 1 to 4 without it; max_j |ATT(-j) - ATT|
suppressPackageStartupMessages({library(quadprog); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE)
d <- read.csv(a[1], stringsAsFactors = FALSE)
truth <- read.csv(a[2]); tr <- truth[truth$dgp == a[3], ]
donors <- sort(setdiff(unique(d$unit_id), "treated")); units <- c("treated", donors)
Y <- tapply(d$Y_observed, list(d$period, d$unit_id), function(x) x[1])[, units]
Z <- as.matrix(d[d$period == 40, c("unit_id", "Z1", "Z2")][match(units, d$unit_id[d$period == 40]), c("Z1", "Z2")])
rownames(Z) <- units
pre <- 1:40; post <- 41:60

simplex_qp <- function(Dmat, dvec) {             # min w'Dw/2 - d'w, sum w = 1, w >= 0; exact zeros from active set
  J <- length(dvec)
  s <- solve.QP(Dmat, dvec, cbind(rep(1, J), diag(J)), c(1, rep(0, J)), meq = 1)
  w <- s$solution; w[s$iact[s$iact > 1] - 1] <- 0
  w
}
section3 <- function(keep) {                     # Steps 1 to 3 with the donors in `keep`
  X <- rbind(t(Z[c("treated", keep), ]), Y[pre, c("treated", keep)])       # 42 x (1 + J)
  X <- X / sqrt(apply(X, 1, var))                                          # Step 2 scaling, V = I
  X1 <- X[, 1]; X0 <- X[, -1, drop = FALSE]
  setNames(simplex_qp(t(X0) %*% X0, as.numeric(t(X0) %*% X1)), keep)
}
w <- section3(donors)
cf_all <- as.numeric(Y[, donors] %*% w)
cf <- cf_all[post]; att <- mean(Y[post, "treated"] - cf)
true_att <- mean(tr$tau_true[post])
out <- list(
  n_predictors = 2 + length(pre), donors_positive = sum(w > 0), max_weight = max(w),
  cf_rmse = sqrt(mean((cf - tr$Y0_true[post])^2)), att = att, att_error = att - true_att,
  rel_att_error = abs(att - true_att) / abs(true_att),
  pre_rmspe = sqrt(mean((Y[pre, "treated"] - cf_all[pre])^2)),
  covariate_balance = sqrt(mean((Z["treated", ] - as.numeric(t(Z[donors, ]) %*% w))^2)))
outcome_qp <- function(rows) simplex_qp(t(Y[rows, donors]) %*% Y[rows, donors], as.numeric(t(Y[rows, donors]) %*% Y[rows, "treated"]))
e <- sapply(pre, function(s) { ws <- outcome_qp(setdiff(pre, s)); Y[s, "treated"] - sum(ws * Y[s, donors]) })
wb <- outcome_qp(1:35)
out$cv_rmse <- sqrt(mean(e^2))
out$block_36_40_rmse <- sqrt(mean((Y[36:40, "treated"] - as.numeric(Y[36:40, donors] %*% wb))^2))
pos <- names(w)[w > 0]
att_j <- sapply(pos, function(j) { k <- setdiff(donors, j); wj <- section3(k); mean(Y[post, "treated"] - as.numeric(Y[post, k] %*% wj)) })
m <- which.max(abs(att_j - att))
out$lodo_donors_left_out <- length(pos); out$lodo_max_abs_att_change <- abs(att_j[[m]] - att)
out$lodo_donor <- names(att_j)[m]; out$lodo_donor_weight <- w[[names(att_j)[m]]]
out$weights <- as.list(w)
cat(toJSON(out, auto_unbox = TRUE, digits = NA), "\n")
