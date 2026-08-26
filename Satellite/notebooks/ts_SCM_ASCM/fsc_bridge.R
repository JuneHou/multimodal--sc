###-------------------------------------------------------------###
###  fsc_bridge.R — run the AUTHORS' FSC code on our panel        ###
###-------------------------------------------------------------###
#
# Thin CLI wrapper around Okano & Kurisu's reference implementation
# (https://github.com/.../FSC, local clone at FSC_DIR below). It contains NO
# FSC logic of its own: it reads a binary block of outcomes written by
# panel_fsc.py, rebuilds their `func_vals_list` structure, calls their
# unmodified functions in the same order their own scripts do
# (service.R for the covariance path, mortality.R for the B-spline path),
# and writes results as CSV for Python to read back.
#
# Their repository is READ-ONLY: we only setwd() into it and source().
#
# usage:  Rscript fsc_bridge.R <meta.txt> <data.bin> <out_prefix>
#
# meta.txt  key=value lines:
#   fsc_dir, op (fit|placebo), method (covmat|bspline),
#   n_groups, n_units, n_periods, M, T_0, K, lam_a, lam_b, lambda (or "cv"),
#   post_period, low, upp
# data.bin  float64, C-order, shape (n_groups, n_units, n_periods, M);
#           unit 1 of every group is the TREATED unit (their index-1 convention)

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
meta_path <- args[1]; data_path <- args[2]; out_prefix <- args[3]

## ---- read meta ----
kv <- read.table(meta_path, sep = "=", stringsAsFactors = FALSE,
                 col.names = c("k", "v"), colClasses = "character")
meta <- setNames(as.list(trimws(kv$v)), trimws(kv$k))
num <- function(k) as.numeric(meta[[k]])
int <- function(k) as.integer(num(k))

FSC_DIR <- meta$fsc_dir
setwd(FSC_DIR)
source("main_functions.R")          # their code, unmodified
suppressPackageStartupMessages({
  library(quadprog); library(Matrix); library(Rearrangement); library(cubicBsplines)
})

n_groups  <- int("n_groups"); n_units <- int("n_units")
n_periods <- int("n_periods"); M <- int("M"); T_0 <- int("T_0")
op <- meta$op; method <- meta$method

## ---- read the outcome block ----
con <- file(data_path, "rb")
v <- readBin(con, "double", n = n_groups * n_units * n_periods * M, size = 8)
close(con)
# C-order (groups, units, periods, M) -> R array with M fastest
A <- array(v, dim = c(M, n_periods, n_units, n_groups))

# their `grids`: service.R L24 is seq(0.01, 0.99, length = M); mortality.R L20 is
# seq(0.01, 0.99, length = 100) with M = 100 — i.e. both are "length = M". Use that,
# so the grid always matches the outcome vector regardless of representation.
grids <- seq(0.01, 0.99, length = M)
K <- if (!is.null(meta$K)) int("K") else 50

## =======================
## Projection back onto the outcome space (their scripts do this; see
## Docs/8-27-fsc-fidelity.md Issue 2). mortality.R L163 applies modif() to the augmented
## outcomes; service.R L105 applies Matrix::nearPD. `repr_kind` says which, and for the
## concatenated representations WHERE: modif must act per channel block, because
## rearranging the whole concatenation would sort values across channels.
## =======================
repr_kind <- if (!is.null(meta$repr_kind)) meta$repr_kind else "none"
n_ch      <- if (!is.null(meta$n_channels)) int("n_channels") else 5
n_grid    <- if (!is.null(meta$n_grid)) int("n_grid") else 0
n_gram    <- if (!is.null(meta$n_gram)) int("n_gram") else 0
low       <- if (!is.null(meta$low)) num("low") else 0
upp       <- if (!is.null(meta$upp)) num("upp") else 1

project_outcome <- function(v) {
  if (repr_kind == "none") return(v)
  out <- v
  nq <- n_ch * n_grid
  if (n_grid > 0) {                       # quantile block: modif per channel
    gq <- seq(0.01, 0.99, length = n_grid)
    for (c in 1:n_ch) {
      idx <- ((c - 1) * n_grid + 1):(c * n_grid)
      out[idx] <- as.vector(modif(vals = out[idx], low = low, upp = upp, grids = gq))
    }
  }
  if (n_gram > 0) {                       # gram block: reconstruct, nearPD, re-vectorise
    idx <- (nq + 1):(nq + n_gram)
    m <- matrix(0, nrow = n_ch, ncol = n_ch)
    m[upper.tri(m, diag = TRUE)] <- out[idx]
    m <- m + t(m); m <- m - diag(diag(m) / 2)
    pd <- as.matrix(nearPD(m, base.matrix = TRUE)$mat)
    out[idx] <- pd[upper.tri(pd, diag = TRUE)]
  }
  out
}

make_list <- function(g) {
  # rebuild func_vals_list exactly as their scripts hold it: unit -> period -> vector
  func_vals_list <- vector("list", n_units)
  for (i in 1:n_units) {
    per <- vector("list", n_periods)
    for (t in 1:n_periods) per[[t]] <- as.vector(A[, t, i, g])
    func_vals_list[[i]] <- per
  }
  func_vals_list
}

cv_fun <- function(lambda, fvl) {
  if (method == "covmat") cross_val_covmat(lambda, fvl, T_0, grids)
  else                    cross_val(lambda, fvl, T_0, K, grids)
}
aug_fun <- function(fvl, lambda) {
  if (method == "covmat") FSCM_aug_covmat(fvl, T_0, lambda = lambda, grids = grids)
  else                    FSCM_aug(fvl, T_0, K = K, lambda = lambda, grids = grids)
}

## =======================
## op = fit   (mirrors service.R L27-83 / mortality.R L110-165)
## =======================
if (op == "fit") {
  lam_a <- num("lam_a"); lam_b <- num("lam_b")
  # lambda modes:
  #   "cv_once"  select ONCE on group 1 and reuse - this mirrors their own scripts,
  #              which run optimise() once and then hardcode the value
  #              (service.R L56-61 "# 0.00186 is selected";
  #               mortality.R L143-148 "# 5.889182 is selected")
  #   "cv"       select per group (more than their usage; O((M*T_0)^2) per evaluation)
  #   <number>   use it directly
  lam_mode <- meta$lambda
  fixed_lambda <- if (lam_mode %in% c("cv", "cv_once")) NA else num("lambda")
  lambda_once <- NA

  w_rows <- list(); s_rows <- list()
  for (g in 1:n_groups) {
    func_vals_list <- make_list(g)
    N <- length(func_vals_list)

    # -- FSC --
    weight_fscm <- FSCM(func_vals_list, T_0)

    # -- select lambda by their optimise() over the interval, then augmented FSC --
    if (!is.na(fixed_lambda)) {
      lambda_opt <- fixed_lambda
    } else if (identical(lam_mode, "cv_once") && !is.na(lambda_once)) {
      lambda_opt <- lambda_once
    } else {
      obj_func <- function(lambda) cv_fun(lambda, func_vals_list)
      lambda_opt <- as.numeric(optimise(obj_func, interval = c(lam_a, lam_b))[1])
      if (identical(lam_mode, "cv_once")) lambda_once <- lambda_opt
    }
    weight_aug <- as.vector(aug_fun(func_vals_list, lambda_opt))

    # -- synthetic outcomes for every period, both estimators --
    # The augmented outcome is reported BOTH unprojected and projected back onto the
    # outcome space (modif / nearPD), so the size of the projection is visible rather
    # than assumed negligible. service.R scores unprojected; mortality.R L163 projects.
    for (t in 1:n_periods) {
      control_matrix <- matrix(0, nrow = N - 1, ncol = M)
      for (i in 2:N) control_matrix[i - 1, ] <- func_vals_list[[i]][[t]]
      scm_out  <- as.vector(t(weight_fscm) %*% control_matrix)
      ascm_out <- as.vector(t(weight_aug)  %*% control_matrix)
      ascm_prj <- project_outcome(ascm_out)
      scm_prj  <- project_outcome(scm_out)
      obs      <- func_vals_list[[1]][[t]]
      s_rows[[length(s_rows) + 1]] <- data.frame(
        group = g, period = t,
        rmse_fsc       = sqrt(mean((obs - scm_out)^2)),
        rmse_fsc_proj  = sqrt(mean((obs - scm_prj)^2)),
        rmse_afsc      = sqrt(mean((obs - ascm_out)^2)),
        rmse_afsc_proj = sqrt(mean((obs - ascm_prj)^2)),
        proj_shift_afsc = sqrt(mean((ascm_out - ascm_prj)^2)),
        proj_shift_fsc  = sqrt(mean((scm_out - scm_prj)^2)))
    }

    w_rows[[length(w_rows) + 1]] <- data.frame(
      group = g, donor = 1:(N - 1),
      weight_fsc = weight_fscm, weight_afsc = weight_aug,
      lambda = lambda_opt,
      lambda_at_bound = (abs(lambda_opt - lam_a) < 1e-8) | (abs(lambda_opt - lam_b) < 1e-8),
      sum_w_fsc = sum(weight_fscm), sum_w_afsc = sum(weight_aug),
      n_nonzero_fsc = sum(weight_fscm != 0), n_negative_afsc = sum(weight_aug < 0))
    cat(sprintf("group %d/%d  lambda=%.6g  nnz_fsc=%d  neg_afsc=%d\n",
                g, n_groups, lambda_opt, sum(weight_fscm != 0), sum(weight_aug < 0)))
  }
  write.csv(do.call(rbind, w_rows), paste0(out_prefix, "_weights.csv"), row.names = FALSE)
  write.csv(do.call(rbind, s_rows), paste0(out_prefix, "_fit.csv"), row.names = FALSE)
}

## =======================
## op = lambda_grid   (DISCLOSED DEVIATION — see Docs/8-27-fsc-fidelity.md Issue 1)
##
## Their scripts pick lambda with optimise() on a LINEAR interval, c(0,1) / c(0,10),
## chosen for their outcome scales. On ours that interval does not bracket the optimum:
## lambda landed on a boundary in 20 of 28 configurations. The CV objective evaluated
## here is THEIRS, unmodified (cross_val / cross_val_covmat); only the search changes,
## from linear optimise() to an explicit log grid whose optimum can be checked for
## interiority.
## =======================
if (op == "lambda_grid") {
  lams <- scan(paste0(out_prefix, "_lambdas.txt"), quiet = TRUE)
  rows <- list()
  for (g in 1:n_groups) {
    func_vals_list <- make_list(g)
    for (lam in lams) {
      # a lambda small enough to leave solve() singular is a failed grid point, not a
      # failed run: record NA and carry on rather than aborting the whole search
      cv <- tryCatch(cv_fun(lam, func_vals_list), error = function(e) NA_real_)
      rows[[length(rows) + 1]] <- data.frame(group = g, lambda = lam, cv = cv)
    }
    cat(sprintf("lambda_grid group %d/%d done (%d lambdas)\n", g, n_groups, length(lams)))
  }
  write.csv(do.call(rbind, rows), paste0(out_prefix, "_lamgrid.csv"), row.names = FALSE)
}

## =======================
## op = placebo   (their placebo / placebo_covmat, service.R L463-469)
## =======================
if (op == "placebo") {
  lam_a <- num("lam_a"); lam_b <- num("lam_b")
  # Their placebo() re-runs optimise() for EVERY rotated unit, which dominates the cost
  # (cross_val_covmat is O((M*T_0)^2): 6.5 s at M=45, 218 s at M=500). Their own scripts
  # instead select lambda once and hardcode it. Passing a lambda here reproduces that
  # through their unmodified API: a half-width of 1e-9 makes optimise() return that
  # value after 2 evaluations instead of ~7.
  if (!(meta$lambda %in% c("cv", "cv_once"))) {
    lam <- num("lambda")
    lam_a <- lam * (1 - 1e-9); lam_b <- lam * (1 + 1e-9)
    cat(sprintf("placebo: lambda pinned at %.8g via interval half-width 1e-9\n", lam))
  }
  post_period <- int("post_period")
  rows <- list()
  for (g in 1:n_groups) {
    func_vals_list <- make_list(g)
    d_vals <- if (method == "covmat")
      placebo_covmat(func_vals_list, T_0, grids, post_period, lam_a, lam_b)
    else
      placebo(func_vals_list, T_0, K, grids, post_period, lam_a, lam_b)
    rows[[length(rows) + 1]] <- data.frame(
      group = g, unit = 1:length(d_vals), magnitude = as.vector(d_vals),
      post_period = post_period, is_treated = c(TRUE, rep(FALSE, length(d_vals) - 1)))
    cat(sprintf("placebo group %d/%d  post_period=%d done\n", g, n_groups, post_period))
  }
  write.csv(do.call(rbind, rows), paste0(out_prefix, "_placebo.csv"), row.names = FALSE)
}

cat("fsc_bridge.R done\n")
