# =============================================================================
# ASCM per treatment site, on the weekly / biweekly / monthly conus_random10
# panels built by 00_build_panels.py.
#
# Derived from scripts/validation/ascm_r_package/02b_fit_ascm_bysite.R. The
# estimator, the SCM/ASCM pair and the output schemas follow that script, so
# results stay comparable with the run already on disk. Three things differ:
#
#   1. PER-SITE GRIDS. 02b asserts one rectangular panel and dies if it has a
#      gap. Nothing sub-monthly can satisfy that -- 0 of 89 treated sites are
#      complete on every biweekly or weekly period. But each treated unit is
#      promoted in turn against the donor pool, so the only balance augsynth
#      needs is within THAT site plus ITS donors. Each site is therefore fitted
#      on the periods it observed, against the donors complete on those
#      periods, both read from the index files 00_build_panels.py writes. The
#      sub-panel is rectangular; the panel as a whole is not, and need not be.
#
#   2. CWZ CONFORMAL INFERENCE, NOT JACKKNIFE+. Chernozhukov, Wuthrich & Zhu
#      conformal inference, via summary(inf_type = "conformal", type = "block"),
#      which cyclically shifts residuals and so keeps the time ordering. It
#      returns what the figures actually need and jackknife+ never gave:
#      a per-period estimate, a pointwise interval and a per-period p-value,
#      plus a joint-null p-value on the average.
#
#      READ THE AVERAGE p CAREFULLY. average_att$p_val tests the JOINT null of
#      zero effect at EVERY post-treatment date. It is not a test of whether
#      the average effect alone is zero, and it is stored as
#      conformal_joint_p to keep that straight.
#
#      The exact cyclic-shift test has a minimum p of 1/(T0 + 1): 0.071 on the
#      13 pre-period months, 0.04 or finer at the shorter cadences. grid_size
#      buys interval RESOLUTION, not statistical information -- the p-value was
#      identical at grid 20 and grid 50 in testing -- so it is a cost knob.
#
#   3. CWZ IS THE ONLY INFERENCE REPORTED. The donor-placebo null, the t
#      interval on the cross-site mean and the leave-one-site-out jackknife
#      are all gone. Each answered a different question with a different null,
#      and carrying several in one table invites quoting whichever is
#      smallest. Where conformal licenses no statement -- notably an interval
#      on the mean of 89 sites, which is not what a per-site procedure
#      produces -- none is made.
#
# THE EVENT PERIOD is pre or post by cadence, decided in 00_build_panels.py and
# read from the panel's phase column -- never recomputed from date > EVENT,
# which would silently flip monthly September to post.
#
# Fitting only -- no figures/maps. (The source script also calls _figures.py
# at the end; that step, and the figures it makes, are not part of this copy.)
#
# Input : test results/validation/hls_flooding_128/hls_2y_random_control/panels/
# Output: test results/validation/hls_flooding_128/hls_2y_random_control/ascm/<cadence>/tables/
#
#   Rscript 02_ascm_hls_2y.R                      # all three cadences
#   Rscript 02_ascm_hls_2y.R --cadence weekly --grid-size 100
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

flag <- function(name, default) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
has_flag <- function(name) name %in% args

CALS <- "/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS"
ROOT <- file.path(CALS, "test results", "validation", "hls_flooding_128", "hls_2y_random_control")
PANELS <- file.path(ROOT, "panels")
DESIGN <- "conus_random10"

cadences <- if (has_flag("--cadence")) {
  strsplit(flag("--cadence", ""), ",")[[1]]
} else {
  c("monthly", "biweekly", "weekly")
}

CONF_GRID <- if (has_flag("--grid-size")) {
  as.integer(flag("--grid-size", "100"))
} else {
  100L
}
# Detects post-period deviation in EITHER direction, as CWZ intend.
CONF_STAT <- function(x) sum(abs(x))

for (p in c("augsynth", "dplyr", "readr"))
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
suppressMessages({library(augsynth); library(dplyr); library(readr)})

set.seed(2026)
EVENT <- as.Date("2024-09-27")
DAMAGE_END <- as.Date("2025-01-01")   # the three months the request asks about
WTOL <- 1e-4   # below this is augsynth's solver noise, not a donor weight


fit_cadence <- function(cadence) {

  message(sprintf("\n=== ASCM %s ===", cadence))
  out_root <- file.path(ROOT, "ascm", cadence)
  tables <- file.path(out_root, "tables")
  dir.create(tables, recursive = TRUE, showWarnings = FALSE)
  tpath <- function(name) file.path(tables, paste0(DESIGN, "_", name, ".csv"))

  dat <- read_csv(file.path(PANELS, sprintf("%s_%s_ndvi.csv", DESIGN, cadence)),
                  show_col_types = FALSE) %>%
    mutate(site = as.character(site), date = as.Date(date),
           ndvi = as.numeric(ndvi), candidate = as.integer(candidate))
  grids <- read_csv(file.path(PANELS, sprintf("site_grids_%s.csv", cadence)),
                    show_col_types = FALSE)
  site_periods <- read_csv(
    file.path(PANELS, sprintf("site_periods_%s.csv", cadence)),
    show_col_types = FALSE) %>% mutate(date = as.Date(date))
  site_donors <- read_csv(
    file.path(PANELS, sprintf("site_donors_%s.csv", cadence)),
    show_col_types = FALSE)

  if (any(dat$ndvi < -1 | dat$ndvi > 1)) stop("NDVI outside [-1,1]")
  if (any(duplicated(dat[, c("site", "date")]))) stop("Duplicate site/date rows")

  donors_only <- dat %>% filter(candidate == 0)
  donor_values <- donors_only %>% select(site, date, ndvi)
  treated_sites <- sort(grids$site)
  all_donors <- sort(unique(donors_only$site))

  periods_of <- split(site_periods$date, site_periods$site)
  phase_of <- split(site_periods$phase, site_periods$site)
  donors_of <- split(site_donors$donor, site_donors$site)

  message(sprintf("%s: %d treated sites, %d donors in the pool, %d-%d periods per site",
                  cadence, length(treated_sites), length(all_donors),
                  min(grids$n_periods), max(grids$n_periods)))

  # ------------------------------------------------------------ one unit ----
  # One treated site on its own grid, against the donors complete on it.
  fit_unit <- function(unit) {

    dates <- periods_of[[unit]]
    phase <- phase_of[[unit]]
    post_mask <- phase == "post"
    pool_ids <- setdiff(donors_of[[unit]], unit)
    # Three, not ten. Each site now gets only the ten donors the design
    # assigned it, so ten is the CEILING, not a floor -- a site that lost one
    # donor to cloud would otherwise be dropped outright. The builder already
    # applies the same floor when it writes site_donors_<cadence>.csv.
    if (length(pool_ids) < 3) stop("donor pool too small for ", unit)

    pool <- donor_values %>%
      filter(site %in% pool_ids, date %in% dates)
    target <- dat %>% filter(site == unit, date %in% dates) %>%
      select(site, date, ndvi)
    if (nrow(target) != length(dates))
      stop(unit, " is not complete on its own grid")

    sub <- bind_rows(pool, target) %>%
      mutate(treated = as.integer(site == unit),
             period = match(date, dates)) %>%
      arrange(site, period)
    if (nrow(sub) != length(dates) * length(unique(sub$site)))
      stop("sub-panel not rectangular for ", unit)

    first_post <- min(which(post_mask))
    observed <- sub %>% filter(treated == 1) %>% arrange(period) %>% pull(ndvi)

    scm  <- augsynth(ndvi ~ treated, site, period, sub,
                     t_int = first_post, progfunc = "none",  scm = TRUE)
    ascm <- augsynth(ndvi ~ treated, site, period, sub,
                     t_int = first_post, progfunc = "ridge", scm = TRUE)

    y_scm  <- as.numeric(predict(scm,  att = FALSE))
    y_ascm <- as.numeric(predict(ascm, att = FALSE))
    if (length(y_scm) != length(dates) || length(y_ascm) != length(dates))
      stop("Prediction length mismatch for ", unit)

    e_scm  <- observed - y_scm
    e_ascm <- observed - y_ascm

    weights <- full_join(
      data.frame(donor = rownames(scm$weights),
                 scm_weight = as.numeric(scm$weights)),
      data.frame(donor = rownames(ascm$weights),
                 ascm_weight = as.numeric(ascm$weights)),
      by = "donor")

    # CWZ conformal, block variant: cyclic shifts of the residuals.
    cwz <- tryCatch(
      summary(ascm, inf_type = "conformal", alpha = 0.05, type = "block",
              stat_func = CONF_STAT, grid_size = CONF_GRID),
      error = function(e) {
        warning(sprintf("conformal failed for %s: %s", unit,
                        conditionMessage(e)))
        NULL
      })

    per_period <- data.frame(period = seq_along(dates),
                             conformal_estimate = NA_real_,
                             conformal_lower = NA_real_,
                             conformal_upper = NA_real_,
                             conformal_p = NA_real_)
    joint_p <- NA_real_; conf_att <- NA_real_
    if (!is.null(cwz)) {
      tab <- cwz$att
      idx <- match(per_period$period, tab$Time)
      per_period$conformal_estimate <- tab$Estimate[idx]
      per_period$conformal_lower <- tab$lower_bound[idx]
      per_period$conformal_upper <- tab$upper_bound[idx]
      per_period$conformal_p <- tab$p_val[idx]
      # JOINT null over every post date, not a test of the average alone.
      joint_p <- as.numeric(cwz$average_att$p_val[1])
      conf_att <- as.numeric(cwz$average_att$Estimate[1])
    }

    pre_scm   <- sqrt(mean(e_scm[!post_mask]^2))
    pre_ascm  <- sqrt(mean(e_ascm[!post_mask]^2))
    post_scm  <- sqrt(mean(e_scm[post_mask]^2))
    post_ascm <- sqrt(mean(e_ascm[post_mask]^2))

    # The three months the request asks about, inside the full post window.
    damage <- post_mask & dates < DAMAGE_END

    list(
      effects = data.frame(
        design = DESIGN, cadence = cadence, site = unit, date = dates,
        period = seq_along(dates), post = post_mask, damage = damage,
        observed = observed, scm_counterfactual = y_scm,
        ascm_counterfactual = y_ascm, scm_effect = e_scm,
        ascm_effect = e_ascm,
        conformal_lower = per_period$conformal_lower,
        conformal_upper = per_period$conformal_upper,
        conformal_p = per_period$conformal_p),
      weights = weights %>% mutate(design = DESIGN, cadence = cadence,
                                   site = unit) %>%
        select(design, cadence, site, donor, scm_weight, ascm_weight),
      summary = data.frame(
        design = DESIGN, cadence = cadence, site = unit,
        donors_offered = length(pool_ids), n_periods = length(dates),
        n_pre = sum(!post_mask), n_post = sum(post_mask),
        n_damage = sum(damage),
        lambda = if (is.null(ascm$lambda)) NA_real_ else as.numeric(ascm$lambda),
        scm_donors_selected  = sum(weights$scm_weight > WTOL, na.rm = TRUE),
        ascm_donors_selected = sum(abs(weights$ascm_weight) > WTOL, na.rm = TRUE),
        ascm_donors_negative = sum(weights$ascm_weight < -WTOL, na.rm = TRUE),
        scm_top_donor_weight = max(weights$scm_weight, na.rm = TRUE),
        scm_pre_rmse = pre_scm, ascm_pre_rmse = pre_ascm,
        scm_att = mean(e_scm[post_mask]), ascm_att = mean(e_ascm[post_mask]),
        scm_att_damage = if (any(damage)) mean(e_scm[damage]) else NA_real_,
        ascm_att_damage = if (any(damage)) mean(e_ascm[damage]) else NA_real_,
        scm_post_rmse = post_scm, ascm_post_rmse = post_ascm,
        scm_post_pre_ratio  = post_scm / pre_scm,
        ascm_post_pre_ratio = post_ascm / pre_ascm,
        conformal_att = conf_att,
        conformal_joint_p = joint_p,
        conformal_grid_size = CONF_GRID,
        # The exact cyclic-shift test cannot go below this.
        conformal_p_floor = 1 / (sum(!post_mask) + 1),
        # Mean of the pointwise conformal bounds over the damage window: a
        # band for the 3-month average, not an interval the estimator claims.
        conformal_damage_lower = if (any(damage))
          mean(per_period$conformal_lower[damage], na.rm = TRUE) else NA_real_,
        conformal_damage_upper = if (any(damage))
          mean(per_period$conformal_upper[damage], na.rm = TRUE) else NA_real_,
        conformal_damage_min_p = if (any(damage))
          suppressWarnings(min(per_period$conformal_p[damage], na.rm = TRUE))
          else NA_real_))
  }

  # -------------------------------------------------------- treated sites ---
  eff <- list(); wts <- list(); sm <- list(); failed <- character()
  for (i in seq_along(treated_sites)) {
    site <- treated_sites[i]
    res <- tryCatch(fit_unit(site),
                    error = function(e) {
                      warning(sprintf("%s %s failed: %s", cadence, site,
                                      conditionMessage(e)))
                      NULL
                    })
    if (is.null(res)) { failed <- c(failed, site); next }
    eff[[site]] <- res$effects
    wts[[site]] <- res$weights
    sm[[site]] <- res$summary
    if (i %% 10 == 0 || i == length(treated_sites))
      message(sprintf("  %s  %d/%d sites", cadence, i, length(treated_sites)))
  }

  site_effects <- bind_rows(eff)
  site_weights <- bind_rows(wts)
  site_summary <- bind_rows(sm)
  if (!nrow(site_summary)) stop("Every site failed for ", cadence)
  if (length(failed))
    writeLines(failed, file.path(tables, paste0(DESIGN, "_failed_sites.txt")))

  # NO DONOR PLACEBOS, NO t INTERVAL, NO JACKKNIFE. CWZ conformal is the
  # inference, and mixing a second null into the same tables invites reading
  # whichever one happens to be smaller. What the placebo p answered -- is
  # this site's post/pre ratio extreme against donors fitted as if treated --
  # conformal answers directly, per period and on the joint null.
  write_csv(site_summary, tpath("site_summary"))
  write_csv(site_effects, tpath("site_effects"))
  write_csv(site_weights %>%
              filter(abs(scm_weight) > WTOL | abs(ascm_weight) > WTOL),
            tpath("site_donor_weights"))

  site_inference <- site_summary %>%
    select(design, cadence, site, donors_offered, n_periods, n_pre, n_post,
           scm_donors_selected, ascm_donors_selected, ascm_donors_negative,
           scm_pre_rmse, ascm_pre_rmse, scm_att, ascm_att,
           scm_att_damage, ascm_att_damage,
           scm_post_pre_ratio, ascm_post_pre_ratio,
           conformal_att, conformal_joint_p, conformal_p_floor,
           conformal_grid_size, conformal_damage_lower,
           conformal_damage_upper, conformal_damage_min_p) %>%
    arrange(ascm_att_damage)
  write_csv(site_inference, tpath("site_inference"))

  donor_mean <- site_weights %>%
    group_by(design, cadence, donor) %>%
    summarise(mean_scm_weight = mean(scm_weight, na.rm = TRUE),
              mean_ascm_weight = mean(ascm_weight, na.rm = TRUE),
              n_sites_using = sum(abs(ascm_weight) > WTOL, na.rm = TRUE),
              max_ascm_weight = max(ascm_weight, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(desc(abs(mean_ascm_weight)))
  write_csv(donor_mean, tpath("donor_weight_mean"))

  # --------------------------------------------- average effect by period ---
  # n_sites varies by date, because the grids do. It is carried so a thin
  # period is visible as thin rather than averaged away silently.
  # Bands here are the MEAN of the per-site CONFORMAL bounds at each period,
  # not a t interval on the cross-site mean. The spread across sites is still
  # reported as sd, but it is not dressed up as an interval.
  att_by_period <- site_effects %>%
    group_by(design, cadence, date, post, damage) %>%
    summarise(n_sites = n(),
              mean_observed = mean(observed),
              mean_scm_counterfactual = mean(scm_counterfactual),
              mean_ascm_counterfactual = mean(ascm_counterfactual),
              mean_scm_effect = mean(scm_effect),
              sd_scm_effect = sd(scm_effect),
              mean_ascm_effect = mean(ascm_effect),
              sd_ascm_effect = sd(ascm_effect),
              median_ascm_effect = median(ascm_effect),
              mean_conformal_lower = mean(conformal_lower, na.rm = TRUE),
              mean_conformal_upper = mean(conformal_upper, na.rm = TRUE),
              n_sites_conformal_p_lt05 = sum(conformal_p < 0.05, na.rm = TRUE),
              n_sites_conformal_available = sum(!is.na(conformal_p)),
              .groups = "drop")
  write_csv(att_by_period, tpath("att_by_period"))

  # ----------------------------------------- the average across sites ------
  # CWZ conformal is a PER-SITE procedure. It licenses no interval on the
  # mean of 89 sites, so none is invented here: the cross-site row carries
  # the point estimate and the spread, and every inferential statement is a
  # count of what conformal itself returned. A t interval or a jackknife on
  # this mean would be a second, unrelated null sitting in the same table.
  describe <- function(estimator, att, pre_rmse, window) {
    data.frame(
      design = DESIGN, cadence = cadence, estimator = estimator,
      window = window, n_sites = length(att),
      mean_att = mean(att, na.rm = TRUE),
      median_att = median(att, na.rm = TRUE),
      sd_att = sd(att, na.rm = TRUE),
      n_sites_positive = sum(att > 0, na.rm = TRUE),
      n_sites_negative = sum(att < 0, na.rm = TRUE),
      mean_pre_rmse = mean(pre_rmse, na.rm = TRUE),
      inference = "CWZ conformal (per site); no interval on this mean",
      conformal_grid_size = CONF_GRID,
      median_conformal_p_floor = stats::median(site_summary$conformal_p_floor,
                                               na.rm = TRUE),
      n_sites_conformal_joint_p_lt05 =
        sum(site_summary$conformal_joint_p < 0.05, na.rm = TRUE),
      n_sites_conformal_joint_available =
        sum(!is.na(site_summary$conformal_joint_p)),
      n_sites_any_period_p_lt05 =
        sum(site_summary$conformal_damage_min_p < 0.05, na.rm = TRUE))
  }

  att_inference <- bind_rows(
    describe("scm",  site_summary$scm_att,  site_summary$scm_pre_rmse,
             "full_post"),
    describe("ascm", site_summary$ascm_att, site_summary$ascm_pre_rmse,
             "full_post"),
    describe("scm",  site_summary$scm_att_damage,  site_summary$scm_pre_rmse,
             "three_month"),
    describe("ascm", site_summary$ascm_att_damage, site_summary$ascm_pre_rmse,
             "three_month"))
  write_csv(att_inference, tpath("att_inference"))

  message(sprintf("  full-post ASCM ATT %+.4f | 3-month %+.4f  (%d sites)",
                  mean(site_summary$ascm_att, na.rm = TRUE),
                  mean(site_summary$ascm_att_damage, na.rm = TRUE),
                  nrow(site_summary)))
  message(sprintf("  conformal: joint p < .05 at %d of %d sites, p floor %.4f, grid %d",
                  sum(site_summary$conformal_joint_p < 0.05, na.rm = TRUE),
                  nrow(site_summary),
                  stats::median(site_summary$conformal_p_floor, na.rm = TRUE),
                  CONF_GRID))
  invisible(NULL)
}

for (cadence in cadences) fit_cadence(cadence)

message("\ndone")
