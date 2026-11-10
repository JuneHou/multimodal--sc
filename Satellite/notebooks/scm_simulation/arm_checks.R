# The collaborator's checks (data/simulation/README.md Section 4.2 and the leave-one-donor-out part of 4.3) for one
# answer's specification, fitted with Synth 1.1-10 as the answer's fit was (synth_fit.R).
#   Rscript arm_checks.R <panel.csv> <spec.json>              the fit: its synthetic path, weights, ATT and 4.2
#   Rscript arm_checks.R <panel.csv> <spec.json> <donor>      the spec refitted without <donor>: its ATT
# 4.2  pre-treatment RMSPE over 1..40 and covariate balance sqrt(1/2 sum_r (Z_1r,40 - sum_j w_j Z_jr,40)^2), from
#      the fit's donor weights
# 4.3  leave one donor out is run by the caller over every donor whose weight in the fit behind the answer is above
#      0.001, as the collaborator's code counts donors (mc_evaluation_r.R lines 105 and 148); ATT(-j) is the mean gap
#      over 41..60
# Every fit runs as the fit behind the answer ran (plan use-nature-skills-analyze-wiggly-coral.md, "refit engine"):
# a spec's outcome_combinations (scm-four-design) are added to the panel first, and Synth fits when the weights are
# convex and the penalty is none, the native engine (scm_native.R) otherwise (four_design/four_translate.R four_fit).
# A spec without those fields is fitted by Synth exactly as before.
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "synth_fit.R"))
source(file.path(here, "scm_native.R"))
source(file.path(here, "four_design", "four_translate.R"))
a <- commandArgs(trailingOnly = TRUE)
d <- scm_data(a[1])
s <- fromJSON(a[2], simplifyVector = FALSE)
post <- as.character(41:60); pre <- as.character(1:40)

rec <- four_fit(d, s)
donors <- unlist(rec$donor_ids)
w <- setNames(as.numeric(unlist(rec$weights)), donors)
path <- unlist(rec$predictions); gaps <- unlist(rec$gaps)
att <- mean(gaps[post])
z40 <- d[d$period == 40, ]; rownames(z40) <- z40$unit_id
zb <- sapply(c("Z1", "Z2"), function(r) z40["treated", r] - sum(w * z40[donors, r]))

if (length(a) >= 3) {
  s2 <- s; s2$controls_identifier <- as.list(setdiff(unlist(s$controls_identifier), a[3]))
  cat(scm_json(list(donor = a[3], att = mean(unlist(four_fit(d, s2)$gaps[post])))), "\n")
} else {
  cat(scm_json(list(predictions = as.list(path), att = att, weights = as.list(w),
                    pre_rmspe = sqrt(mean(gaps[pre]^2)), covariate_balance = sqrt(mean(zb^2)))), "\n")
}
