# The checks of spec.json in scm-four-design (and of settings.json in scm-four-configure), the same rules as the
# Python checks of four_design/method.py (scs_spec.validate with the field kind "option", and _rules). A refusal names
# the same fields in both, which four_design/checks.py tests. Plan use-nature-skills-analyze-wiggly-coral.md, section
# "Fields of scm-four-design".
suppressPackageStartupMessages(library(jsonlite))

FOUR_DONORS <- sprintf("donor_%02d", 1:29)
FOUR_OPTIONS <- list(
  predictors = list(every_outcome = c("covariate_periods"), summaries = c("covariate_periods", "summaries")),
  v = list(judgement = c("values"), pre_period_fit = character(0), validation = c("training_end")),
  weights = list(convex = character(0), extrapolation = character(0)),
  penalty = list(none = character(0), distance = c("lambda")))
# what scm-four-configure fixes: the package's reference design
FOUR_REFERENCE <- list(predictors = list(option = "every_outcome", covariate_periods = list(40L)),
                       v = list(option = "pre_period_fit"), weights = list(option = "convex"),
                       penalty = list(option = "none"))

four_is_object <- function(v) is.list(v) && !is.null(names(v))
four_is_array <- function(v) is.list(v) && is.null(names(v))
four_is_number <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
four_shown <- function(v) if (is.null(v)) "null" else as.character(toJSON(v, auto_unbox = TRUE, null = "null"))

four_periods_ok <- function(v) four_is_array(v) && length(v) > 0 &&
  all(vapply(v, function(x) four_is_number(x) && x == round(x) && x >= 1 && x <= 40, TRUE))

four_summary_ok <- function(s) {
  if (!four_is_object(s) || !length(s)) return(FALSE)
  keys_ok <- all(grepl("^[1-9][0-9]*$", names(s))) && all(as.integer(names(s)) %in% 1:40)
  keys_ok && all(vapply(s, four_is_number, TRUE)) && any(vapply(s, function(w) w != 0, TRUE))
}

# One choice: NULL when it is well formed, else its error (the first, as the Python typing stops at the first).
four_check_choice <- function(f, v) {
  opts <- FOUR_OPTIONS[[f]]
  if (!(four_is_object(v) && "option" %in% names(v)))
    return(sprintf("field '%s': expected an object with \"option\" and the values that option carries, got %s",
                   f, four_shown(v)))
  o <- v$option
  if (!(is.character(o) && length(o) == 1 && o %in% names(opts)))
    return(sprintf("field '%s': option %s is not one of %s", f, four_shown(o), paste(names(opts), collapse = ", ")))
  got <- setdiff(names(v), "option")
  want <- opts[[o]]
  if (!setequal(got, want))
    return(sprintf("field '%s': option '%s' carries %s, got %s", f, o,
                   if (length(want)) paste0("'", want, "'", collapse = ", ") else "no values",
                   if (length(got)) paste0("'", got, "'", collapse = ", ") else "none"))
  if (f == "v" && o == "judgement") {
    x <- v$values
    if (!(four_is_array(x) && length(x) > 0 && all(vapply(x, function(y) four_is_number(y) && y > 0, TRUE))))
      return(sprintf("field 'v.values': expected a non-empty list of numbers > 0, got %s", four_shown(x)))
  }
  if (f == "v" && o == "validation") {
    x <- v$training_end
    if (!(four_is_number(x) && x == round(x))) return(sprintf("field 'v.training_end': expected an integer, got %s",
                                                             four_shown(x)))
    if (!(x >= 1 && x <= 39)) return(sprintf("field 'v.training_end': %s is outside the allowed range", four_shown(x)))
  }
  if (f == "penalty" && o == "distance") {
    x <- v$lambda
    if (!four_is_number(x)) return(sprintf("field 'penalty.lambda': expected a number, got %s", four_shown(x)))
    if (!(x >= 0)) return(sprintf("field 'penalty.lambda': %s is outside the allowed range", four_shown(x)))
  }
  NULL
}

check_four <- function(given, mode) {
  if (!four_is_object(given)) return("spec must be a JSON object")     # {} is an object here, [] is not
  if (mode == "configure") {
    present <- intersect(names(given), names(FOUR_OPTIONS))
    if (length(present))
      return(paste0("protocol violation: scm-four-configure takes Class-1 fields only, but got ",
                    paste(present, collapse = ", ")))
    unknown <- names(given)
    return(if (length(unknown)) paste0("unknown field(s): ", paste(unknown, collapse = ", ")) else character(0))
  }
  errors <- character(0)
  missing <- setdiff(names(FOUR_OPTIONS), names(given))
  if (length(missing)) errors <- c(errors, paste0("missing required field(s): ", paste(missing, collapse = ", ")))
  unknown <- setdiff(names(given), names(FOUR_OPTIONS))
  if (length(unknown)) errors <- c(errors, paste0("unknown field(s): ", paste(unknown, collapse = ", ")))
  ok <- list()
  for (f in names(FOUR_OPTIONS)) {
    if (!(f %in% names(given))) next
    e <- four_check_choice(f, given[[f]])
    if (is.null(e)) ok[[f]] <- given[[f]] else errors <- c(errors, e)
  }
  p <- ok$predictors; v <- ok$v
  # rule 1: every period is a whole number from 1 to 40
  if (!is.null(p)) {
    if (!four_periods_ok(p$covariate_periods)) {
      errors <- c(errors, "field 'predictors.covariate_periods': expected a non-empty list of whole periods from 1 to 40")
      p <- NULL
    } else if (identical(p$option, "summaries") &&
               !(four_is_array(p$summaries) && length(p$summaries) > 0 && all(vapply(p$summaries, four_summary_ok, TRUE)))) {
      errors <- c(errors, paste0("field 'predictors.summaries': expected a non-empty list of objects, each mapping ",
                                 "periods from \"1\" to \"40\" to numbers, at least one not 0"))
      p <- NULL
    }
  }
  # rule 3: under validation every period of the predictors is at most training_end
  if (!is.null(p) && !is.null(v) && identical(v$option, "validation")) {
    used <- as.numeric(unlist(p$covariate_periods))
    if (identical(p$option, "summaries")) used <- c(used, as.numeric(unlist(lapply(p$summaries, names))))
    if (any(used > v$training_end))
      errors <- c(errors, "fields 'predictors','v': under validation every period of the predictors is at most training_end")
  }
  # rule 4: under judgement, one number in values for each predictor
  if (!is.null(p) && !is.null(v) && identical(v$option, "judgement")) {
    n <- 2 + if (identical(p$option, "summaries")) length(p$summaries) else 40
    if (length(v$values) != n)
      errors <- c(errors, sprintf("field 'v.values': %d numbers for %d predictors", length(v$values), n))
  }
  # rule 5: penalty distance requires weights convex
  if (!is.null(ok$penalty) && !is.null(ok$weights) && identical(ok$penalty$option, "distance") &&
      identical(ok$weights$option, "extrapolation"))
    errors <- c(errors, "fields 'penalty','weights': penalty distance requires weights convex")
  errors
}
