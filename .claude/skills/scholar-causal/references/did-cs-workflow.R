# =============================================================================
# Callaway & Sant'Anna (2021) Staggered DiD — Canonical Workflow Template
#
# Source vignettes (all by Brantly Callaway):
#   https://bcallaway11.github.io/did/articles/did-basics.html
#   https://bcallaway11.github.io/did/articles/multi-period-did.html
#   https://bcallaway11.github.io/did/articles/pre-testing.html
#
# Citation:
#   Callaway, Brantly and Pedro H.C. Sant'Anna (2021).
#   "Difference-in-Differences with Multiple Time Periods."
#   Journal of Econometrics, 225(2), 200-230.
#
# Why this estimator: Goodman-Bacon (2021, J. Econometrics 225(2):254-277)
# decomposes the standard two-way fixed-effects DiD estimator into a weighted
# average of all possible 2x2 comparisons. Under staggered adoption AND
# heterogeneous-over-time treatment effects, one of those 2x2s compares newly
# treated units against already-treated units, where the "control" path
# itself contains the already-treated unit's evolving treatment effect.
# That comparison can receive a negative implicit weight, and under
# sufficient heterogeneity the aggregate TWFE estimate can have the wrong
# sign relative to every individual ATT. The CS-2021 estimator avoids this
# contamination by restricting comparisons to never-treated or
# not-yet-treated units. Under homogeneous-over-time effects, TWFE is
# unbiased — the problem is heterogeneity, not the comparison per se.
#
# Critical pre-test caveat from the vignette:
#   "Whether or not the parallel trends assumption holds in pre-treatment
#   periods does not actually tell you if it holds in the current period
#   (and this is when you need it to hold!)" — pre-tests provide
#   credibility evidence, not definitive validation.
# =============================================================================

# install.packages("did")
library(did)

# -----------------------------------------------------------------------------
# Section 1: Simulated example for sanity-checking the workflow
# -----------------------------------------------------------------------------

set.seed(1814)
sp <- reset.sim()
sp$te <- 0
time.periods <- 4
sp$te.e <- 1:time.periods
dta <- build_sim_dataset(sp)

nrow(dta)
head(dta)
# Columns: G (treatment group, 0 = never treated), X (covariate),
#          id (unit), cluster, period, Y (outcome), treat.

# -----------------------------------------------------------------------------
# Section 2: Estimate group-time average treatment effects ATT(g,t)
# -----------------------------------------------------------------------------
# Default est_method = "dr" (doubly robust); alternatives: "ipw", "reg".
# Default control_group = "nevertreated"; alternative: "notyettreated".
# xformla = ~X adjusts for time-invariant covariate X (conditional PT).

example_attgt <- att_gt(
  yname   = "Y",
  tname   = "period",
  idname  = "id",
  gname   = "G",
  xformla = ~ X,
  data    = dta
)
summary(example_attgt)
# Reports group-time effects ATT(g,t) with simultaneous confidence bands
# AND a pre-test p-value for the parallel trends assumption.

# Plot: separate subplot per cohort; red = pre-treatment pseudo-effect,
#       blue = post-treatment effect with 95% simultaneous CI.
ggdid(example_attgt)

# -----------------------------------------------------------------------------
# Section 3: Aggregate ATT(g,t) into reporting parameters
# -----------------------------------------------------------------------------
# CS-2021 recommend reporting BOTH a dynamic (event-study) aggregation
# and at least one of {simple, group, calendar}. These are DIFFERENT
# ESTIMANDS, not different summaries of the same estimand:
#   - simple   = average of all post-treatment ATT(g,t), weighted by
#                group size AND exposure length (overweights early cohorts).
#   - group    = average within each cohort, then average across cohorts
#                with cohort weights — overweighting of early cohorts removed.
#   - calendar = average within each calendar period across treated cohorts.
#   - dynamic  = by event time; the standard event-study aggregation.
# Choose by research question, not by convention. See CS-2021 Section 4.2
# and Roth-Sant'Anna-Bilinski-Poe (2023, J. Econometrics) for tradeoffs.

# (a) Simple aggregation — group-size weighted overall ATT.
agg.simple <- aggte(example_attgt, type = "simple")
summary(agg.simple)

# (b) Dynamic aggregation — by event time (length of exposure).
# Use min_e / max_e to restrict the event window for the plot, but note
# that these arguments SILENTLY DROP cohorts that cannot be observed at
# the requested window endpoints. In short pre-periods or staggered
# designs the effective estimand can shift. Verify cohort sizes after
# restriction with summary(aggte(..., min_e = ..., max_e = ...)).
agg.es <- aggte(example_attgt, type = "dynamic")
summary(agg.es)
ggdid(agg.es)

# (c) Group aggregation — average effect within each cohort.
agg.gs <- aggte(example_attgt, type = "group")
summary(agg.gs)
ggdid(agg.gs)

# (d) Calendar-time aggregation — average effect within each calendar period.
agg.ct <- aggte(example_attgt, type = "calendar")
summary(agg.ct)
ggdid(agg.ct)

# -----------------------------------------------------------------------------
# Section 4: Alternative control group — "not yet treated"
# -----------------------------------------------------------------------------
# Useful when (a) there is no never-treated group, or (b) the never-treated
# group is small or selected. The not-yet-treated comparison set expands
# over calendar time as later cohorts have not entered yet.

example_attgt_altcontrol <- att_gt(
  yname         = "Y",
  tname         = "period",
  idname        = "id",
  gname         = "G",
  xformla       = ~ X,
  data          = dta,
  control_group = "notyettreated"
)
summary(example_attgt_altcontrol)

# -----------------------------------------------------------------------------
# Section 5: Real-data example — minimum-wage effect on teen employment
# -----------------------------------------------------------------------------
# `mpdta` is shipped with the package: county-level panel, 2003-2007.
# Columns: year, countyreal (id), lpop (log population),
#          lemp (log employment), first.treat (treatment timing year),
#          treat indicator.

data(mpdta)
head(mpdta)

# (a) Unconditional parallel trends
mw.attgt <- att_gt(
  yname   = "lemp",
  gname   = "first.treat",
  idname  = "countyreal",
  tname   = "year",
  xformla = ~ 1,
  data    = mpdta
)
summary(mw.attgt)
ggdid(mw.attgt, ylim = c(-.3, .3))

# (b) Dynamic effects — event study
mw.dyn <- aggte(mw.attgt, type = "dynamic")
summary(mw.dyn)
ggdid(mw.dyn, ylim = c(-.3, .3))

# (c) Balanced event study — restrict to cohorts with enough exposure
# `balance_e = e` keeps only cohorts observed through at least event-time e
# (i.e., periods 0..e post-treatment), so the event-study composition is
# constant up to event-time e. balance_e = 1 therefore keeps cohorts with
# at least the immediate and one-period-after post-treatment observations
# and drops the latest-treated cohort whose e=1 cannot be observed.
mw.dyn.balance <- aggte(mw.attgt, type = "dynamic", balance_e = 1)
summary(mw.dyn.balance)
ggdid(mw.dyn.balance, ylim = c(-.3, .3))

# (d) Conditional parallel trends — adjust for log county population
mw.attgt.X <- att_gt(
  yname   = "lemp",
  gname   = "first.treat",
  idname  = "countyreal",
  tname   = "year",
  xformla = ~ lpop,
  data    = mpdta
)
summary(mw.attgt.X)

# -----------------------------------------------------------------------------
# Section 6: Pre-testing for parallel trends
# -----------------------------------------------------------------------------
# Two distinct pre-tests are available:
#
# (i) The pre-test reported by summary(att_gt(...)) — uses the
#     simultaneous-confidence-band machinery on the group-time effects
#     during pre-treatment periods. Output: "P-value for pre-test of
#     parallel trends assumption". Higher p-values support PT.
#
# (ii) conditional_did_pretest() — when parallel trends is only plausible
#      conditional on covariates, this is the right diagnostic. Its
#      primary value-add is for CONTINUOUS X: it tests conditional PT
#      via an integrated-moments / Cramer-von-Mises-type statistic on
#      the X distribution (CS-2021 Sec. 5), catching violations that
#      the pooled coefficient pre-test averages over the support of X.
#      The easier categorical case (opposite-sign pre-trends across
#      groups defined by a discrete X) is a special case.

# (ii) Conditional pre-test example
cdp <- conditional_did_pretest(
  yname   = "Y",
  tname   = "period",
  idname  = "id",
  gname   = "G",
  xformla = ~ X,
  data    = dta
)
summary(cdp)

# Reporting note (CS-2021): the package reports SIMULTANEOUS confidence
# bands rather than pointwise intervals. This is the correct multiple-
# testing adjustment for an event-study plot — standard event-study
# regressions typically report pointwise CIs and therefore overstate
# significance once readers scan across event times.

# -----------------------------------------------------------------------------
# Section 6b: Anticipation testing
# -----------------------------------------------------------------------------
# CS-2021 lists "no anticipation" as a co-equal identifying assumption with
# parallel trends. The `anticipation = k` argument to att_gt() allows
# treatment effects to begin up to k periods BEFORE treatment, and returns
# ATT estimates conditional on this anticipation window. Standard practice:
# fit anticipation ∈ {0, 1, 2}, compare aggregated ATTs, report stability.

example_cs0 <- att_gt(yname = "Y", tname = "period", idname = "id", gname = "G",
                      xformla = ~ X, data = dta)                       # baseline
example_cs1 <- att_gt(yname = "Y", tname = "period", idname = "id", gname = "G",
                      xformla = ~ X, data = dta, anticipation = 1)     # 1-period
example_cs2 <- att_gt(yname = "Y", tname = "period", idname = "id", gname = "G",
                      xformla = ~ X, data = dta, anticipation = 2)     # 2-period

# Compare overall ATT across anticipation windows. Stability supports the
# no-anticipation assumption; substantial movement suggests the headline
# estimand is contaminated by anticipated behavior.
sapply(list(example_cs0, example_cs1, example_cs2),
       function(o) aggte(o, type = "simple")$overall.att)

# -----------------------------------------------------------------------------
# Section 7: What to put in the manuscript
# -----------------------------------------------------------------------------
# 1. State that you use CS-2021 to avoid the forbidden-comparisons bias
#    of TWFE under staggered adoption with heterogeneous effects.
# 2. State the comparison-group choice (never-treated vs not-yet-treated)
#    and why it fits your setting.
# 3. State the estimator (`est_method`: dr / ipw / reg) and any covariates
#    in `xformla`. Use "doubly robust" (default `dr`) unless you have a
#    reason to prefer one of the others.
# 4. Report the simple OR group aggregation as the headline ATT, plus the
#    dynamic event-study as the main figure.
# 5. Report the pre-test p-value AND the conditional pretest p-value if
#    you used covariates. Acknowledge the CS-2021 caveat that pre-tests
#    do not validate parallel trends in the post-treatment period.
# 6. Pair with HonestDiD sensitivity to parallel-trends violations
#    (Rambachan & Roth 2023). The full spec — relative-magnitudes M-bar
#    values, smoothness-bound M values, and createSensitivityPlot code —
#    is in SKILL.md PART 4 under "HonestDiD". Do NOT interpret a
#    non-significant pre-trend F as validation: pre-test power is low
#    and pre-test-conditional inference is biased (Roth 2022, AERI 4(3)).
#    The actual robustness statement is the HonestDiD breakdown M-bar.
# 7. Pair with at least one alternative TWFE-free estimator: Sun-Abraham,
#    LP-DiD, Borusyak-Jaravel-Spiess imputation, or de Chaisemartin-
#    D'Haultfoeuille (the last if treatment can switch off). See
#    Strategy 9 Modern DiD Battery in references/strategies.md.
# 8. Report the post-treatment OUTCOME LEVEL for the treated group
#    alongside the ATT (Samii 2025); also report number of treated
#    cohorts and cohort sizes from the panelView audit.
