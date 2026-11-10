# Fig 5 (RQ iii): breeding evidence. Model 1: mean EOD width trend over the season (secondary
# context panel). Model 2: mixture-model-derived proportion of elongated (breeding-condition)
# fish per species/system/date - the primary breeding-evidence test. Shape-atypicality
# circularity check (confidently-classified fish only). Model 3: proportion of candidate-
# juvenile fish (flagged via waveform shape clustering, see
# 06_1/06_2_*_Waveform_Clustering.py) per system/date. Amplitude ruled out as a breeding/
# juvenile metric (heavily confounded by fish position relative to the electrodes, not just
# body size/condition).

library(tidyverse)
library(glmmTMB)
library(DHARMa)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), SPECIES_PALETTE, species_palette(), SPECIES_CODES, save_panel_pdf()

DATA_BUNDLE_FILE <- "01 Data/derived_data_bundle.rds"  # written by 3_Analyze_tracking_results.R
OUTPUT_DIR       <- "03 Analysis Output/figures"

bundle   <- readRDS(DATA_BUNDLE_FILE)
fish_dat <- bundle$fish_dat

# 1. DATA PREP ----------------------------------------------------------------

breeding_dat <- fish_dat |>
  filter(!is.na(mean_width_us), !is.na(T_best)) |>
  mutate(Session_date_num = as.numeric(Session_date - min(Session_date)),  # days since first session
         Species  = factor(Species, levels = SPECIES_CODES),
         Site     = factor(Site),
         log_width = log(mean_width_us),
         # centered so the intercept isn't collinear with T_best*Species (T_best never near 0) -
         # this was the cause of the non-positive-definite Hessian in earlier fit attempts
         T_best_c = T_best - mean(T_best),
         # SLP/PLP/PLS/JR1 were sampled weekly across the season (genuine repeat visits);
         # every LN sub-site and JR2 was visited exactly once (confirmed against
         # Recordings_Sessions_Metadata.csv) - for those, Session_date is perfectly aliased
         # with site identity, so a Session_date_num trend term is not identifiable from a pure
         # site effect. Any model with a Session_date_num (or Session_date_num:System) term
         # must be restricted to Repeat_sampled == TRUE.
         Repeat_sampled = Site %in% c("SLP", "PLP", "PLS", "JR1"))

glimpse(breeding_dat)

# 1B. EXPLORATORY: raw per-site width vs date, with temperature overlay ------------------
# Moved here from 3_Analyze_tracking_results.R (breeding-condition width exploration belongs
# in this script, not the Fig 1 site-overview script) - context/QC only, not part of the
# polished Fig 5 panel set below (which is model-based and raw-data-first per System, not per
# individual site). Kept as its own diagnostic PDFs, not composited into fig5_panels.rds.
build_panel_width_by_site <- function(dat, temp_scale, theme_fn) {
  ggplot(dat, aes(x = visit_midpoint, y = mean_width_us)) +
    geom_point(aes(color = Species), alpha = 0.4) +
    stat_summary(aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
    scale_y_continuous(name = "Mean EOD width (\u00b5s)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (\u00b0C)")) +
    scale_x_datetime(name = "Date") +
    scale_color_manual(values = species_palette(theme_fn)) +
    theme_fn(base_size = 18) +
    theme(axis.title.y.right = element_text(color = "#D55E00"),
          axis.text.y.right  = element_text(color = "#D55E00"))
}

width_by_site_specs <- list(
  JR            = list(temp_scale = 20, data = subset(fish_dat, System == "JR")),
  JR_only_known = list(temp_scale = 20, data = subset(fish_dat, System == "JR" & Species != "Unknown")),
  PLP           = list(temp_scale = 4,  data = subset(fish_dat, Site == "PLP")),
  PL_only_known = list(temp_scale = 4,  data = subset(fish_dat, Site == "PLP" & Species != "Unknown")),
  PLS           = list(temp_scale = 2,  data = subset(fish_dat, Site == "PLS")),
  PLS_only_known = list(temp_scale = 2,  data = subset(fish_dat, Site == "PLS" & Species != "Unknown")),
  PL            = list(temp_scale = 4,  data = subset(fish_dat, System == "PL")),
  PL_only_known = list(temp_scale = 4,  data = subset(fish_dat, System == "PL" & Species != "Unknown")),
  LN_shore      = list(temp_scale = 2,  data = subset(fish_dat, System == "LN" & Direction == "Orthogonal")),
  LN_shore_only_known = list(temp_scale = 2,  data = subset(fish_dat, System == "LN" & Direction == "Orthogonal" & Species != "Unknown")),
  LN_open_water = list(temp_scale = 2,  data = subset(fish_dat, System == "LN" & Orientation == "Vertical")),
  LN_open_water_only_known = list(temp_scale = 2,  data = subset(fish_dat, System == "LN" & Orientation == "Vertical" & Species != "Unknown")),
  LN            = list(temp_scale = 2,  data = subset(fish_dat, System == "LN")),
  LN_only_known = list(temp_scale = 2,  data = subset(fish_dat, System == "LN" & Species != "Unknown")),
  SL            = list(temp_scale = 2,  data = subset(fish_dat, System == "SL")),
  SL_only_known = list(temp_scale = 2,  data = subset(fish_dat, System == "SL" & Species != "Unknown"))
)

for (site_name in names(width_by_site_specs)) {
  spec <- width_by_site_specs[[site_name]]
  panel_dark  <- build_panel_width_by_site(spec$data, spec$temp_scale, theme_black)
  panel_white <- build_panel_width_by_site(spec$data, spec$temp_scale, theme_white)
  save_panel_pdf(panel_dark,  file.path(OUTPUT_DIR, paste0("4_exploratory_width_", site_name, "_dark.pdf")))
  save_panel_pdf(panel_white, file.path(OUTPUT_DIR, paste0("4_exploratory_width_", site_name, "_white.pdf")))
}

# 2. MODEL: does mean EOD width trend upward over the season? -----------------------------
# Modeled on log(width): pulse width/duration is known to vary with temperature via a Q10-type
# (multiplicative, ~1.6) relationship, not an additive one - log-width makes both the
# temperature and date effects proportional on the natural width scale. T_best is included
# (interacted with Species, since thermal sensitivity may differ by species) specifically to
# CONTROL for this confound - without it, any seasonal temperature drift would be conflated
# with the "date" trend. Not every fish is expected to elongate (only breeding-ready
# individuals - the sample also includes juveniles/subadults) - dispformula lets the residual
# SPREAD of width change with date/species too, so a growing subgroup of elongating fish
# diverging from a stable non-breeding majority shows up as increasing variance over time,
# rather than being averaged away into a single population-wide mean shift.
# System dropped as a fixed effect - collinear/redundant with the 22-level Site random
# intercept (same singularity issue found in 3b's habitat model), which is what caused the
# non-positive-definite Hessian / non-convergence in an earlier version of this model.
# Fit restricted to Repeat_sampled sites (SLP/PLP/PLS/JR1) - the Session_date_num trend is not
# identifiable from a pure site effect for the single-visit LN/JR2 sites (see Repeat_sampled
# definition above), so including them would let site-to-site differences masquerade as a
# seasonal trend. The raw scatter in the result plot still shows ALL fish (observed data is
# informative regardless); only the fitted trend line is restricted to the valid subset.
# Site changed from a RANDOM to a FIXED effect for this restricted fit - only 4 site levels
# remain (down from 22), too few to reliably estimate a random-intercept variance alongside
# this many fixed+dispersion terms; this caused the same non-positive-definite Hessian seen
# earlier. With just 4 groups (and JR1/SLP each the only site for their system), a fixed
# 4-level factor is the more stable, appropriate choice.
width_trend_dat <- filter(breeding_dat, Repeat_sampled)
m_width_trend <- glmmTMB(
  log_width ~ Species * Session_date_num + Species * T_best_c + Site,
  dispformula = ~ Species * Session_date_num,
  family = gaussian(),
  data = width_trend_dat
)
summary(m_width_trend)

# Implied Q10 per species from the log(width) ~ T_best slope. Width is expected to SHRINK with
# warming (Q10_width < 1) since EOD rate/frequency increases with temperature (literature
# Q10 ~ 1.4-1.7) and width scales roughly inversely with rate - so Q10_width is converted to a
# rate-equivalent Q10 (1 / Q10_width) for direct comparison against the literature frequency Q10.
width_T_coefs <- fixef(m_width_trend)$cond
base_T_slope  <- width_T_coefs[["T_best_c"]]
species_T_slopes <- sapply(SPECIES_CODES, function(sp) {
  term <- paste0("Species", sp, ":T_best_c")
  base_T_slope + ifelse(term %in% names(width_T_coefs), width_T_coefs[[term]], 0)
})
names(species_T_slopes)[1] <- SPECIES_CODES[1]  # baseline species (GL) has no named interaction term
implied_Q10_width <- exp(species_T_slopes * 10)
implied_Q10_rate_equivalent <- 1 / implied_Q10_width
cat("\n=== Implied Q10 (width) per species - expect < 1 (width shrinks with warming) ===\n")
print(implied_Q10_width)
cat("\n=== Rate-equivalent Q10 (1/Q10_width) - compare against literature ~1.4-1.7 ===\n")
print(implied_Q10_rate_equivalent)

sim_res_width <- simulateResiduals(m_width_trend, n = 1000)
testDispersion(sim_res_width)
testOutliers(sim_res_width)

pdf(file.path(OUTPUT_DIR, "4_width_trend_diagnostics.pdf"), width = 8, height = 6)
plot(sim_res_width)
plotResiduals(sim_res_width, model.frame(m_width_trend)$Species)
dev.off()

# 2B. PROPORTION-ELONGATED MODEL (primary breeding-evidence test) -----------------------
# The population MEAN trend above is diluted by the (likely majority) non-breeding/juvenile
# fish, so it is not actually sensitive to "do breeding-condition fish preferentially occupy
# certain habitats/seasons" - only a population-wide mean shift would show up there. Instead,
# fit a per-species two-component mixture on log_width to find a DATA-DRIVEN cutoff between a
# baseline (non-breeding) and an elongated (breeding-condition) pulse population, then model
# the PROPORTION of elongated fish per session/site - this isolates the breeding subgroup
# rather than averaging it into the majority, and directly tests both whether the proportion
# rises in the rainy season and whether it is concentrated in particular systems/sites
# (candidate hypothesis: JR > SL > PL, well-oxygenated/protected habitats used for breeding).
library(mixtools)

set.seed(1)
elongated_dat <- breeding_dat
elongated_dat$is_elongated <- NA
mixture_summary <- tibble()

for (sp in levels(breeding_dat$Species)) {
  sp_idx    <- breeding_dat$Species == sp
  sp_widths <- breeding_dat$log_width[sp_idx]
  mix <- normalmixEM(sp_widths, k = 2, maxit = 1000, epsilon = 1e-4)
  elongated_component <- which.max(mix$mu)  # higher-mean component = "elongated"
  elongated_dat$is_elongated[sp_idx] <- mix$posterior[, elongated_component] > 0.5
  mixture_summary <- bind_rows(mixture_summary, tibble(
    Species = sp, width_baseline_us = exp(min(mix$mu)), width_elongated_us = exp(max(mix$mu)),
    lambda_elongated = mix$lambda[elongated_component]
  ))
}

cat("\n=== Per-species mixture fit: baseline vs elongated width, and overall elongated fraction ===\n")
print(mixture_summary)

# Restricted to repeat-sampled sites (SLP/PLP/PLS/JR1) for the Session_date_num*System trend -
# LN/JR2 dropped here since their date is aliased with site identity (see Repeat_sampled
# above). System now effectively has 3 levels: JR (=JR1 only), PL, SL.
elongated_summary <- elongated_dat |>
  filter(Repeat_sampled) |>
  group_by(Species, Session_date, Session_date_num, System, Site) |>
  summarise(n_total = n(), n_elongated = sum(is_elongated), .groups = "drop") |>
  mutate(prop_elongated = n_elongated / n_total)

m_elongated_list <- list()
for (sp in levels(breeding_dat$Species)) {
  sp_dat <- filter(elongated_summary, Species == sp)
  # PN: quasi-complete separation on System (PL has 0/10 site-days with any elongated fish) -
  # the full Session_date_num*System model gives NaN SEs and nonsensical coefficients, so PN
  # drops the System interaction and keeps only a usable date-trend estimate.
  sp_formula <- if (sp == "PN") {
    cbind(n_elongated, n_total - n_elongated) ~ Session_date_num + (1 | Site)
  } else {
    cbind(n_elongated, n_total - n_elongated) ~ Session_date_num * System + (1 | Site)
  }
  m <- glmmTMB(sp_formula, family = binomial(), data = sp_dat)
  m_elongated_list[[sp]] <- m
  cat("\n===", sp, "- proportion elongated model ===\n")
  print(summary(m))
}

# 2C. SHAPE ATYPICALITY vs WIDTH/DATE - circularity check (confidently-classified fish only) ---
# The classifier's waveform-length normalization removes ABSOLUTE duration from the shape
# features, so "Unknown" isn't a trivial re-labeling of "wide pulse". But if hormonal breeding
# state alters waveform SHAPE as well as width, the same underlying process could drive both
# the width trend AND a fish being flagged Unknown - meaning the Unknown-fish width trend
# alone isn't independent confirmation of breeding. To test this without relying on the
# discrete Unknown threshold, use lda_dist_centroid (continuous distance from the assigned
# species' reference centroid) WITHIN confidently-classified fish only (Species != "Unknown",
# so membership here does not depend on the width/shape link being tested) - if atypicality
# tracks width/date even among these fish, that is independent evidence for the same
# hormonal-shape-change story, not a re-observation of the classification threshold.
atypicality_dat <- breeding_dat |>
  filter(Species != "Unknown") |>
  mutate(Species = droplevels(Species))

m_atypicality <- glmmTMB(
  lda_dist_centroid ~ Species * Session_date_num + Species * T_best_c + (1 | Site),
  family = gaussian(),
  data = atypicality_dat
)
summary(m_atypicality)

sim_res_atyp <- simulateResiduals(m_atypicality, n = 1000)
testDispersion(sim_res_atyp)

cat("\n=== Correlation of lda_dist_centroid with log_width, per species (confidently-classified only) ===\n")
atypicality_dat |>
  group_by(Species) |>
  summarise(cor_dist_width = cor(lda_dist_centroid, log_width, use = "complete.obs"), n = n()) |>
  print()

# 2D. CANDIDATE-JUVENILE SITE/TEMPORAL DISTRIBUTION (RQ iii recruitment/site-specificity) ---
# Candidate juveniles identified via unsupervised waveform-shape clustering, refined per
# 06_2_Species_Grouped_Waveform_Clustering.py: species-split GMM (not DBSCAN - DBSCAN is
# density-based and dumped the diffuse/plastic juvenile population into noise regardless of
# eps tuning; a GMM component can be broad and still register as a genuine cluster) fit
# separately within "Unknown" fish. Components 1, 7, and 9 (of the BIC-selected component
# count) look juvenile-like on visual inspection of waveform_clusters_Unknown.png. NOTE: width
# was considered as an alternative/simpler juvenile discriminator but ruled out - juvenile
# pulses tend to be LONGER (not narrower) than adult EODs, and this may vary by species, so a
# width-based split would likely mislabel or miss juveniles; shape-clustering remains the
# primary tool.
juvenile_clusters <- read.csv(file.path(OUTPUT_DIR, "waveform_species_grouped_cluster_assignment.csv")) |>
  filter(Species == "Unknown") |>
  mutate(candidate_juvenile = cluster_within_species %in% c(1, 7, 9)) |>
  select(fish_key, candidate_juvenile)

# left_join (not inner_join) - only Unknown fish can be flagged candidate_juvenile, but
# n_total below still needs ALL species' fish per session/site as the denominator, not just
# Unknown, to preserve "proportion of ALL detected fish showing juvenile signatures".
juvenile_dat <- breeding_dat |>
  left_join(juvenile_clusters, by = "fish_key") |>
  mutate(candidate_juvenile = replace_na(candidate_juvenile, FALSE))

cat("\n=== Candidate-juvenile fraction by Species ===\n")
juvenile_dat |>
  group_by(Species) |>
  summarise(n = n(), pct_juvenile = 100 * mean(candidate_juvenile)) |>
  print()

# Restricted to repeat-sampled sites (SLP/PLP/PLS/JR1) for the Session_date_num*System trend -
# see Repeat_sampled definition in Part 1 above.
juvenile_summary <- juvenile_dat |>
  filter(Repeat_sampled) |>
  group_by(Session_date, Session_date_num, System, Site) |>
  summarise(n_total = n(), n_juvenile = sum(candidate_juvenile), .groups = "drop") |>
  mutate(prop_juvenile = n_juvenile / n_total)

m_juvenile <- glmmTMB(
  cbind(n_juvenile, n_total - n_juvenile) ~ Session_date_num * System + (1 | Site),
  family = betabinomial(), data = juvenile_summary
)
summary(m_juvenile)

sim_res_juv <- simulateResiduals(m_juvenile, n = 1000)
testDispersion(sim_res_juv)
testOutliers(sim_res_juv)

# 2E. EXPORT MODEL RESULTS ----------------------------------------------------
library(broom.mixed)

coef_tables <- c(
  list(m_width_trend = m_width_trend),
  setNames(m_elongated_list, paste0("m_elongated_", names(m_elongated_list))),
  list(m_atypicality = m_atypicality, m_juvenile = m_juvenile)
)
model_coefficients <- imap(coef_tables, \(m, name) tidy(m, effects = "fixed", conf.int = TRUE) |> mutate(model = name, .before = 1)) |>
  bind_rows()
write.csv(model_coefficients, file.path(OUTPUT_DIR, "4_model_coefficients.csv"), row.names = FALSE)

# 3. RESULT PLOT (Fig 5 panel A) --------------------------------------------------------
# PRIMARY: raw mean_width_us per fish over the season, faceted by System - the actual
# observed data, for ALL fish/systems (including LN, even though no trend is fit there).
# Model trend line (from m_width_trend, T_best held at its overall mean so the line isolates
# the date trend net of temperature) drawn on top as the supplementary layer - predicted per
# Site (now a fixed effect) then averaged across sites within System for a single display
# line per system facet; only JR/PL/SL get a line since the model was fit on Repeat_sampled
# data only (no valid Site level exists for LN).
newdata_width <- width_trend_dat |>
  distinct(System, Site, Species) |>
  cross_join(tibble(Session_date_num = seq(min(width_trend_dat$Session_date_num),
                                            max(width_trend_dat$Session_date_num), length.out = 50)))
newdata_width$T_best_c <- 0  # holds T_best at its overall mean (T_best_c is already mean-centered)
pred_width <- predict(m_width_trend, newdata = newdata_width, se.fit = TRUE)
newdata_width$fit_site <- pred_width$fit
newdata_width$se_site  <- pred_width$se.fit

newdata_width <- newdata_width |>
  group_by(System, Species, Session_date_num) |>
  summarise(fit = mean(fit_site), se = mean(se_site), .groups = "drop") |>
  mutate(upper = exp(fit + se),
         lower = exp(fit - se),
         fit   = exp(fit))

build_panel_width_trend <- function(theme_fn) {
  ggplot(breeding_dat, aes(x = Session_date_num, y = mean_width_us, color = Species)) +
    geom_point(alpha = 0.15, size = 0.8) +
    geom_ribbon(data = newdata_width, aes(y = fit, ymin = lower, ymax = upper, fill = Species),
                color = NA, alpha = 0.25) +
    geom_line(data = newdata_width, aes(y = fit), linewidth = 0.9) +
    scale_color_manual(values = species_palette(theme_fn)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    facet_wrap(~ System, scales = "free_y") +
    labs(x = "Days since first session",
         y = "Mean EOD width (\u00b5s) - points = observed, line = fitted trend (at mean T_best)") +
    theme_fn(base_size = 16)
}

fig5_panel_width_trend_dark  <- build_panel_width_trend(theme_black)
fig5_panel_width_trend_white <- build_panel_width_trend(theme_white)

save_panel_pdf(fig5_panel_width_trend_dark,  file.path(OUTPUT_DIR, "fig5_panel_width_trend_dark.pdf"))
save_panel_pdf(fig5_panel_width_trend_white, file.path(OUTPUT_DIR, "fig5_panel_width_trend_white.pdf"))

# 4. RESULT PLOT (Fig 5 panel B) --------------------------------------------------------
# PRIMARY: raw observed proportion of elongated (breeding-condition) fish per session/site,
# faceted by Species, colored by System - the actual test of the habitat/season hypothesis.
# Model trend line (from the per-species binomial GLMM, Site random effect averaged out)
# overlaid per System as the supplementary layer.
newdata_elongated <- elongated_summary |>
  distinct(Species, System) |>
  cross_join(tibble(Session_date_num = seq(min(elongated_summary$Session_date_num),
                                            max(elongated_summary$Session_date_num), length.out = 50)))

newdata_elongated$fit <- NA_real_
for (sp in levels(breeding_dat$Species)) {
  sp_rows <- newdata_elongated$Species == sp
  newdata_elongated$fit[sp_rows] <- predict(m_elongated_list[[sp]], newdata = newdata_elongated[sp_rows, ],
                                             type = "response", re.form = NA)
}

build_panel_proportion_elongated <- function(theme_fn) {
  ggplot(elongated_summary, aes(x = Session_date_num, y = prop_elongated, color = System)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_line(data = newdata_elongated, aes(y = fit), linewidth = 0.9) +
    facet_wrap(~ Species) +
    labs(x = "Days since first session",
         y = "Proportion of fish in elongated (breeding-condition) state per session/site") +
    theme_fn(base_size = 16)
}

fig5_panel_proportion_elongated_dark  <- build_panel_proportion_elongated(theme_black)
fig5_panel_proportion_elongated_white <- build_panel_proportion_elongated(theme_white)

save_panel_pdf(fig5_panel_proportion_elongated_dark,  file.path(OUTPUT_DIR, "fig5_panel_proportion_elongated_dark.pdf"))
save_panel_pdf(fig5_panel_proportion_elongated_white, file.path(OUTPUT_DIR, "fig5_panel_proportion_elongated_white.pdf"))

# 5. RESULT PLOT (Fig 5 panel C) --------------------------------------------------------
# PRIMARY: raw observed proportion of candidate-juvenile fish per session/site, colored by
# System - tests whether juvenile appearance (recruitment) is seasonally/spatially patterned.
# Model trend line (from m_juvenile, Site random effect averaged out) as the secondary layer.
newdata_juvenile <- juvenile_summary |>
  distinct(System) |>
  cross_join(tibble(Session_date_num = seq(min(juvenile_summary$Session_date_num),
                                            max(juvenile_summary$Session_date_num), length.out = 50)))
newdata_juvenile$fit <- predict(m_juvenile, newdata = newdata_juvenile, type = "response", re.form = NA)

build_panel_juvenile <- function(theme_fn) {
  ggplot(juvenile_summary, aes(x = Session_date_num, y = prop_juvenile, color = System)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_line(data = newdata_juvenile, aes(y = fit), linewidth = 0.9) +
    labs(x = "Days since first session",
         y = "Proportion of candidate-juvenile fish per session/site") +
    theme_fn(base_size = 16)
}

fig5_panel_juvenile_dark  <- build_panel_juvenile(theme_black)
fig5_panel_juvenile_white <- build_panel_juvenile(theme_white)

save_panel_pdf(fig5_panel_juvenile_dark,  file.path(OUTPUT_DIR, "fig5_panel_juvenile_dark.pdf"))
save_panel_pdf(fig5_panel_juvenile_white, file.path(OUTPUT_DIR, "fig5_panel_juvenile_white.pdf"))

saveRDS(
  list(panel_width_trend_dark = fig5_panel_width_trend_dark,
       panel_width_trend_white = fig5_panel_width_trend_white,
       panel_proportion_elongated_dark = fig5_panel_proportion_elongated_dark,
       panel_proportion_elongated_white = fig5_panel_proportion_elongated_white,
       panel_juvenile_dark = fig5_panel_juvenile_dark,
       panel_juvenile_white = fig5_panel_juvenile_white),
  file.path(OUTPUT_DIR, "fig5_panels.rds")
)

