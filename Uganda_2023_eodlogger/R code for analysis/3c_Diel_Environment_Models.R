# Fig 1 (overlay, RQ ii): diel fish activity ~ time-of-day + environment (temperature/light/DO).
# GAMM per DielGroup, with a concurvity check and raw-data-first result plots.

library(tidyverse)
library(mgcv)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), SPECIES_PALETTE, save_panel_pdf()

DATA_BUNDLE_FILE <- "01 Data/derived_data_bundle.rds"  # written by 3_Analyze_tracking_results.R
OUTPUT_DIR       <- "03 Analysis Output/figures"

bundle       <- readRDS(DATA_BUNDLE_FILE)
fish_raw     <- bundle$fish_raw
register_raw <- bundle$register_raw
light_temp   <- bundle$light_temp
DO_combined  <- bundle$DO_combined

# 1. DATA PREP ----------------------------------------------------------------

# Repeat-sampled systems (PL, SL, JR) contribute repeated sessions at the same site. LN sites were
# each sampled only once, but a single LN session still spans a near-full day-night cycle
# (~21-34h) - so LN CAN inform a diel pattern, as long as sessions are pooled by a shared
# grouping (Orientation/vegetation cover) rather than by individual site identity, which would be
# aliased with session (same issue as the LN site-level analysis in 3b). LNI (floating island)
# sessions excluded here too - treated separately in 7_LNI_Floating_Island_Distribution.R.
TS_BIN_MIN <- 10  # coarser than script 3's 1-min descriptive bins - reduces zero-inflation for GAMM fitting

fish_diel <- fish_raw |>
  filter(!grepl("^LNI", Site)) |>
  mutate(DielGroup = case_when(
    System %in% c("PL", "SL", "JR") ~ "Repeat_shore",
    System == "LN" & Orientation == "Vertical"   ~ "LN_open_water",
    System == "LN" & Direction   == "Orthogonal" ~ paste0("LN_shore_", Veg_cover_extent),
    TRUE ~ NA_character_
  ))

# Time skeleton built from the SESSION'S actual recording window (register_raw), not from where
# fish happened to be detected - anchoring to observed detections (as an earlier version of this
# script did) truncates or entirely drops sessions/species with sparse or zero activity, instead
# of correctly representing them as long runs of 0. This matters a lot here: low/zero activity at
# open-water sites is exactly the pattern under test, and must not be excluded from the data.
session_windows <- register_raw |>
  filter(!grepl("^LNI", Site), !is.na(t_first_rec_start), !is.na(t_last_rec_end)) |>
  distinct(System, Site, Session_date, Logger_ID, t_first_rec_start, t_last_rec_end,
           Sunrise, Sunset, Orientation, Direction, Veg_cover_extent, Moon_illuminated_fraction) |>
  mutate(DielGroup = case_when(
    System %in% c("PL", "SL", "JR") ~ "Repeat_shore",
    System == "LN" & Orientation == "Vertical"   ~ "LN_open_water",
    System == "LN" & Direction   == "Orthogonal" ~ paste0("LN_shore_", Veg_cover_extent),
    TRUE ~ NA_character_
  )) |>
  filter(!is.na(DielGroup))

activity_skeleton <- session_windows |>
  rowwise() |>
  mutate(datetime = list(seq(floor_date(t_first_rec_start, unit = paste(TS_BIN_MIN, "minutes")),
                              t_last_rec_end, by = paste(TS_BIN_MIN, "min")))) |>
  ungroup() |>
  select(System, Site, Session_date, Logger_ID, DielGroup, Sunrise, Sunset,
         Moon_illuminated_fraction, datetime) |>
  unnest(datetime) |>
  cross_join(tibble(Species = SPECIES_CODES))

# expand each tracked fish's entry/exit window into TS_BIN_MIN-minute presence bins (same
# approach as script 3's ts_expanded, kept at species level rather than collapsed to sums), then
# count per bin - this only supplies the ACTUAL counts, the skeleton above supplies the full range
ts_expanded_diel <- fish_diel |>
  rowwise() |>
  mutate(datetime = list(seq(floor_date(entry_time, unit = paste(TS_BIN_MIN, "minutes")),
                              exit_time, by = paste(TS_BIN_MIN, "min")))) |>
  ungroup() |>
  select(Species, System, Site, Session_date, Logger_ID, datetime) |>
  unnest(datetime)

fish_counts <- ts_expanded_diel |>
  count(System, Site, Session_date, Logger_ID, datetime, Species, name = "n_fish")

activity_ts <- activity_skeleton |>
  left_join(fish_counts, by = c("System", "Site", "Session_date", "Logger_ID", "datetime", "Species")) |>
  mutate(n_fish = replace_na(n_fish, 0),
         std_time_ss = as.numeric(difftime(datetime, Sunset, units = "hours")))

# Environmental time series matching: round continuous logger timestamps to the same
# TS_BIN_MIN-minute bin and average, then join on (System, Site, datetime_bin) - a coarse-bin
# join is simpler and more robust here than a true nearest-neighbour time join, and matches the
# resolution already used for the fish activity bins above.
light_temp_binned <- light_temp |>
  mutate(datetime = floor_date(Datetime, unit = paste(TS_BIN_MIN, "minutes"))) |>
  group_by(System, Site, datetime) |>
  summarise(T_C = mean(T_C, na.rm = TRUE), Lux = mean(Lux, na.rm = TRUE), .groups = "drop")

DO_binned <- DO_combined |>
  mutate(datetime = floor_date(Datetime, unit = paste(TS_BIN_MIN, "minutes"))) |>
  group_by(System, Site, datetime) |>
  summarise(DO_AS = mean(DO_AS, na.rm = TRUE), .groups = "drop")

activity_dat <- activity_ts |>
  left_join(light_temp_binned, by = c("System", "Site", "datetime")) |>
  left_join(DO_binned, by = c("System", "Site", "datetime"))

# sanity check: how many sessions/bins fall into each diel group, and is any group too sparse?
count(activity_dat, DielGroup, Site)

glimpse(activity_dat)

# 2. MODEL FITTING --------------------------------------------------------------
# One GAMM per DielGroup (5 groups) rather than per individual site - Site enters as a random
# effect (absorbs site-to-site baseline differences) and Species as a by-variable smooth (each
# species gets its own diel curve within a group). n_fish is heavily zero-dominated -> negative
# binomial family. std_time_ss spans >24h for the longest sessions, so folded to std_time_mod
# (0-24h) for a genuine cyclic time-of-day smooth (bs="cc", wrapping at 0/24h) rather than
# conflating diel pattern with multi-day drift.
# T_C/Lux/Moon_illuminated_fraction go in the main model (good coverage, one Moon value per
# session); DO_AS (55% missing) is fit as a separate sub-model restricted to sessions with
# actual logger DO coverage, rather than forced into the main model and losing over half the
# data to listwise deletion.
activity_dat <- activity_dat |>
  mutate(Species = factor(Species, levels = SPECIES_CODES),
         Site    = factor(Site),
         std_time_mod = std_time_ss %% 24,
         std_time_sr = as.numeric(difftime(datetime, Sunrise, units = "hours")),
         Photoperiod = ifelse(std_time_ss >= 0 & std_time_sr <= 0, "Night", "Day"))

# Is Lux at night actually tracking moonlight, or just residual sensor noise/artificial light?
# (Lux's instability in LN_shore_None's model could reflect it partly proxying moonlight rather
# than canopy cover, which the user wants to keep Lux for.)
night_lux_moon <- activity_dat |> filter(Photoperiod == "Night", !is.na(Lux))
cat("\n=== Lux vs. Moon_illuminated_fraction at night (all DielGroups pooled) ===\n")
print(cor.test(night_lux_moon$Lux, night_lux_moon$Moon_illuminated_fraction, method = "spearman"))

DIEL_GROUPS <- unique(activity_dat$DielGroup)

# Descriptive summary per DielGroup - also used to decide which groups have too few actual
# detections to support a meaningful GAMM (a degenerate fit with huge SEs/edf~0 doesn't mean
# "no diel pattern", it means "not enough data to say anything" - e.g. LN_open_water has only
# 9 total fish-bin detections across all species, confirmed via manual check).
diel_group_summary <- activity_dat |>
  summarise(n_rows = n(), total_fish = sum(n_fish), nonzero_bins = sum(n_fish > 0),
            n_sessions = n_distinct(Site, Session_date, Logger_ID), .by = DielGroup)
print(diel_group_summary)

MIN_TOTAL_FISH_FOR_GAM <- 30  # below this, treat the group as descriptive-only, not GAMM-able

# Site random effect only makes sense with >1 site per group (LN_shore_Partial = LN17 only)
fit_diel_gam <- function(dat, env_terms) {
  site_term <- if (n_distinct(dat$Site) > 1) "+ s(Site, bs = 're')" else ""
  form <- as.formula(paste("n_fish ~ s(std_time_mod, bs = 'cc', by = Species, k = 10) + Species +",
                            paste(env_terms, collapse = " + "), site_term))
  gam(form, family = nb(), knots = list(std_time_mod = c(0, 24)), data = dat, method = "REML")
}

diel_models    <- list()  # main models: T_C + Lux
diel_models_DO <- list()  # DO sub-models, DO-covered sessions only
diel_data_main <- list()  # main models' fitting data, kept for prediction (covariate means)

for (g in DIEL_GROUPS) {
  dat_main <- activity_dat |> filter(DielGroup == g, !is.na(T_C), !is.na(Lux))
  if (sum(dat_main$n_fish) < MIN_TOTAL_FISH_FOR_GAM) {
    message("Skipping main model for ", g, " - only ", sum(dat_main$n_fish),
            " total detections, too sparse for a meaningful GAMM (report descriptively instead)")
    next
  }
  # Moon_illuminated_fraction only identifiable in Repeat_shore, which has enough independent
  # sessions per site to separate moon phase from site identity (52 sessions). The LN groups
  # have only ~5 sessions, one per site/date, so Moon there is aliased with Site (confirmed:
  # adding it collapsed s(Site) to ~0 edf while Moon absorbed that variance) - excluded there.
  main_env_terms <- if (g == "Repeat_shore") c("T_C", "Lux", "Moon_illuminated_fraction") else c("T_C", "Lux")
  diel_models[[g]] <- tryCatch(
    fit_diel_gam(dat_main, main_env_terms),
    error = function(e) { message("Main model failed for ", g, ": ", conditionMessage(e)); NULL }
  )
  if (!is.null(diel_models[[g]])) diel_data_main[[g]] <- dat_main

  dat_do <- activity_dat |> filter(DielGroup == g, !is.na(DO_AS))
  if (nrow(dat_do) >= 50 && sum(dat_do$n_fish) >= MIN_TOTAL_FISH_FOR_GAM) {
    diel_models_DO[[g]] <- tryCatch(
      fit_diel_gam(dat_do, "DO_AS"),
      error = function(e) { message("DO model failed for ", g, ": ", conditionMessage(e)); NULL }
    )
  } else {
    message("Skipping DO model for ", g, " - insufficient DO coverage/detections (", nrow(dat_do),
            " rows, ", sum(dat_do$n_fish), " fish)")
  }
}

for (g in names(diel_models)) {
  if (!is.null(diel_models[[g]])) {
    env_label <- if (g == "Repeat_shore") "T_C + Lux + Moon_illuminated_fraction" else "T_C + Lux"
    cat("\n=== Main model (", env_label, "):", g, "===\n")
    print(summary(diel_models[[g]]))
  }
}
for (g in names(diel_models_DO)) {
  if (!is.null(diel_models_DO[[g]])) {
    cat("\n=== DO sub-model:", g, "===\n")
    print(summary(diel_models_DO[[g]]))
  }
}
summary(activity_dat)

# 3. CONCURVITY CHECK (Lux vs. time-of-day smooth) -----------------------------
# LN_shore_None's Lux coefficient had a huge, unstable SE (-36.07 +/- 428.8) - concurvity()
# quantifies whether that's because Lux is largely redundant with the cyclic time-of-day smooth
# (both track day/night) rather than carrying independent information. "worst" is the most
# conservative measure (upper bound on how much of one term's effect could be reconstructed
# from the others); >0.8 is generally considered concerning.
for (g in names(diel_models)) {
  if (!is.null(diel_models[[g]])) {
    cat("\n=== Concurvity:", g, "===\n")
    print(concurvity(diel_models[[g]], full = FALSE)$worst)
  }
}

# 4. EXPORT MODEL RESULTS ----------------------------------------------------
library(broom)

tidy_gam_terms <- function(m) {
  bind_rows(
    tidy(m, parametric = TRUE)  |> mutate(term_type = "parametric", .before = 1),
    tidy(m, parametric = FALSE) |> mutate(term_type = "smooth",     .before = 1)
  )
}

model_coefficients <- bind_rows(
  imap(diel_models,    \(m, name) if (!is.null(m)) tidy_gam_terms(m) |> mutate(model = paste0(name, "_main"), .before = 1)),
  imap(diel_models_DO, \(m, name) if (!is.null(m)) tidy_gam_terms(m) |> mutate(model = paste0(name, "_DO"),   .before = 1))
)
write.csv(model_coefficients, file.path(OUTPUT_DIR, "3c_model_coefficients.csv"), row.names = FALSE)

# 5. RESULT PLOTS (Fig 1 panel) --------------------------------------------
# Predicted diel activity curve per species, faceted by System (JR/PL/SL - the actual
# repeated-measurement units within the pooled Repeat_shore model) plus LN_shore_Complete/
# LN_shore_None as their own facets (LN has no repeated measurements, so it isn't broken out
# by System). For each facet, the curve is the average of that facet's own sites' predictions
# (s(Site) included, not excluded, so real site-to-site baseline differences are reflected) -
# a genuine composite prediction: T_C/Lux are NOT held at a single fixed mean but instead follow
# their own average hour-of-day trajectory (from the data), so the curve reflects both the
# fitted time-of-day smooth AND the indirect contribution of covariates that themselves cycle
# with time of day. Moon_illuminated_fraction (Repeat_shore only) has no intra-day cycle, so it
# stays at that facet's overall mean. std_time_mod = 0 is sunset; ~12 is the approximate
# equatorial sunrise (visual reference line only).
TIME_BIN_H <- 1  # hour-of-day bin for estimating each covariate's own average diel trajectory

diel_pred_all <- NULL
diel_raw_all  <- NULL  # actual observed mean counts per hour-of-day bin, shown as a background
                        # reference layer under the fitted curve (per user preference: two rainy-
                        # season months should be shown as "what we found" first, with the GAMM's
                        # smoothed pattern as the supplementary "what this suggests generally")
for (g in names(diel_models)) {
  dat_g <- diel_data_main[[g]]
  dat_g$time_bin <- floor(dat_g$std_time_mod / TIME_BIN_H) * TIME_BIN_H

  facet_keys <- if (g == "Repeat_shore") unique(dat_g$System) else g

  for (facet_key in facet_keys) {
    dat_f <- if (g == "Repeat_shore") dat_g |> filter(System == facet_key) else dat_g

    raw_means <- dat_f |>
      group_by(time_bin, Species) |>
      summarise(raw_mean = mean(n_fish, na.rm = TRUE), .groups = "drop") |>
      mutate(Facet = as.character(facet_key))
    diel_raw_all <- bind_rows(diel_raw_all, raw_means)

    env_means <- dat_f |>
      group_by(time_bin) |>
      summarise(T_C = mean(T_C, na.rm = TRUE), Lux = mean(Lux, na.rm = TRUE), .groups = "drop")

    pred_grid <- expand.grid(
      std_time_mod = seq(0, 24, by = 0.25),
      Species = factor(SPECIES_CODES, levels = SPECIES_CODES)
    )
    pred_grid$time_bin <- floor(pred_grid$std_time_mod / TIME_BIN_H) * TIME_BIN_H
    pred_grid <- pred_grid |> left_join(env_means, by = "time_bin")
    # fall back to this facet's overall mean for any hour bin with no coverage
    pred_grid$T_C <- ifelse(is.na(pred_grid$T_C), mean(dat_f$T_C, na.rm = TRUE), pred_grid$T_C)
    pred_grid$Lux <- ifelse(is.na(pred_grid$Lux), mean(dat_f$Lux, na.rm = TRUE), pred_grid$Lux)
    if (g == "Repeat_shore") {
      pred_grid$Moon_illuminated_fraction <- mean(dat_f$Moon_illuminated_fraction, na.rm = TRUE)
    }

    # average the site-level prediction (s(Site) included) across this facet's actual sites
    facet_sites <- unique(dat_f$Site)
    site_fits <- NULL
    site_ses  <- NULL
    for (s in facet_sites) {
      pg <- pred_grid
      pg$Site <- s
      pv <- predict(diel_models[[g]], newdata = pg, type = "response", se.fit = TRUE)
      site_fits <- cbind(site_fits, pv$fit)
      site_ses  <- cbind(site_ses, pv$se.fit)
    }
    pred_grid$fit <- rowMeans(site_fits)
    pred_grid$se  <- sqrt(rowMeans(site_ses^2))  # approx combined SE across averaged sites
    pred_grid$Facet <- as.character(facet_key)
    diel_pred_all <- bind_rows(diel_pred_all,
                                pred_grid[, c("std_time_mod", "Species", "fit", "se", "Facet")])
  }
}
diel_pred_all$Facet <- factor(diel_pred_all$Facet,
                              levels = c("JR", "PL", "SL", "LN_shore_Complete", "LN_shore_None"))
diel_raw_all$Facet <- factor(diel_raw_all$Facet,
                             levels = c("JR", "PL", "SL", "LN_shore_Complete", "LN_shore_None"))

# PRIMARY layer = actual observed mean counts per hour-of-day bin (points, per species); the
# GAMM's fitted curve is drawn on top as the supplementary "what this suggests generally" layer,
# not the sole/primary representation of the data (two rainy-season months describe what was
# found, not a general population model).
build_panel_diel <- function(theme_fn, ref_line_color) {
  ggplot(diel_pred_all, aes(x = std_time_mod, y = fit, color = Species, fill = Species)) +
    geom_vline(xintercept = 0,  linewidth = 0.5, colour = ref_line_color, lty = 2) +
    geom_vline(xintercept = 12, linewidth = 0.5, colour = ref_line_color, lty = 2) +
    geom_point(data = diel_raw_all, aes(x = time_bin + TIME_BIN_H / 2, y = raw_mean, color = Species),
               inherit.aes = FALSE, alpha = 0.35, size = 1.2) +
    geom_ribbon(aes(ymin = pmax(fit - se, 0), ymax = fit + se), color = NA, alpha = 0.2) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = species_palette(theme_fn)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    facet_wrap(~ Facet, scales = "free_y") +
    labs(x = "Time since sunset (h, mod 24)", y = "Fish count per bin (points = observed, line = fitted)") +
    theme_fn(base_size = 16)
}

fig1_panel_diel_dark  <- build_panel_diel(theme_black, "white")
fig1_panel_diel_white <- build_panel_diel(theme_white, "black")

fig1_panel_diel_white

save_panel_pdf(fig1_panel_diel_dark,  file.path(OUTPUT_DIR, "fig1_panel_diel_dark.pdf"))
save_panel_pdf(fig1_panel_diel_white, file.path(OUTPUT_DIR, "fig1_panel_diel_white.pdf"))

saveRDS(
  list(panel_diel_dark = fig1_panel_diel_dark, panel_diel_white = fig1_panel_diel_white),
  file.path(OUTPUT_DIR, "fig1_panels.rds")
)
