# Fig 3 (RQ i refinement): paired ground/surface (PLP/SLP) and orthogonal/parallel (JR)
# distribution comparisons - restricted to genuinely PAIRED simultaneous deployments only
# (PLP, SLP, JR1's L5/L6 pair). LN is NOT a paired design (every site sampled once) and is
# already covered by 3b's Orientation model - excluded here. JR2 (single one-off logger, no
# pair) is excluded too.

library(tidyverse)
library(glmmTMB)
library(DHARMa)
library(performance)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), SPECIES_PALETTE, species_palette(), SPECIES_CODES, save_panel_pdf()

DATA_BUNDLE_FILE <- "01 Data/derived_data_bundle.rds"  # written by 3_Analyze_tracking_results.R
OUTPUT_DIR       <- "03 Analysis Output/figures"

bundle       <- readRDS(DATA_BUNDLE_FILE)
fish_dat     <- bundle$fish_dat
register_raw <- bundle$register_raw

# 1. DATA PREP ----------------------------------------------------------------

# one row per session, with the habitat descriptors constant within a session
session_info <- fish_dat |>
  distinct(System, Site, Session_date, Logger_ID, Vertical_location, Direction, Orientation,
           T_best, DO_best, pH_mean, Cond_mean)

# species x Photoperiod counts per session (Unknown kept as a genuine 5th category)
session_species_counts <- fish_dat |>
  count(System, Site, Session_date, Logger_ID, Species, Photoperiod, name = "n_fish")

# Day/Night each need their OWN offset (hours of daylight vs darkness actually covered by that
# session), not the full Recorded_dur_h - derived from Sunrise/Sunset (already the sole
# sunset->next-sunrise pair within each session's standardized window, see script 2).
session_dur <- register_raw |>
  distinct(System, Site, Session_date, Logger_ID, Recorded_dur_h, Sunrise, Sunset) |>
  mutate(Night = as.numeric(difftime(Sunrise, Sunset, units = "hours")),
         Day   = Recorded_dur_h - Night) |>
  select(System, Site, Session_date, Logger_ID, Day, Night) |>
  pivot_longer(cols = c(Day, Night), names_to = "Photoperiod", values_to = "dur_h")

# cross_join with all 5 species x Day/Night so absent combinations get an explicit 0 count
paired_dat <- session_info |>
  cross_join(tibble(Species = SPECIES_CODES)) |>
  cross_join(tibble(Photoperiod = c("Day", "Night"))) |>
  left_join(session_species_counts,
            by = c("System", "Site", "Session_date", "Logger_ID", "Species", "Photoperiod")) |>
  mutate(n_fish = replace_na(n_fish, 0)) |>
  left_join(session_dur, by = c("System", "Site", "Session_date", "Logger_ID", "Photoperiod")) |>
  mutate(Species     = factor(Species, levels = SPECIES_CODES),
         Photoperiod = factor(Photoperiod, levels = c("Day", "Night")),
         Session_ID  = paste(System, Site, Session_date, Logger_ID, sep = "_"))

glimpse(paired_dat)

# 2. GROUND vs SURFACE (PLP: L3 ground/L4 surface; SLP: L2 ground/L1 surface) ------------------
# Same Session_date + Site = one simultaneous ground+surface deployment pair -> Session_date
# (nested within Site) is the natural pairing unit, used as a random intercept. Photoperiod
# entered as a category (Day/Night) rather than continuous time-of-day: the hypothesis here is
# a simple day/night contrast in vertical position, not a fine within-day gradient (that finer
# question is what 3c's diel GAMMs are for) - a factor is simpler, directly interpretable, and
# consistent with how Photoperiod is used everywhere else in this codebase. Two-way interactions
# only (no Species:Vertical_location:Photoperiod 3-way) to keep the model identifiable given the
# available sample size (8 paired sessions).
dat_ground_surface <- paired_dat |> filter(Site %in% c("PLP", "SLP"))

m_ground_surface <- glmmTMB(
  n_fish ~ Species * Vertical_location + Species * Photoperiod + Vertical_location * Photoperiod +
    Site + offset(log(dur_h)) + (1 | Session_date),
  family = nbinom2,
  data = dat_ground_surface
)
summary(m_ground_surface)

check_singularity(m_ground_surface)
sim_res_gs <- simulateResiduals(m_ground_surface, n = 1000)
testDispersion(sim_res_gs)
testOutliers(sim_res_gs)

pdf(file.path(OUTPUT_DIR, "3d_ground_surface_diagnostics.pdf"), width = 8, height = 6)
plot(sim_res_gs)
plotResiduals(sim_res_gs, model.frame(m_ground_surface)$Vertical_location)
plotResiduals(sim_res_gs, model.frame(m_ground_surface)$Photoperiod)
dev.off()

# 3. JR1: ORTHOGONAL (L5) vs PARALLEL (L6) --------------------------------------------------
# JR2 excluded - single one-off logger, not part of the paired L5/L6 deployment.
dat_jr <- paired_dat |> filter(Site == "JR1")

m_jr_direction <- glmmTMB(
  n_fish ~ Species * Direction + Species * Photoperiod + Direction * Photoperiod +
    offset(log(dur_h)) + (1 | Session_date),
  family = nbinom2,
  data = dat_jr
)
summary(m_jr_direction)

check_singularity(m_jr_direction)
sim_res_jr <- simulateResiduals(m_jr_direction, n = 1000)
testDispersion(sim_res_jr)
testOutliers(sim_res_jr)

pdf(file.path(OUTPUT_DIR, "3d_jr_direction_diagnostics.pdf"), width = 8, height = 6)
plot(sim_res_jr)
plotResiduals(sim_res_jr, model.frame(m_jr_direction)$Direction)
plotResiduals(sim_res_jr, model.frame(m_jr_direction)$Photoperiod)
dev.off()

# 4. EXPORT MODEL RESULTS ----------------------------------------------------
library(broom.mixed)

model_coefficients <- bind_rows(
  tidy(m_ground_surface, effects = "fixed", conf.int = TRUE) |> mutate(model = "m_ground_surface", .before = 1),
  tidy(m_jr_direction,   effects = "fixed", conf.int = TRUE) |> mutate(model = "m_jr_direction",    .before = 1)
)
write.csv(model_coefficients, file.path(OUTPUT_DIR, "3d_model_coefficients.csv"), row.names = FALSE)

# 5. RESULT PLOTS (Fig 3 panels) --------------------------------------------------------

# Panel A: predicted visits per 12h (reference duration) by Vertical_location x Species,
# faceted by Photoperiod - directly visualizes the "surface use increases at night" hypothesis
newdata_gs <- expand.grid(
  Species           = factor(SPECIES_CODES, levels = SPECIES_CODES),
  Vertical_location = factor(levels(factor(dat_ground_surface$Vertical_location))),
  Photoperiod       = factor(c("Day", "Night"), levels = c("Day", "Night"))
)
newdata_gs$Site         <- dat_ground_surface$Site[1]  # held fixed; Site effect shown via intercept only
newdata_gs$dur_h        <- 12  # reference: predicted visits per 12h
newdata_gs$Session_date <- NA

pred_gs <- predict(m_ground_surface, newdata = newdata_gs, type = "response", se.fit = TRUE, re.form = NA)
newdata_gs$fit <- pred_gs$fit
newdata_gs$se  <- pred_gs$se.fit

build_panel_ground_surface_predicted <- function(theme_fn, errorbar_color) {
  ggplot(newdata_gs, aes(x = Vertical_location, y = fit, fill = Species)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(fit - se, 0), ymax = fit + se),
                  position = position_dodge(width = 0.8), width = 0.2, color = errorbar_color) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    facet_wrap(~ Photoperiod) +
    labs(x = "Logger position", y = "Predicted visits per 12h") +
    theme_fn(base_size = 16)
}

fig3_panel_ground_surface_predicted_dark  <- build_panel_ground_surface_predicted(theme_black, "white")
fig3_panel_ground_surface_predicted_white <- build_panel_ground_surface_predicted(theme_white, "black")

save_panel_pdf(fig3_panel_ground_surface_predicted_dark,  file.path(OUTPUT_DIR, "fig3_panel_ground_surface_predicted_dark.pdf"))
save_panel_pdf(fig3_panel_ground_surface_predicted_white, file.path(OUTPUT_DIR, "fig3_panel_ground_surface_predicted_white.pdf"))

# PRIMARY panel: raw observed visit counts (what was actually recorded), not a model
# prediction - two rainy-season months describe what was found, not a general population
# model; the predicted panel above is a supplementary companion, not the main result.
build_panel_ground_surface_raw <- function(theme_fn) {
  ggplot(dat_ground_surface, aes(x = Vertical_location, y = n_fish, fill = Species)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    facet_wrap(~ Photoperiod) +
    labs(x = "Logger position", y = "Visits per session/Photoperiod") +
    theme_fn(base_size = 16)
}

fig3_panel_ground_surface_dark  <- build_panel_ground_surface_raw(theme_black)
fig3_panel_ground_surface_white <- build_panel_ground_surface_raw(theme_white)

save_panel_pdf(fig3_panel_ground_surface_dark,  file.path(OUTPUT_DIR, "fig3_panel_ground_surface_dark.pdf"))
save_panel_pdf(fig3_panel_ground_surface_white, file.path(OUTPUT_DIR, "fig3_panel_ground_surface_white.pdf"))

# Panel B: predicted visits per 12h by Direction x Species, faceted by Photoperiod (JR1 only)
newdata_jr <- expand.grid(
  Species     = factor(SPECIES_CODES, levels = SPECIES_CODES),
  Direction   = factor(levels(factor(dat_jr$Direction))),
  Photoperiod = factor(c("Day", "Night"), levels = c("Day", "Night"))
)
newdata_jr$dur_h        <- 12
newdata_jr$Session_date <- NA

pred_jr <- predict(m_jr_direction, newdata = newdata_jr, type = "response", se.fit = TRUE, re.form = NA)
newdata_jr$fit <- pred_jr$fit
newdata_jr$se  <- pred_jr$se.fit

build_panel_jr_direction_predicted <- function(theme_fn, errorbar_color) {
  ggplot(newdata_jr, aes(x = Direction, y = fit, fill = Species)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(fit - se, 0), ymax = fit + se),
                  position = position_dodge(width = 0.8), width = 0.2, color = errorbar_color) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    scale_x_discrete(labels = c(Orthogonal = "Orthogonal\n(mid-river)", Parallel = "Parallel\n(shore)")) +
    facet_wrap(~ Photoperiod) +
    labs(x = "Array direction", y = "Predicted visits per 12h") +
    theme_fn(base_size = 16)
}

fig3_panel_jr_direction_predicted_dark  <- build_panel_jr_direction_predicted(theme_black, "white")
fig3_panel_jr_direction_predicted_white <- build_panel_jr_direction_predicted(theme_white, "black")

save_panel_pdf(fig3_panel_jr_direction_predicted_dark,  file.path(OUTPUT_DIR, "fig3_panel_jr_direction_predicted_dark.pdf"))
save_panel_pdf(fig3_panel_jr_direction_predicted_white, file.path(OUTPUT_DIR, "fig3_panel_jr_direction_predicted_white.pdf"))

# PRIMARY panel: raw observed visit counts, same raw-first rationale as ground/surface above.
build_panel_jr_direction_raw <- function(theme_fn) {
  ggplot(dat_jr, aes(x = Direction, y = n_fish, fill = Species)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    scale_x_discrete(labels = c(Orthogonal = "Orthogonal\n(mid-river)", Parallel = "Parallel\n(shore)")) +
    facet_wrap(~ Photoperiod) +
    labs(x = "Array direction", y = "Visits per session/Photoperiod") +
    theme_fn(base_size = 16)
}

fig3_panel_jr_direction_dark  <- build_panel_jr_direction_raw(theme_black)
fig3_panel_jr_direction_white <- build_panel_jr_direction_raw(theme_white)

save_panel_pdf(fig3_panel_jr_direction_dark,  file.path(OUTPUT_DIR, "fig3_panel_jr_direction_dark.pdf"))
save_panel_pdf(fig3_panel_jr_direction_white, file.path(OUTPUT_DIR, "fig3_panel_jr_direction_white.pdf"))

saveRDS(
  list(panel_ground_surface_dark = fig3_panel_ground_surface_dark,
       panel_ground_surface_white = fig3_panel_ground_surface_white,
       panel_ground_surface_predicted_dark = fig3_panel_ground_surface_predicted_dark,
       panel_ground_surface_predicted_white = fig3_panel_ground_surface_predicted_white,
       panel_jr_direction_dark = fig3_panel_jr_direction_dark,
       panel_jr_direction_white = fig3_panel_jr_direction_white,
       panel_jr_direction_predicted_dark = fig3_panel_jr_direction_predicted_dark,
       panel_jr_direction_predicted_white = fig3_panel_jr_direction_predicted_white),
  file.path(OUTPUT_DIR, "fig3_panels.rds")
)

# 6. FISH-LEVEL CORRELATION: did the paired loggers detect the same fish? --------------------
# Proxy analysis using entry_time/exit_time overlap between each pair's two loggers within the
# same session (fish-level, from tracked_fish_summary.csv - already available locally). A true
# pulse-level check (same EOD pulse train appearing on both channels, the gold-standard proof)
# needs the per-event pulse-level tracked tables, which are on an external HDD not yet
# configured (same blocker as RQ iv / 5_Within_Event_Movement.R) - not attempted here.
pair_definitions <- tribble(
  ~Site,  ~logger_a, ~logger_b, ~label_a,      ~label_b,
  "PLP",  "L3",      "L4",      "Ground",      "Surface",
  "SLP",  "L2",      "L1",      "Ground",      "Surface",
  "JR1",  "L5",      "L6",      "Orthogonal",  "Parallel"
)

overlap_results <- NULL
for (i in seq_len(nrow(pair_definitions))) {
  site_i <- pair_definitions$Site[i]
  dat_a <- fish_dat |> filter(Site == site_i, Logger_ID == pair_definitions$logger_a[i])
  dat_b <- fish_dat |> filter(Site == site_i, Logger_ID == pair_definitions$logger_b[i])

  for (sess in intersect(unique(dat_a$Session_date), unique(dat_b$Session_date))) {
    fa <- dat_a |> filter(Session_date == sess)
    fb <- dat_b |> filter(Session_date == sess)
    for (r in seq_len(nrow(fa))) {
      overlap_idx <- which(fa$entry_time[r] < fb$exit_time & fa$exit_time[r] > fb$entry_time)
      if (length(overlap_idx) == 0) next
      overlap_s <- as.numeric(difftime(
        pmin(fa$exit_time[r], fb$exit_time[overlap_idx]),
        pmax(fa$entry_time[r], fb$entry_time[overlap_idx]),
        units = "secs"))
      overlap_results <- bind_rows(overlap_results, tibble(
        Site           = site_i,
        Session_date   = sess,
        fish_key_a     = fa$fish_key[r],
        fish_key_b     = fb$fish_key[overlap_idx],
        species_a      = as.character(fa$Species[r]),
        species_b      = as.character(fb$Species[overlap_idx]),
        overlap_s      = overlap_s,
        same_species   = as.character(fa$Species[r]) == as.character(fb$Species[overlap_idx])
      ))
    }
  }
}

# Summary: of all fish detected at logger A, what fraction have >=1 overlapping fish at logger B?
overlap_summary <- NULL
for (i in seq_len(nrow(pair_definitions))) {
  site_i <- pair_definitions$Site[i]
  n_a <- fish_dat |> filter(Site == site_i, Logger_ID == pair_definitions$logger_a[i]) |> nrow()
  n_b <- fish_dat |> filter(Site == site_i, Logger_ID == pair_definitions$logger_b[i]) |> nrow()
  ov  <- overlap_results |> filter(Site == site_i)
  overlap_summary <- bind_rows(overlap_summary, tibble(
    Site       = site_i,
    pair       = paste0(pair_definitions$label_a[i], " (", pair_definitions$logger_a[i], ") vs ",
                         pair_definitions$label_b[i], " (", pair_definitions$logger_b[i], ")"),
    n_fish_a   = n_a,
    n_fish_b   = n_b,
    n_overlaps = nrow(ov),
    pct_a_with_overlap       = 100 * n_distinct(ov$fish_key_a) / n_a,
    pct_b_with_overlap       = 100 * n_distinct(ov$fish_key_b) / n_b,
    pct_overlap_same_species = if (nrow(ov) > 0) 100 * mean(ov$same_species) else NA_real_,
    median_overlap_s         = if (nrow(ov) > 0) median(ov$overlap_s) else NA_real_
  ))
}

print(overlap_summary)
write.csv2(overlap_summary, file.path(OUTPUT_DIR, "3d_paired_logger_fish_overlap_summary.csv"), row.names = FALSE)
write.csv2(overlap_results, file.path(OUTPUT_DIR, "3d_paired_logger_fish_overlap_detail.csv"), row.names = FALSE)
