# Fig 2 (RQ i): species distribution ~ habitat + environment. Pooled cross-system GLMM plus
# an LN-specific deep-dive (nested LRT: null -> Orientation -> Site, and shore-only
# null -> Veg_cover_extent -> Site).

library(tidyverse)
library(glmmTMB)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), species_palette(), SPECIES_CODES, save_panel_pdf()

DATA_BUNDLE_FILE <- "01 Data/derived_data_bundle.rds"  # written by 3_Analyze_tracking_results.R
OUTPUT_DIR        <- "03 Analysis Output/figures"

bundle       <- readRDS(DATA_BUNDLE_FILE)
fish_dat     <- bundle$fish_dat
register_raw <- bundle$register_raw

# 1. DATA PREP ----------------------------------------------------------------

# one row per session, with the habitat/environment covariates that are constant within a session
session_info <- fish_dat |>
  distinct(System, Site, Session_date, Logger_ID, Orientation, Direction, Vertical_location,
           Veg_cover_extent, T_best, T_best_sd, DO_best, DO_best_sd,
           pH_mean, Cond_mean, Turbidity_mean, Lux_mean)

# species counts per session (Unknown kept as a genuine 5th category, not dropped)
session_species_counts <- fish_dat |>
  count(System, Site, Session_date, Logger_ID, Species, name = "n_fish")

# cross_join with all 4 species so absent species get an explicit 0 count (needed for count models)
habitat_dat <- session_info |>
  cross_join(tibble(Species = SPECIES_CODES)) |>
  left_join(session_species_counts, by = c("System", "Site", "Session_date", "Logger_ID", "Species")) |>
  mutate(n_fish = replace_na(n_fish, 0))

# LNI (floating island) sessions are treated separately in 7_LNI_Floating_Island_Distribution.R
habitat_dat <- habitat_dat |> filter(!grepl("^LNI", Site))

# recording duration, needed as a model offset since sessions vary from ~21h to ~34h and raw
# visit counts (option 1, kept as-is per user decision) scale with how long the logger ran
habitat_dat <- habitat_dat |>
  left_join(register_raw |> distinct(System, Site, Session_date, Logger_ID, Recorded_dur_h),
            by = c("System", "Site", "Session_date", "Logger_ID")) |>
  mutate(Species = factor(Species, levels = SPECIES_CODES),
         System  = factor(System),
         Session_ID = paste(System, Site, Session_date, Logger_ID, sep = "_"))

glimpse(habitat_dat)
summary(habitat_dat)

# 2. MODEL FITTING --------------------------------------------------------------
# Raw visit counts as response (user decision: option 1 - keep counts, use a robust count model
# rather than switching to presence/absence or a rate). Negative binomial to absorb the extreme
# overdispersion from high-traffic creek/paired-logger sessions (PLS L9/L10, SLP L1/L2, PLP L3/L4)
# instead of averaging it away. log(Recorded_dur_h) offset accounts for variable session length.
# Turbidity_mean excluded (missing in ~45% of sessions - would cost too much data via listwise
# deletion). Lux_mean excluded (a session-long average washes out the diel light signal into a
# number with no clear physical interpretation, unlike T/DO/pH which are meaningful as means).
# T_best, DO_best, pH_mean, Cond_mean still have some missingness (see summary() above) so this
# model's effective sample size is smaller than 65 sessions - noted for review at diagnostics.
# (1 | Site) dropped: check_singularity() confirmed its ~0 variance is a boundary/singular fit,
# redundant once System is already a fixed effect.
m_habitat <- glmmTMB(
  n_fish ~ Species * System + T_best + DO_best + pH_mean + Cond_mean +
    offset(log(Recorded_dur_h)) + (1 | Session_ID),
  family = nbinom2,
  data = habitat_dat
)

summary(m_habitat)

# 3. DIAGNOSTICS ------------------------------------------------------------
library(DHARMa)
library(performance)

check_singularity(m_habitat)

sim_res <- simulateResiduals(m_habitat, n = 1000)
testDispersion(sim_res)
testOutliers(sim_res)

# use the model's own model.frame, not habitat_dat directly - glmmTMB drops NA rows internally,
# so habitat_dat (260 rows) and sim_res (204 rows) no longer line up
mf <- model.frame(m_habitat)

pdf(file.path(OUTPUT_DIR, "3b_habitat_model_diagnostics.pdf"), width = 8, height = 6)
plot(sim_res)
plotResiduals(sim_res, mf$Species)
plotResiduals(sim_res, mf$System)
dev.off()

# 4. LN SITE-LEVEL ANALYSIS ---------------------------------------------------
# JR/PL/SL are near-single-site systems sampled repeatedly (SLP for L1-2, PLP for L3-4, JR1 for
# L5-6, JR2 a one-off exception) - Site there is redundant with System, which is why (1 | Site)
# above collapsed to ~0 variance. LN is the opposite: every site was sampled exactly once, spread
# across the lake - Site and Session_ID are perfectly aliased there, so the pooled model can't
# isolate genuine LN site-to-site spatial variation from ordinary session-to-session noise.
# Test it directly within LN only, where Site is estimable as its own fixed effect.
habitat_dat_LN <- habitat_dat |> filter(System == "LN")

m_LN_null <- glmmTMB(
  n_fish ~ Species + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN
)

m_LN_site <- glmmTMB(
  n_fish ~ Species + Site + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN
)

anova(m_LN_null, m_LN_site)

# Veg_cover_extent (below) only partly explained the site effect and is shore-only anyway. Before
# that, test the coarser but universally-defined shore/open-water split itself: Orientation is
# deterministic of vegetation presence (all Vertical loggers are open-water with none, all
# Horizontal ones are along the shore) and, per field experience, fish strongly prefer structure/
# vegetation - so shore vs open-water is a prime candidate for what's driving the site effect.
m_LN_orient <- glmmTMB(
  n_fish ~ Species + Orientation + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN
)

anova(m_LN_null, m_LN_orient, m_LN_site)

# Raw Site identity confirmed to matter (see anova above) but isn't itself interpretable - test
# whether the more parsimonious, biologically interpretable Veg_cover_extent captures a
# meaningful share of that site-to-site signal. Veg_cover_extent is only defined for Orthogonal
# (shore) loggers, so this comparison is restricted to that subset - refit null/site on the SAME
# shore-only rows here for a fair (equal-N) comparison against the veg-cover model.
habitat_dat_LN_shore <- habitat_dat_LN |> filter(Direction == "Orthogonal")

m_LN_shore_null <- glmmTMB(
  n_fish ~ Species + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN_shore
)

m_LN_shore_veg <- glmmTMB(
  n_fish ~ Species + Veg_cover_extent + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN_shore
)

m_LN_shore_site <- glmmTMB(
  n_fish ~ Species + Site + offset(log(Recorded_dur_h)),
  family = nbinom2,
  data = habitat_dat_LN_shore
)

anova(m_LN_shore_null, m_LN_shore_veg, m_LN_shore_site)

# 5. EXPORT MODEL RESULTS ----------------------------------------------------
library(broom.mixed)

coef_tables <- list(
  m_habitat       = m_habitat,
  m_LN_null       = m_LN_null,
  m_LN_site       = m_LN_site,
  m_LN_orient     = m_LN_orient,
  m_LN_shore_null = m_LN_shore_null,
  m_LN_shore_veg  = m_LN_shore_veg,
  m_LN_shore_site = m_LN_shore_site
)
model_coefficients <- imap(coef_tables, \(m, name) tidy(m, effects = "fixed", conf.int = TRUE) |> mutate(model = name, .before = 1)) |>
  bind_rows()
write.csv(model_coefficients, file.path(OUTPUT_DIR, "3b_model_coefficients.csv"), row.names = FALSE)

model_comparisons <- bind_rows(
  as.data.frame(anova(m_LN_null, m_LN_site, m_LN_orient)) |> rownames_to_column("model") |> mutate(comparison = "LN_null_site_orient", .before = 1),
  as.data.frame(anova(m_LN_shore_null, m_LN_shore_veg, m_LN_shore_site)) |> rownames_to_column("model") |> mutate(comparison = "LN_shore_null_veg_site", .before = 1)
)
write.csv(model_comparisons, file.path(OUTPUT_DIR, "3b_model_comparisons.csv"), row.names = FALSE)

# 6. RESULT PLOTS (Fig 2 panels) --------------------------------------------

# Panel A: pooled cross-system model - predicted visits per 24h by System x Species, holding
# continuous covariates at their sample mean and random effects out (population-level prediction)
newdata_system <- expand.grid(
  Species = factor(SPECIES_CODES, levels = SPECIES_CODES),
  System  = factor(levels(habitat_dat$System), levels = levels(habitat_dat$System))
)
newdata_system$T_best         <- mean(habitat_dat$T_best, na.rm = TRUE)
newdata_system$DO_best        <- mean(habitat_dat$DO_best, na.rm = TRUE)
newdata_system$pH_mean        <- mean(habitat_dat$pH_mean, na.rm = TRUE)
newdata_system$Cond_mean      <- mean(habitat_dat$Cond_mean, na.rm = TRUE)
newdata_system$Recorded_dur_h <- 24  # reference: predicted visits per 24h
newdata_system$Session_ID     <- NA

pred_system <- predict(m_habitat, newdata = newdata_system, type = "response",
                        se.fit = TRUE, re.form = NA)
newdata_system$fit <- pred_system$fit
newdata_system$se  <- pred_system$se.fit

# built as a function of theme + errorbar color so the same panel can be rendered for both the
# poster/presentation (dark) and manuscript (white/print) themes
build_panel_system_predicted <- function(theme_fn, errorbar_color) {
  ggplot(newdata_system, aes(x = System, y = fit, fill = Species)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = pmax(fit - se, 0), ymax = fit + se),
                  position = position_dodge(width = 0.8), width = 0.2, color = errorbar_color) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    labs(x = "System", y = "Predicted visits per 24h") +
    theme_fn(base_size = 16)
}

fig2_panel_system_predicted_dark  <- build_panel_system_predicted(theme_black, "white")
fig2_panel_system_predicted_white <- build_panel_system_predicted(theme_white, "black")

# PRIMARY panel: raw observed visit counts per session (what was actually recorded), not a
# model prediction - two rainy-season months are enough to describe what was found, not to
# generalize a population-level model; the predicted panel above is kept as a supplementary
# companion (what the model suggests about general patterns), not the main result.
build_panel_system_raw <- function(theme_fn) {
  ggplot(habitat_dat, aes(x = System, y = n_fish, fill = Species)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    labs(x = "System", y = "Visits per session") +
    theme_fn(base_size = 16)
}

fig2_panel_system_raw_dark  <- build_panel_system_raw(theme_black)
fig2_panel_system_raw_white <- build_panel_system_raw(theme_white)

# Panel B: LN shore vs open water - raw visit counts, showing the structure-preference pattern
build_panel_LN_orientation <- function(theme_fn) {
  ggplot(habitat_dat_LN, aes(x = Orientation, y = n_fish, fill = Species)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    labs(x = "Orientation", y = "Visits per session") +
    theme_fn(base_size = 16)
}

fig2_panel_LN_orientation_dark  <- build_panel_LN_orientation(theme_black)
fig2_panel_LN_orientation_white <- build_panel_LN_orientation(theme_white)

fig2_panel_system_raw_white
fig2_panel_LN_orientation_white

save_panel_pdf(fig2_panel_system_raw_dark,        file.path(OUTPUT_DIR, "fig2_panel_system_dark.pdf"))
save_panel_pdf(fig2_panel_system_raw_white,       file.path(OUTPUT_DIR, "fig2_panel_system_white.pdf"))
save_panel_pdf(fig2_panel_system_predicted_dark,  file.path(OUTPUT_DIR, "fig2_panel_system_predicted_dark.pdf"))
save_panel_pdf(fig2_panel_system_predicted_white, file.path(OUTPUT_DIR, "fig2_panel_system_predicted_white.pdf"))
save_panel_pdf(fig2_panel_LN_orientation_dark,  file.path(OUTPUT_DIR, "fig2_panel_LN_orientation_dark.pdf"))
save_panel_pdf(fig2_panel_LN_orientation_white, file.path(OUTPUT_DIR, "fig2_panel_LN_orientation_white.pdf"))

# kept for convenience (e.g. quick reload while tweaking figures) - not required for the
# Illustrator workflow, where the PDFs above are the actual deliverable
saveRDS(
  list(panel_system_raw_dark = fig2_panel_system_raw_dark, panel_system_raw_white = fig2_panel_system_raw_white,
       panel_system_predicted_dark = fig2_panel_system_predicted_dark,
       panel_system_predicted_white = fig2_panel_system_predicted_white,
       panel_LN_orientation_dark = fig2_panel_LN_orientation_dark,
       panel_LN_orientation_white = fig2_panel_LN_orientation_white),
  file.path(OUTPUT_DIR, "fig2_panels.rds")
)
