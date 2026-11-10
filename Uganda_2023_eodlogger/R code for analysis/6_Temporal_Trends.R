# Fig 2 addition (RQ v): does visit activity trend change over the season (weekly repeat
# visits), and does that trend differ by System/Species? Folds into Fig 2 as an additional
# color/facet panel (not a separate main figure - stays within the 5-figure + 1-supplementary
# publication budget).

library(tidyverse)
library(glmmTMB)
library(DHARMa)
library(performance)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), species_palette(), SPECIES_CODES, save_panel_pdf()

DATA_BUNDLE_FILE <- "01 Data/derived_data_bundle.rds"  # written by 3_Analyze_tracking_results.R
OUTPUT_DIR       <- "03 Analysis Output/figures"

bundle       <- readRDS(DATA_BUNDLE_FILE)
fish_dat     <- bundle$fish_dat
register_raw <- bundle$register_raw

# 1. DATA PREP ----------------------------------------------------------------
# Restricted to genuinely repeat-sampled sites (SLP/PLP/PLS/JR1) - every LN sub-site and JR2 was
# visited exactly once (confirmed against Recordings_Sessions_Metadata.csv during the Fig 5
# redesign), so Session_date is perfectly aliased with site identity there and cannot support a
# temporal-trend claim (see 4_Breeding_Indicators.R for the full rationale/verification).
REPEAT_SAMPLED_SITES <- c("SLP", "PLP", "PLS", "JR1")

session_info <- fish_dat |>
  filter(Site %in% REPEAT_SAMPLED_SITES) |>
  distinct(System, Site, Session_date, Logger_ID, T_best, T_best_sd, DO_best, DO_best_sd,
           pH_mean, Cond_mean, Turbidity_mean, Lux_mean)

# species counts per session (Unknown kept as a genuine 5th category, matching 3b's convention)
session_species_counts <- fish_dat |>
  filter(Site %in% REPEAT_SAMPLED_SITES) |>
  count(System, Site, Session_date, Logger_ID, Species, name = "n_fish")

# cross_join with all 5 species so absent species get an explicit 0 count (needed for count models)
temporal_dat <- session_info |>
  cross_join(tibble(Species = SPECIES_CODES)) |>
  left_join(session_species_counts, by = c("System", "Site", "Session_date", "Logger_ID", "Species")) |>
  mutate(n_fish = replace_na(n_fish, 0)) |>
  left_join(register_raw |> distinct(System, Site, Session_date, Logger_ID, Recorded_dur_h),
            by = c("System", "Site", "Session_date", "Logger_ID")) |>
  mutate(Species = factor(Species, levels = SPECIES_CODES),
         System  = factor(System),
         Session_ID = paste(System, Site, Session_date, Logger_ID, sep = "_"),
         Session_date_num = as.numeric(Session_date - min(Session_date)))  # days since first session

glimpse(temporal_dat)
summary(temporal_dat)

# 2. MODEL --------------------------------------------------------------------
# Species*Session_date_num + Species*System (no System:Session_date_num) - the full 3-way
# interaction scope was tried first but showed significant underdispersion (p=0.024, DHARMa):
# with missingness in T_best/DO_best/pH_mean/Cond_mean dropping to only 37 usable sessions, the
# full scope (26 fixed params) was overparameterized (~7:1 obs:param ratio). Dropping the
# weaker System:Session_date_num term (PL wasn't significant there, only SL was) keeps the two
# interactions that answer RQ v most directly: does the trend differ by species, and does
# overall level differ by system. Same count-model pattern as 3b's habitat model (nbinom2,
# log(Recorded_dur_h) offset). Site omitted - with only 4 sites left (JR1/PLP/PLS/SLP) it is
# largely redundant with System (only PL has 2 distinct sites), same reasoning as 3b/Fig 5.
m_temporal <- glmmTMB(
  n_fish ~ Species * Session_date_num + Species * System +
    T_best + DO_best + pH_mean + Cond_mean + offset(log(Recorded_dur_h)) + (1 | Session_ID),
  family = nbinom2,
  data = temporal_dat
)
summary(m_temporal)

# 3. DIAGNOSTICS ----------------------------------------------------------------

check_singularity(m_temporal)

sim_res_temporal <- simulateResiduals(m_temporal, n = 1000)
testDispersion(sim_res_temporal)
testOutliers(sim_res_temporal)

# use the model's own model.frame - glmmTMB drops NA rows internally, so temporal_dat and
# sim_res_temporal may not line up row-for-row otherwise
mf <- model.frame(m_temporal)

pdf(file.path(OUTPUT_DIR, "6_temporal_trend_diagnostics.pdf"), width = 8, height = 6)
plot(sim_res_temporal)
plotResiduals(sim_res_temporal, mf$Species)
plotResiduals(sim_res_temporal, mf$System)
dev.off()

# 4. EXPORT MODEL RESULTS ----------------------------------------------------
library(broom.mixed)

model_coefficients <- tidy(m_temporal, effects = "fixed", conf.int = TRUE) |> mutate(model = "m_temporal", .before = 1)
write.csv(model_coefficients, file.path(OUTPUT_DIR, "6_model_coefficients.csv"), row.names = FALSE)

# 5. RESULT PLOT (Fig 2 additional panel) --------------------------------------------------
# PRIMARY: raw n_fish per session over the season, faceted by System, colored by Species - the
# actual observed data (raw-data-first convention). Model-predicted trend (population-level,
# continuous covariates held at their sample mean, random effect averaged out) drawn on top as
# the supplementary layer.
newdata_temporal <- temporal_dat |>
  distinct(System, Species) |>
  cross_join(tibble(Session_date_num = seq(min(temporal_dat$Session_date_num),
                                            max(temporal_dat$Session_date_num), length.out = 50)))
newdata_temporal$T_best         <- mean(temporal_dat$T_best, na.rm = TRUE)
newdata_temporal$DO_best        <- mean(temporal_dat$DO_best, na.rm = TRUE)
newdata_temporal$pH_mean        <- mean(temporal_dat$pH_mean, na.rm = TRUE)
newdata_temporal$Cond_mean      <- mean(temporal_dat$Cond_mean, na.rm = TRUE)
newdata_temporal$Recorded_dur_h <- 24  # reference: predicted visits per 24h
newdata_temporal$Session_ID     <- NA

pred_temporal <- predict(m_temporal, newdata = newdata_temporal, type = "response",
                          se.fit = TRUE, re.form = NA)
newdata_temporal$fit   <- pred_temporal$fit
newdata_temporal$upper <- pred_temporal$fit + pred_temporal$se.fit
newdata_temporal$lower <- pmax(pred_temporal$fit - pred_temporal$se.fit, 0)

build_panel_temporal_trend <- function(theme_fn) {
  ggplot(temporal_dat, aes(x = Session_date_num, y = n_fish, color = Species)) +
    geom_point(alpha = 0.3, size = 1.2) +
    geom_ribbon(data = newdata_temporal, aes(y = fit, ymin = lower, ymax = upper, fill = Species),
                color = NA, alpha = 0.2) +
    geom_line(data = newdata_temporal, aes(y = fit), linewidth = 0.9) +
    scale_color_manual(values = species_palette(theme_fn)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    facet_wrap(~ System, scales = "free_y") +
    labs(x = "Days since first session",
         y = "Visits per session - points = observed, line = predicted (per 24h, covariates at mean)") +
    theme_fn(base_size = 16)
}

fig2_panel_temporal_trend_dark  <- build_panel_temporal_trend(theme_black)
fig2_panel_temporal_trend_white <- build_panel_temporal_trend(theme_white)

save_panel_pdf(fig2_panel_temporal_trend_dark,  file.path(OUTPUT_DIR, "fig2_panel_temporal_trend_dark.pdf"))
save_panel_pdf(fig2_panel_temporal_trend_white, file.path(OUTPUT_DIR, "fig2_panel_temporal_trend_white.pdf"))

saveRDS(
  list(panel_temporal_trend_dark = fig2_panel_temporal_trend_dark,
       panel_temporal_trend_white = fig2_panel_temporal_trend_white),
  file.path(OUTPUT_DIR, "fig2_temporal_panel.rds")
)
