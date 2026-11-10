# Fig 1 (RQ i/ii descriptive): site overview - timelines, species composition, EOD width,
# movement/position panel. Loads Python tracking output + environmental data + session
# metadata, derives the canonical Species column, and exports the shared derived_data_bundle.rds
# that every downstream RQ script (3b/3c/3d/4/5/6/7) loads instead of re-sourcing this script.

library(ggpubr)
library(tidyverse)
library(lubridate)
library(sf)
library(openxlsx)
library(scales)
library(readxl)
library(suncalc)

rm(list=ls())


# ---- Set Directories ---------------------------------------------------------
TRACK_DIR         <- "F:/Tracking Results/final_factor3.0_prob0.8"  # root containing tracking summaries (output from python scripts 04_1 + 04_1c reclassification)
# ENV_DATA_FILE     <- "01 Data/Environmental Data/Complete_env_data_summary_per_day.csv"   # semicolon-separated
ENV_DATA_DIR      <- "01 Data/Environmental Data/"   # directory containing environmental data files
ELECTRODE_METADATA_FILE <- "eod_logger_4/site_electrode_metadata.csv"  # electrode index -> physical position lookup
OUTPUT_DIR        <- "03 Analysis Output/figures"
DATA_BUNDLE_FILE  <- "01 Data/derived_data_bundle.rds"  # shared loaded/derived data for downstream RQ scripts (3b, 3c, 3d, 4, 5, 6, 7)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Plot Settings ------------------------------
# theme_black(), SPECIES_PALETTE, SPECIES_CODES, cbf_palette - shared with all downstream RQ scripts
source("02 Code/R code/00_plot_helpers.R")


# 2. DATA LOADING ------------------------------------------------------------

# ---- Session metadata (script 2 output) --------------------------------------
# Replaces the old Excel-based register, its separate electrode-metadata join, and its
# manual single-coordinate sunrise/sunset calc. Sunrise here already IS the next sunrise
# after Sunset (script 2 computes both within a standardized 24h window per session), so
# no separate Sunrise_next/Sunset_next is needed.
register_raw <- read.csv2("01 Data/Recordings_Sessions_Environment_Complete.csv") |>
  mutate(
    t_first_rec_start = as.POSIXct(t_first_rec_start, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi"),
    t_last_rec_end    = as.POSIXct(t_last_rec_end,    format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi"),
    Sunrise           = as.POSIXct(Sunrise, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi"),
    Sunset            = as.POSIXct(Sunset,  format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi"),
    Session_date      = as.Date(Session_date),
    Session_date_str  = format(Session_date, "%Y%m%d"),
    # Applies to all horizontal loggers (any System/Direction): edge_ch 0 = no cover reaches into
    # the array, 7 = cover reaches the far end (complete), in between = partial. Direction-agnostic
    # (doesn't matter which end - Ch_0 or Ch_7 - vegetation grows from). Not applicable to vertical
    # (Point) loggers (NA). LN sessions vary logger-to-logger (kept per-session, not a site constant);
    # PL/SL are consistently partial (edge 3-4), JR varies by logger (L5 partial, L6 complete).
    Veg_cover_extent  = case_when(
      Orientation != "Horizontal" ~ NA_character_,
      Vegetation_cover_edge_ch <= 0 ~ "None",
      Vegetation_cover_edge_ch >= 7 ~ "Complete",
      TRUE ~ "Partial"
    )
  ) |>
  filter(!is.na(t_first_rec_start))

# ---- Environmental data (semicolon-separated) --------------------------------
# Summary data
env_raw <- read.csv2(file.path(ENV_DATA_DIR, "Complete_env_data_summary_per_day.csv"))

env_clean <- env_raw |>
  # Force all numeric-intended columns to numeric regardless of how read_delim
  # inferred them (e.g. cells containing "NA" strings cause character inference)
  mutate(across(c(T_C_mean, T_C_sd, T_C_min, T_C_max,
                  T_C_DO_mean, T_C_DO_sd, T_C_DO_min, T_C_DO_max,
                  T_C_MM_mean, T_C_MM_sd, T_C_MM_min, T_C_MM_max,
                  DO_AS_mean, DO_AS_sd, DO_AS_min, DO_AS_max,
                  DO_AS_MM_mean, DO_AS_MM_sd, DO_AS_MM_min, DO_AS_MM_max,
                  pH_mean, pH_sd, pH_min, pH_max,
                  Cond_mean, Cond_sd, Cond_min, Cond_max,
                  Turbidity_mean, Turbidity_sd, Turbidity_min, Turbidity_max,
                  Lux_mean, Lux_sd, Lux_min, Lux_max),
                ~ suppressWarnings(as.numeric(.)))) |>
  mutate(
    Session_date = as.Date(Session_date),
    Session_date_str = format(Session_date, "%Y%m%d"),
    # Temperature priority: logger > DO-logger > manual
    T_best    = coalesce(T_C_mean,    T_C_DO_mean,  T_C_MM_mean),
    T_best_sd = coalesce(T_C_sd,      T_C_DO_sd,    T_C_MM_sd),
    # DO priority: logger > manual
    DO_best    = coalesce(DO_AS_mean,  DO_AS_MM_mean),
    DO_best_sd = coalesce(DO_AS_sd,    DO_AS_MM_sd)
  )


# Daily light and temperature data from loggers (for diel plots)
light_temp <- read.csv2(file.path(ENV_DATA_DIR, "Temp_Light_All_Logger_Data.csv"))
light_temp$Datetime <- as.POSIXct(light_temp$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
light_temp$Datetime_fix <- as.POSIXct(light_temp$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
light_temp$Datetime_fix_loop <- as.POSIXct(light_temp$Datetime_fix_loop, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
# Photoperiod already computed upstream (scripts 1_1/1_2) per-record - no re-derivation needed here.

# Create a running index for plots that increases every time the date changes (i.e., midnight) to avoid lines connecting the last point of one day to the first point of the next day
light_temp <- light_temp %>%
  arrange(Datetime) %>%
  mutate(Index_plot = cumsum(c(1, diff(as.Date(Datetime)) != 0)))
light_dat <- subset(light_temp, Source == "HOBO")
temp_dat <- subset(light_temp, Source == "TinyTag")


## 1.2 DO PyroScience logger
DO_pyro <- read.csv2(file.path(ENV_DATA_DIR, "All_pyroscience_data.csv"))
DO_pyro$Datetime <- as.POSIXct(DO_pyro$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_pyro$Datetime_fix <- as.POSIXct(DO_pyro$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")

DO_pyro$Source <- "Pyro"

## 1.2 DO MiniDOT logger
DO_miniDOT <- read.csv2(file.path(ENV_DATA_DIR, "All_miniDOT_data.csv"))
DO_miniDOT$Datetime <- as.POSIXct(DO_miniDOT$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_miniDOT$Datetime_fix <- as.POSIXct(DO_miniDOT$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_miniDOT$Source <- "miniDOT"

# 1.3 Combined DO data
DO_combined <- rbind.data.frame(DO_pyro, DO_miniDOT)
# Photoperiod already computed upstream (scripts 1_4/1_5) per-record - no re-derivation needed here.

# tracking data -----------------------------------------------------------
fish_track_list <- list.files(TRACK_DIR, pattern = "tracked_fish_summary\\.csv",
                              recursive = TRUE, full.names = TRUE)
fish_raw <- NULL
for(ft_file in fish_track_list){
  folder_parts <- strsplit(basename(dirname(ft_file)), "_")[[1]]
  logger_id <- folder_parts[1]
  system <- basename(dirname(dirname(ft_file)))
  site <- folder_parts[3]
  session_date <- as.Date(folder_parts[2], format = "%Y%m%d")
  # Check if logger ID in register_raw before continuing
  if(!any(register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site)){
    next
  }
  ft_dat <- read.csv2(ft_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  # session folder is named Logger_SessionDate_Site, its parent folder is System
  ft_dat$Logger_ID <- logger_id
  ft_dat$Session_date <- session_date
  ft_dat$Site <- site
  ft_dat$System <- system
  ft_dat$Site_logger <- paste(site, logger_id, sep="_")
  ft_dat$Placement <- register_raw$Placement[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  # Orientation: Horizontal/Vertical; Direction: Orthogonal/Parallel (relative to shore/flow, horizontal loggers only)
  ft_dat$Orientation <- register_raw$Orientation[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Direction <- register_raw$Direction[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Vertical_location <- register_raw$Vertical_location[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Ch_0 <- register_raw$Ch_0[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Ch_7 <- register_raw$Ch_7[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Vegetation_cover_edge_ch <- register_raw$Vegetation_cover_edge_ch[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Veg_cover_extent <- register_raw$Veg_cover_extent[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Sunrise <- register_raw$Sunrise[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ft_dat$Sunset <- register_raw$Sunset[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  fish_raw <- bind_rows(fish_raw, ft_dat)
}
fish_raw$entry_time <- as.POSIXct(fish_raw$entry_time, tz = "Africa/Nairobi")
fish_raw$exit_time <- as.POSIXct(fish_raw$exit_time, tz = "Africa/Nairobi")
fish_raw$visit_midpoint <- fish_raw$entry_time + (fish_raw$exit_time - fish_raw$entry_time)/2
fish_raw$visit_duration_min <- as.numeric(difftime(fish_raw$exit_time, fish_raw$entry_time, units = "mins"))
fish_raw$std_time_ss <- as.numeric(difftime(fish_raw$visit_midpoint, fish_raw$Sunset, units = "hours"))
fish_raw$std_time_sr <- as.numeric(difftime(fish_raw$visit_midpoint, fish_raw$Sunrise, units = "hours"))
fish_raw$Photoperiod <- ifelse(fish_raw$std_time_ss >= 0 & fish_raw$std_time_sr <= 0, "Night", "Day")
# electrode index -> physical distance/depth: fixed 8-electrode array at 0.5 m spacing (Ch_0 to Ch_7 = 3.5 m),
# same span regardless of orientation (Horizontal: distance from Ch_0; Vertical: depth from Ch_0)
fish_raw$gradient_endpoint_m <- 3.5
fish_raw$location_m <- fish_raw$mean_location * (fish_raw$gradient_endpoint_m / 7)
# Canonical species label used by all downstream plots/models: uncertain-flagged fish become
# a genuine "Unknown" category (kept in the data) rather than being silently dropped.
fish_raw$Species <- ifelse(fish_raw$species_uncertain == "True", "Unknown", fish_raw$species_assigned)

# join with environmental data
fish_dat <- fish_raw |>
  left_join(env_clean |> select(System, Site, Session_date, Logger_ID,
                                 T_best, T_best_sd, DO_best, DO_best_sd,
                                 pH_mean, Cond_mean, Turbidity_mean, Lux_mean),
            by = c("System", "Site", "Logger_ID",
                   "Session_date")) 

event_list <- list.files(TRACK_DIR, pattern = "tracked_event_summary\\.csv",
                         recursive = TRUE, full.names = TRUE)

events_raw <- NULL
for(ev_file in event_list){
  folder_parts <- strsplit(basename(dirname(ev_file)), "_")[[1]]
  logger_id <- folder_parts[1]
  system <- basename(dirname(dirname(ev_file)))
  site <- folder_parts[3]
  session_date <- as.Date(folder_parts[2], format = "%Y%m%d")
  # Check if logger ID in register_raw before continuing
  if(!any(register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site)){
    next
  }
  # read ev_dat and set channels_used column to character to avoid type mismatch when binding rows
  ev_dat <- read.csv2(ev_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  ev_dat$channels_used <- as.character(ev_dat$channels_used)
  # session folder is named Logger_SessionDate_Site, its parent folder is System
  ev_dat$Logger_ID <- logger_id
  ev_dat$Session_date <- session_date
  ev_dat$Site <- site
  ev_dat$System <- system
  ev_dat$Site_logger <- paste(site, logger_id, sep="_")
  ev_dat$Placement <- register_raw$Placement[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ev_dat$Sunrise <- register_raw$Sunrise[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  ev_dat$Sunset <- register_raw$Sunset[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  events_raw <- bind_rows(events_raw, ev_dat)
}
events_raw$eod_start_time <- as.POSIXct(events_raw$eod_start_time, tz = "Africa/Nairobi")
events_raw$eod_end_time <- as.POSIXct(events_raw$eod_end_time, tz = "Africa/Nairobi")
events_raw$event_start_time <- as.POSIXct(events_raw$event_start_time, tz = "Africa/Nairobi")
events_raw$event_end_time <- as.POSIXct(events_raw$event_end_time, tz = "Africa/Nairobi")
events_raw$std_time_ss <- as.numeric(difftime(events_raw$eod_start_time, events_raw$Sunset, units = "hours"))
events_raw$std_time_sr <- as.numeric(difftime(events_raw$eod_start_time, events_raw$Sunrise, units = "hours"))
events_raw$Photoperiod <- ifelse(events_raw$std_time_ss >= 0 & events_raw$std_time_sr <= 0, "Night", "Day")

species_list <- list.files(TRACK_DIR, pattern = "tracked_species_summary\\.csv",
                           recursive = TRUE, full.names = TRUE)

species_raw <- NULL
for(sp_file in species_list){
  folder_parts <- strsplit(basename(dirname(sp_file)), "_")[[1]]
  logger_id <- folder_parts[1]
  system <- basename(dirname(dirname(sp_file)))
  site <- folder_parts[3]
  session_date <- as.Date(folder_parts[2], format = "%Y%m%d")
  # Check if logger ID in register_raw before continuing
  if(!any(register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site)){
    next
  }
  sp_dat <- read.csv2(sp_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  # session folder is named Logger_SessionDate_Site, its parent folder is System
  sp_dat$Logger_ID <- logger_id
  sp_dat$Session_date <- session_date
  sp_dat$Site <- site
  sp_dat$System <- system
  sp_dat$Site_logger <- paste(site, logger_id, sep="_")
  sp_dat$Placement <- register_raw$Placement[register_raw$Session_date == session_date & register_raw$Logger_ID == logger_id & register_raw$System == system & register_raw$Site == site]
  species_raw <- bind_rows(species_raw, sp_dat)
}

# ---- Recompute time series from fish_raw, now including Unknown (species-uncertain) fish -----
# ts_raw is built per-channel and does not account for species assignment (unlike fish_raw).
# Rebuild a presence time series directly from entry_time/exit_time of each fish, ignoring
# channel identity.

TS_BIN_MIN <- 1  # time bin resolution (minutes) for the recomputed presence time series

ts_expanded <- fish_raw |>
  rowwise() |>
  mutate(datetime = list(seq(floor_date(entry_time, unit = paste(TS_BIN_MIN, "minutes")),
                              exit_time,
                              by = paste(TS_BIN_MIN, "min")))) |>
  ungroup() |>
  select(fish_id, Species, Site, System, Placement, Orientation, Direction,
         Session_date, Logger_ID, Sunrise, Sunset, datetime) |>
  unnest(datetime)

ts_new <- ts_expanded |>
  count(Site, System, Placement, Orientation, Direction, Session_date, Logger_ID,
        Sunrise, Sunset,
        datetime, Species, name = "Number") |>
  pivot_wider(names_from = Species, values_from = Number, values_fill = 0,
              names_prefix = "n_")

for (sp_col in paste0("n_", SPECIES_CODES)) {
  if (!sp_col %in% names(ts_new)) ts_new[[sp_col]] <- 0
}

ts_new$std_time_ss <- as.numeric(difftime(ts_new$datetime, ts_new$Sunset, units = "hours"))
ts_new$std_time_sr <- as.numeric(difftime(ts_new$datetime, ts_new$Sunrise, units = "hours"))
ts_new$Photoperiod <- ifelse(ts_new$std_time_ss >= 0 & ts_new$std_time_sr <= 0, "Night", "Day")

# ---- Environmental data coverage tally (per System x Site) ---------------------
# register_raw already carries the sensor aggregate columns joined in script 2, one row
# per actual recording session; distinguishes continuous logger coverage (which supports
# diel analysis, RQ ii) from manual spot measurements (broad but not diel-resolution, RQ i).
env_coverage_summary <- register_raw |>
  transmute(System, Site, Session_date,
            has_temp_logger  = !is.na(T_C_mean),
            has_light_logger = !is.na(Lux_mean),
            has_DO_logger    = !is.na(DO_AS_mean),
            has_manual_meas  = !is.na(MM_n) & MM_n > 0) |>
  group_by(System, Site) |>
  summarise(n_sessions       = n(),
            n_with_temp      = sum(has_temp_logger),
            n_with_light     = sum(has_light_logger),
            n_with_DO_logger = sum(has_DO_logger),
            n_with_manual    = sum(has_manual_meas),
            date_min         = min(Session_date),
            date_max         = max(Session_date),
            .groups = "drop")

write.csv2(env_coverage_summary, file.path(OUTPUT_DIR, "env_coverage_summary.csv"), row.names = FALSE)

# ---- Export shared data bundle for downstream RQ scripts (3b, 3c, 3d, 4, 5, 6, 7) -----------
# Avoids every downstream script re-sourcing this file's ~350 lines of loading/parsing/derivation.
saveRDS(
  list(
    register_raw          = register_raw,
    env_clean             = env_clean,
    light_temp            = light_temp,
    DO_combined           = DO_combined,
    fish_raw              = fish_raw,
    fish_dat              = fish_dat,
    events_raw            = events_raw,
    species_raw           = species_raw,
    ts_new                = ts_new,
    env_coverage_summary  = env_coverage_summary
  ),
  DATA_BUNDLE_FILE
)

# 3. PLOTS -----------------------------------------------

# Time Series -------------------------------------------------------------

# summarize ts dataset per Site and per 30 min interval
# compute sum, mean, sd, and 95% CI of number of fish per species per 30 min interval
# 30 min interval = 0.5 hours, so we can round std_time to the nearest 0.5
ts_new$std_time_rounded <- round(ts_new$std_time_ss * 2) / 2

# Special dataset for LN to separate open water and shore loggers
ts_LN_stack <- ts_new %>%
  filter(System == "LN") |> 
  group_by(Site, System, Direction, Orientation, std_time_rounded) %>%
  summarise(
    n_GL = sum(n_GL, na.rm = TRUE),
    n_PD = sum(n_PD, na.rm = TRUE),
    n_MV = sum(n_MV, na.rm = TRUE),
    n_PN = sum(n_PN, na.rm = TRUE),
    n_Unknown = sum(n_Unknown, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = starts_with("n_"), names_to = "Species", values_to = "Number") %>%
  mutate(Species = str_remove(Species, "^n_"))

ts_stack <- ts_new %>%
  # filter(System != "LN") |> 
  group_by(Site, System, std_time_rounded) %>%
  summarise(
    n_GL = sum(n_GL, na.rm = TRUE),
    n_PD = sum(n_PD, na.rm = TRUE),
    n_MV = sum(n_MV, na.rm = TRUE),
    n_PN = sum(n_PN, na.rm = TRUE),
    n_Unknown = sum(n_Unknown, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = starts_with("n_"), names_to = "Species", values_to = "Number") %>%
  mutate(Species = str_remove(Species, "^n_")) %>%
  # fill time bins with no detections of a species with an explicit 0, so area plots
  # ramp from zero instead of appearing/disappearing out of nothing
  group_by(Site, System) %>%
  complete(std_time_rounded = full_seq(std_time_rounded, 0.5), Species = SPECIES_CODES,
           fill = list(Number = 0)) %>%
  ungroup()

# Special subset for LN data because it has different sites and we want to combine the shore sites and the open water sites separately for overview plots
# Shore = horizontal loggers extending shore-perpendicular; Open water = vertically floating loggers (see user domain notes)
ts_LN_shore_stack <- ts_LN_stack %>%
  filter(System == "LN" & Direction == "Orthogonal") %>%
  group_by(System, std_time_rounded, Species) %>%
  summarise(Number = sum(Number, na.rm = TRUE), .groups = "drop") %>%
  mutate(Site = "LN_Shore") %>%
  group_by(System, Site) %>%
  complete(std_time_rounded = full_seq(std_time_rounded, 0.5), Species = SPECIES_CODES,
           fill = list(Number = 0)) %>%
  ungroup()

ts_LN_open_water_stack <- ts_LN_stack %>%
  filter(System == "LN" & Orientation == "Vertical") %>%
  group_by(System, std_time_rounded, Species) %>%
  summarise(Number = sum(Number, na.rm = TRUE), .groups = "drop") %>%
  mutate(Site = "LN_Open_Water") %>%
  group_by(System, Site) %>%
  complete(std_time_rounded = full_seq(std_time_rounded, 0.5), Species = SPECIES_CODES,
           fill = list(Number = 0)) %>%
  ungroup()

# border/accent color visible against each theme's background (dark plots used a hardcoded
# "white" border - not visible on the white/print theme, so this is now theme-aware)
theme_border_color <- function(theme_fn) if (identical(theme_fn, theme_white)) "black" else "white"

build_panel_fish_per_hour <- function(dat, theme_fn) {
  ggplot(dat, aes(x = std_time_rounded, y = Number, fill = Species)) +
    geom_vline(xintercept = -0.5, linewidth = .5, colour = theme_border_color(theme_fn), lty = 2) +
    geom_vline(xintercept = 12, linewidth = .5, colour = theme_border_color(theme_fn), lty = 2) +
    geom_area(alpha = .8, linewidth = .5, col = theme_border_color(theme_fn)) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    xlim(-4, 20) +
    labs(x = "Time relative to sunset (h)", y = "Number of fish detected") +
    theme_fn(base_size = 28)
}

fish_per_hour_specs <- list(
  PLP           = subset(ts_stack, Site == "PLP"),
  PLS           = subset(ts_stack, Site == "PLS"),
  PL            = subset(ts_stack, Site == "PL"),
  SL            = subset(ts_stack, System == "SL"),
  LN_shore      = ts_LN_shore_stack,
  LN_open_water = ts_LN_open_water_stack,
  LN            = subset(ts_stack, System == "LN"),
  JR            = subset(ts_stack, Site == "JR1")
)


# Species composition donut charts ----------------------------------------

species_prop <- fish_raw |>
  # filter(!Site == "PLS") |>
  group_by(System) |>
  count(Species) |>
  mutate(pct   = n / sum(n) * 100,
         label = sprintf("%s\n%.1f%%", Species, pct))

species_prop_LN <- fish_raw |>
  filter(System == "LN") |>
  group_by(System, Direction, Orientation) |>
  count(Species) |>
  mutate(pct   = n / sum(n) * 100,
         label = sprintf("%s\n%.1f%%", Species, pct))

build_panel_spcomp <- function(dat, theme_fn) {
  ggplot(dat, aes(x = 2, y = n, fill = Species)) +
    geom_col(width = 1, color = theme_border_color(theme_fn)) +
    coord_polar(theta = "y") +
    xlim(0.3, 3.4) +
    # labels placed OUTSIDE the ring (x=2.8, beyond the ring's outer edge at x=2.5) - same
    # position_stack(vjust=0.5) angular placement as before, just pushed further from center
    geom_text(aes(x = 2.8, label = label), position = position_stack(vjust = 0.5),
              size = 10.5, color = theme_border_color(theme_fn)) +
    scale_fill_manual(values = species_palette(theme_fn), guide = "none") +
    labs(caption = paste0("n = ", sum(dat$n))) +
    theme_fn(base_size = 18) +
    theme(plot.caption = element_text(color = theme_border_color(theme_fn), size = 14, hjust = 1),
          axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text = element_blank(), axis.title = element_blank())
}

spcomp_specs <- list(
  PL            = subset(species_prop, System == "PL"),
  SL            = subset(species_prop, System == "SL"),
  LN_shore      = subset(species_prop_LN, System == "LN" & Direction == "Orthogonal"),
  LN_open_water = subset(species_prop_LN, System == "LN" & Orientation == "Vertical"),
  LN            = subset(species_prop, System == "LN"),
  JR            = subset(species_prop, System == "JR")
)


# Movement/position overview panel -----------------------------------------
# Back-to-back day/night stacked histogram of encounters along electrode position, one plot
# per repeat-sampled/paired-logger group - a simple descriptive companion to the timeline
# panels above (NOT the deeper paired ground/surface & shore/open-water statistical comparison
# planned for Fig 3/3d_Paired_Logger_Distribution.R). LNI excluded (separate script).
# PL/JR split into their distinct paired-logger sub-groups (site or Direction); LN split by
# shore vegetation coverage (Complete/None) plus its open-water (Vertical) loggers pooled.
movement_groups <- list(
  PLP = fish_raw |> filter(Site == "PLP"),
  PLS = fish_raw |> filter(Site == "PLS"),
  SLP = fish_raw |> filter(System == "SL"),
  JR_Orthogonal     = fish_raw |> filter(System == "JR", Direction == "Orthogonal"),
  JR_Parallel       = fish_raw |> filter(System == "JR", Direction == "Parallel"),
  LN_shore_Complete = fish_raw |> filter(System == "LN", Direction == "Orthogonal", Veg_cover_extent == "Complete"),
  LN_shore_None     = fish_raw |> filter(System == "LN", Direction == "Orthogonal", Veg_cover_extent == "None"),
  LN_open_water     = fish_raw |> filter(System == "LN", Orientation == "Vertical")
)

build_panel_movement <- function(dat, theme_fn, zero_line_color) {
  hist_dat <- dat |>
    # mean_location is the differential channel-pair center (8 single-ended electrodes -> 7
    # pairs, offset 0.5, range 0.5-6.5); +1 shifts to start electrode numbering at 1, then
    # floor(...)+0.5 snaps to the 7 pair-center bins (1.5-7.5) so each bar spans one full
    # electrode-to-electrode interval (1-2, 2-3, ..., 7-8).
    mutate(loc_bin = floor(mean_location + 1) + 0.5) |>
    count(Photoperiod, loc_bin, Species, name = "n")

  ggplot(hist_dat, aes(x = loc_bin, y = ifelse(Photoperiod == "Night", -n, n), fill = Species)) +
    geom_col(width = 1, alpha = .8) +
    geom_hline(yintercept = 0, linewidth = 0.4, colour = zero_line_color) +
    scale_fill_manual(values = species_palette(theme_fn)) +
    scale_x_continuous(breaks = 1:8, limits = c(0.5, 8.5)) +
    scale_y_continuous(labels = abs) +
    labs(x = "Electrode position", y = "Encounters (day up / night down)") +
    theme_fn(base_size = 28)
}

for (grp_name in names(movement_groups)) {
  panel_dark  <- build_panel_movement(movement_groups[[grp_name]], theme_black, "white")
  panel_white <- build_panel_movement(movement_groups[[grp_name]], theme_white, "black")
  save_panel_pdf(panel_dark,  file.path(OUTPUT_DIR, paste0("fig1_panel_movement_", grp_name, "_dark.pdf")))
  save_panel_pdf(panel_white, file.path(OUTPUT_DIR, paste0("fig1_panel_movement_", grp_name, "_white.pdf")))
}


# Environmental Data Plots ------------------------------------------------

# Temperature -----------------------------------------------------------

build_panel_temp_boxplot <- function(dat, theme_fn) {
  ggplot(dat, aes(x = factor(Photoperiod), y = T_C)) +
    geom_boxplot(fill = "#D55E00", alpha = 1, col = theme_border_color(theme_fn)) +
    ylim(20, 30) +
    labs(x = "Photoperiod", y = "Temperature (\u00b0C)") +
    theme_fn(base_size = 18)
}

temp_boxplot_specs <- list(
  PL = subset(temp_dat, Site == "PLP"),
  LN = subset(temp_dat, System == "LN"),
  JR = subset(temp_dat, System == "JR")
)

build_panel_DO_boxplot <- function(dat, theme_fn) {
  ggplot(dat, aes(x = factor(Photoperiod), y = DO_AS)) +
    geom_boxplot(fill = "dodgerblue", alpha = 1, col = theme_border_color(theme_fn)) +
    ylim(0, 120) +
    labs(x = "Photoperiod", y = "Air Saturation (%)") +
    theme_fn(base_size = 36)
}

DO_boxplot_specs <- list(
  PL = subset(DO_combined, System == "PL"),
  SL = subset(DO_combined, System == "SL"),
  LN = subset(DO_combined, System == "LN"),
  JR = subset(DO_combined, System == "JR")
)


# Save all plots as dark/white pairs, then export as pdfs ----------------
# doubled size (was 8x6) since labels were overlapping at the smaller size, plus a PNG copy
# of each into a "png" subfolder for quick use in slides - matches save_panel_pdf()'s
# convention used by every downstream RQ script.

panel_groups <- list(
  fish_per_hour = list(build_fn = build_panel_fish_per_hour, specs = fish_per_hour_specs),
  spcomp        = list(build_fn = build_panel_spcomp,        specs = spcomp_specs),
  temp_boxplot  = list(build_fn = build_panel_temp_boxplot,  specs = temp_boxplot_specs),
  DO_boxplot    = list(build_fn = build_panel_DO_boxplot,    specs = DO_boxplot_specs)
)

all_panels_dark  <- list()
all_panels_white <- list()

for (group_name in names(panel_groups)) {
  build_fn <- panel_groups[[group_name]]$build_fn
  specs    <- panel_groups[[group_name]]$specs
  for (spec_name in names(specs)) {
    dat <- specs[[spec_name]]
    panel_dark  <- build_fn(dat, theme_black)
    panel_white <- build_fn(dat, theme_white)
    # spcomp's "n = " annotation is baked-in data, not a removable caption - keep it
    save_fn <- if (group_name == "spcomp") save_panel_pdf_with_caption else save_panel_pdf
    save_fn(panel_dark,  file.path(OUTPUT_DIR, paste0(spec_name, "_", group_name, "_dark.pdf")))
    save_fn(panel_white, file.path(OUTPUT_DIR, paste0(spec_name, "_", group_name, "_white.pdf")))
    all_panels_dark[[paste0(spec_name, "_", group_name)]]  <- panel_dark
    all_panels_white[[paste0(spec_name, "_", group_name)]] <- panel_white
  }
}

ggexport(plotlist = all_panels_dark,  filename = file.path(OUTPUT_DIR, "all_plots_dark.pdf"))
ggexport(plotlist = all_panels_white, filename = file.path(OUTPUT_DIR, "all_plots_white.pdf"))

