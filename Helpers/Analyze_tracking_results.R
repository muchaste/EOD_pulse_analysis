# Analyze eod logger tracking results and combine with environmental data

library(ggpubr)
library(tidyverse)
library(lubridate)
library(sf)
library(openxlsx)
library(scales)
library(ggpubr)
library(readxl)
library(suncalc)

rm(list=ls())


# ---- Set Directories ---------------------------------------------------------
TRACK_DIR         <- "E:/Tracking_Summaries/"              # root containing tracking summaries (output from python script 04_1)
# ENV_DATA_FILE     <- "E:/Environmental_data/Complete_env_data_summary_per_day.csv"   # semicolon-separated
ENV_DATA_DIR      <- "E:/Environmental_data/"   # directory containing environmental data files
SAMPLING_REGISTER <- "E:/Recordings and Measurements.xlsx"
OUTPUT_DIR        <- "E:/outputs/figures"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Plot Settings ------------------------------

SPECIES_PALETTE <- c(
  "GL" = "#CC79A7",
  "PN" = "#56B4E9",
  "PD" = "#009E73",
  "MV" = "#E69F00",
  "Unknown" = "#999999"
)


# Colorblind-friendly palette (Wong 2011), reordered for contrast on black
# Order: yellow, sky blue, orange, bluish green, vermillion, blue, reddish purple
cbf_palette <- c(
  "#F0E442", "#56B4E9", "#E69F00", "#009E73",
  "#D55E00", "#0072B2", "#CC79A7"
)

theme_black <- function(base_size = 12, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # backgrounds
      plot.background  = element_rect(fill = "black", color = NA),
      panel.background = element_rect(fill = "black", color = NA),
      legend.background = element_rect(fill = "black", color = NA),
      legend.key        = element_rect(fill = "black", color = NA),
      strip.background  = element_rect(fill = "black", color = "white", linewidth = 0.5),
      
      # axes
      axis.line  = element_line(color = "white", linewidth = 0.5),
      axis.ticks = element_line(color = "white", linewidth = 0.4),
      
      # text
      plot.title    = element_text(color = "white", face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "white", hjust = 0),
      plot.caption  = element_text(color = "grey70", hjust = 1),
      axis.title    = element_text(color = "white"),
      axis.text     = element_text(color = "white"),
      legend.title  = element_text(color = "white"),
      legend.text   = element_text(color = "white"),
      strip.text    = element_text(color = "white", face = "bold"),
      
      # grid (off by default as in theme_classic, but white if enabled)
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}


# ---- Site coordinates and photoperiod ----------------------------------------
# UTM zone 36 South (EPSG:32736): E 377288.48, N 9960402.35
coords_utm <- st_sfc(st_point(c(377288.48, 9960402.35)), crs = 32736)
coords_wgs <- st_transform(coords_utm, crs = 4326)
SITE_LON <- st_coordinates(coords_wgs)[1, "X"]
SITE_LAT <- st_coordinates(coords_wgs)[1, "Y"]

SUNRISE_FALLBACK_H <- 6.5    # 06:30 UTC
SUNSET_FALLBACK_H  <- 18.617 # 18:37 UTC


# Annotate diel night periods on time-of-day plots
night_ribbons <- function(alpha = 0.12) {
  sr <- photo_summary$sunrise_hour
  ss <- photo_summary$sunset_hour
  list(
    annotate("rect", xmin = sr,  xmax = ss, ymin = -Inf, ymax = Inf,
             fill = "darkgrey", alpha = alpha),
    annotate("rect", xmin = sr+0.5, xmax = ss - 0.5, ymin = -Inf, ymax = Inf,
             fill = "lightgrey", alpha = alpha)
  )
}


# 2. DATA LOADING ------------------------------------------------------------

# ---- Sampling register ------------------------------------------------------
# Timezone was encoded in Excel file in UTC, so the time has shifted by +3 hours when read in R. Correct by subtracting 3 hours to get local time (Africa/Nairobi).

register_raw <- read_xlsx(SAMPLING_REGISTER, sheet = "Recordings_overview") |>
  mutate(
    t_first_rec_start  = as.POSIXct(t_first_rec_start, tz = "Africa/Nairobi") - hours(3),
    t_last_rec_end     = as.POSIXct(t_last_rec_end,    tz = "Africa/Nairobi") - hours(3),
    Session_duration_h = as.numeric(difftime(t_last_rec_end, t_first_rec_start,
                                             units = "hours")),
    Session_date       = as.Date(Start_Date),
    Session_date_str   = format(as.Date(t_first_rec_start), "%Y%m%d"),
  )

# Remove rows where t_rec is NA (no recordings were made)
register_raw <- register_raw |> filter(!is.na(t_first_rec_start))

# add sunrise/sunset columns to register for each session based on the session date and site coordinates
suntimes <- getSunlightTimes(
    date = as.Date(register_raw$Session_date, tz = "Africa/Nairobi"),
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

suntimes_next <- getSunlightTimes(
    date = as.Date(register_raw$Session_date, tz = "Africa/Nairobi") + 1,
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

register_raw$Sunrise <- suntimes$sunrise
register_raw$Sunset <- suntimes$sunset
register_raw$Sunrise_next <- suntimes_next$sunrise
register_raw$Sunset_next <- suntimes_next$sunset

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

suntimes_env <- getSunlightTimes(
    date = as.Date(light_temp$Datetime, tz = "Africa/Nairobi"),
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

suntimes_next <- getSunlightTimes(
    date = as.Date(light_temp$Datetime, tz = "Africa/Nairobi") + 1,
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

light_temp$Sunrise <- suntimes_env$sunrise
light_temp$Sunset <- suntimes_env$sunset
light_temp$Sunrise_next <- suntimes_next$sunrise
light_temp$Sunset_next <- suntimes_next$sunset
light_temp$std_time_ss <- as.numeric(difftime(light_temp$Datetime, light_temp$Sunset, units = "hours"))

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

suntimes_DO <- getSunlightTimes(
    date = as.Date(DO_combined$Datetime),
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

suntimes_DO_next <- getSunlightTimes(
    date = as.Date(DO_combined$Datetime) + 1,
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

DO_combined$Sunrise <- suntimes_DO$sunrise
DO_combined$Sunset <- suntimes_DO$sunset
DO_combined$Sunrise_next <- suntimes_DO_next$sunrise
DO_combined$Sunset_next <- suntimes_DO_next$sunset
DO_combined$std_time_ss <- as.numeric(difftime(DO_combined$Datetime, DO_combined$Sunset, units = "hours"))


# Manual Measurements
# setwd(datadir)
MM_dat <- read.xlsx(SAMPLING_REGISTER, sheet = "Manual_measurements")
MM_dat$Date <- convertToDate(MM_dat$Date)
MM_dat$Datetime <- convertToDateTime(MM_dat$Time)
date(MM_dat$Datetime) <- MM_dat$Date

suntimes_MM <- getSunlightTimes(
    date = as.Date(MM_dat$Datetime, tz = "Africa/Nairobi"),
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

suntimes_MM_next <- getSunlightTimes(
    date = as.Date(MM_dat$Datetime, tz = "Africa/Nairobi") + 1,
    lat  = SITE_LAT,
    lon  = SITE_LON,
    keep = c("sunrise", "sunset"),
    tz   = "Africa/Nairobi"
  )

MM_dat$Sunrise <- suntimes_MM$sunrise
MM_dat$Sunset <- suntimes_MM$sunset
MM_dat$Sunrise_next <- suntimes_MM_next$sunrise
MM_dat$Sunset_next <- suntimes_MM_next$sunset
MM_dat$std_time_ss <- as.numeric(difftime(MM_dat$Datetime, MM_dat$Sunset, units = "hours"))

# tracking data -----------------------------------------------------------
fish_track_list <- list.files(TRACK_DIR, pattern = "tracked_fish_summary\\.csv",
                              recursive = TRUE, full.names = TRUE)
fish_raw <- NULL
for(ft_file in fish_track_list){
  ft_dat <- read.csv2(ft_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  ft_dat$Session_date_str <- strsplit(basename(ft_file), "_")[[1]][2]
  ft_dat$Session_date <-  as.Date(ft_dat$Session_date_str, format="%Y%m%d")
  ft_dat$Logger_id <- strsplit(basename(ft_file), "_")[[1]][1]
  ft_dat$Site <- register_raw$Site[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Site_logger <- paste(ft_dat$Site[1], ft_dat$Logger_id, sep="_")
  ft_dat$System <- register_raw$System[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Placement <- register_raw$Placement[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Sunrise <- register_raw$Sunrise[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Sunset <- register_raw$Sunset[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Sunrise_next <- register_raw$Sunrise_next[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  ft_dat$Sunset_next <- register_raw$Sunset_next[register_raw$Session_date == ft_dat$Session_date[1] & register_raw$Logger_ID == ft_dat$Logger_id[1]]
  fish_raw <- bind_rows(fish_raw, ft_dat)
}
fish_raw$entry_time <- as.POSIXct(fish_raw$entry_time, tz = "Africa/Nairobi")
fish_raw$exit_time <- as.POSIXct(fish_raw$exit_time, tz = "Africa/Nairobi")
fish_raw$visit_midpoint <- fish_raw$entry_time + (fish_raw$exit_time - fish_raw$entry_time)/2
fish_raw$visit_duration_min <- as.numeric(difftime(fish_raw$exit_time, fish_raw$entry_time, units = "mins"))
fish_raw$std_time_ss <- as.numeric(difftime(fish_raw$visit_midpoint, fish_raw$Sunset, units = "hours"))
fish_raw$std_time_sr <- as.numeric(difftime(fish_raw$visit_midpoint, fish_raw$Sunrise_next, units = "hours"))
fish_raw$Photoperiod <- ifelse(fish_raw$std_time_ss >= 0 & fish_raw$std_time_sr <= 0, "Night", "Day")

# join with environmental data
fish_dat <- fish_raw |>
  left_join(env_clean |> select(System, Site, Session_date,
                                 T_best, T_best_sd, DO_best, DO_best_sd,
                                 pH_mean, Cond_mean, Turbidity_mean, Lux_mean),
            by = c("System", "Site",
                   "Session_date")) 

event_list <- list.files(TRACK_DIR, pattern = "tracked_event_summary\\.csv",
                         recursive = TRUE, full.names = TRUE)

events_raw <- NULL
for(ev_file in event_list){
  # read ev_dat and set channels_used column to character to avoid type mismatch when binding rows
  ev_dat <- read.csv2(ev_file, sep = ",", dec = ".", stringsAsFactors = FALSE) 
  ev_dat$channels_used <- as.character(ev_dat$channels_used)
  ev_dat$Session_date_str <- strsplit(basename(ev_file), "_")[[1]][2]
  ev_dat$Session_date <-  as.Date(ev_dat$Session_date_str, format="%Y%m%d")
  ev_dat$Logger_id <- strsplit(basename(ev_file), "_")[[1]][1]
  ev_dat$Site <- register_raw$Site[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Site_logger <- paste(ev_dat$Site[1], ev_dat$Logger_id, sep="_")
  ev_dat$System <- register_raw$System[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Placement <- register_raw$Placement[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Sunrise <- register_raw$Sunrise[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Sunset <- register_raw$Sunset[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Sunrise_next <- register_raw$Sunrise_next[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  ev_dat$Sunset_next <- register_raw$Sunset_next[register_raw$Session_date == ev_dat$Session_date[1] & register_raw$Logger_ID == ev_dat$Logger_id[1]]
  events_raw <- bind_rows(events_raw, ev_dat)
}
events_raw$eod_start_time <- as.POSIXct(events_raw$eod_start_time, tz = "Africa/Nairobi")
events_raw$eod_end_time <- as.POSIXct(events_raw$eod_end_time, tz = "Africa/Nairobi")
events_raw$event_start_time <- as.POSIXct(events_raw$event_start_time, tz = "Africa/Nairobi")
events_raw$event_end_time <- as.POSIXct(events_raw$event_end_time, tz = "Africa/Nairobi")
events_raw$std_time_ss <- as.numeric(difftime(events_raw$eod_start_time, events_raw$Sunset, units = "hours"))
events_raw$std_time_sr <- as.numeric(difftime(events_raw$eod_start_time, events_raw$Sunrise_next, units = "hours"))
events_raw$Photoperiod <- ifelse(events_raw$std_time_ss >= 0 & events_raw$std_time_sr <= 0, "Night", "Day")

species_list <- list.files(TRACK_DIR, pattern = "tracked_species_summary\\.csv",
                           recursive = TRUE, full.names = TRUE)

species_raw <- NULL
for(sp_file in species_list){
  sp_dat <- read.csv2(sp_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  sp_dat$Session_date_str <- strsplit(basename(sp_file), "_")[[1]][2]
  sp_dat$Session_date <-  as.Date(sp_dat$Session_date_str, format="%Y%m%d")
  sp_dat$Logger_id <- strsplit(basename(sp_file), "_")[[1]][1]
  sp_dat$Site <- register_raw$Site[register_raw$Session_date == sp_dat$Session_date[1] & register_raw$Logger_ID == sp_dat$Logger_id[1]]
  sp_dat$Site_logger <- paste(sp_dat$Site[1], sp_dat$Logger_id, sep="_")
  sp_dat$System <- register_raw$System[register_raw$Session_date == sp_dat$Session_date[1] & register_raw$Logger_ID == sp_dat$Logger_id[1]]
  sp_dat$Placement <- register_raw$Placement[register_raw$Session_date == sp_dat$Session_date[1] & register_raw$Logger_ID == sp_dat$Logger_id[1]]
  species_raw <- bind_rows(species_raw, sp_dat)
}

ts_list <- list.files(TRACK_DIR, pattern = "session_fish_timeseries\\.csv",
                      recursive = TRUE, full.names = TRUE)

ts_raw <- NULL
for(ts_file in ts_list){
  ts_dat <- read.csv2(ts_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  ts_dat$Session_date_str <- strsplit(basename(ts_file), "_")[[1]][2]
  ts_dat$Session_date <-  as.Date(ts_dat$Session_date_str, format="%Y%m%d")
  ts_dat$Logger_id <- strsplit(basename(ts_file), "_")[[1]][1]
  ts_dat$Site <- register_raw$Site[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_dat$Site_logger <- paste(ts_dat$Site[1], ts_dat$Logger_id, sep="_")
  ts_dat$System <- register_raw$System[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_dat$Placement <- register_raw$Placement[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  # Summarize number of fish over all columns that start with "GL", "PD", etc. (species codes)
  ts_dat$n_GL <- rowSums(ts_dat[, grepl("^GL", names(ts_dat))], na.rm = TRUE)
  ts_dat$n_PD <- rowSums(ts_dat[, grepl("^PD", names(ts_dat))], na.rm = TRUE)
  ts_dat$n_MV <- rowSums(ts_dat[, grepl("^MV", names(ts_dat))], na.rm = TRUE)
  ts_dat$n_PN <- rowSums(ts_dat[, grepl("^PN", names(ts_dat))], na.rm = TRUE)
  ts_dat$Sunrise <- register_raw$Sunrise[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_dat$Sunset <- register_raw$Sunset[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_dat$Sunrise_next <- register_raw$Sunrise_next[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_dat$Sunset_next <- register_raw$Sunset_next[register_raw$Session_date == ts_dat$Session_date[1] & register_raw$Logger_ID == ts_dat$Logger_id[1]]
  ts_raw <- bind_rows(ts_raw, ts_dat)
}
ts_raw$datetime <- as.POSIXct(ts_raw$datetime, tz = "Africa/Nairobi")
ts_raw$std_time_ss <- as.numeric(difftime(ts_raw$datetime, ts_raw$Sunset, units = "hours"))
ts_raw$std_time_sr <- as.numeric(difftime(ts_raw$datetime, ts_raw$Sunrise_next, units = "hours"))
ts_raw$Photoperiod <- ifelse(ts_raw$std_time_ss >= 0 & ts_raw$std_time_sr <= 0, "Night", "Day")

# ---- Recompute time series from fish_raw, excluding species-uncertain detections -----------
# ts_raw is built per-channel and does not account for species assignment uncertainty (unlike
# fish_raw). Rebuild a presence time series directly from entry_time/exit_time of each fish,
# ignoring channel identity, after filtering out uncertain species assignments.

TS_BIN_MIN <- 1  # time bin resolution (minutes) for the recomputed presence time series

fish_filt <- fish_raw |> filter(species_uncertain == "False")

ts_expanded <- fish_filt |>
  rowwise() |>
  mutate(datetime = list(seq(floor_date(entry_time, unit = paste(TS_BIN_MIN, "minutes")),
                              exit_time,
                              by = paste(TS_BIN_MIN, "min")))) |>
  ungroup() |>
  select(fish_id, species_assigned, Site, System, Placement, Session_date, Logger_id,
         Sunrise, Sunset, Sunrise_next, Sunset_next, datetime) |>
  unnest(datetime)

ts_new <- ts_expanded |>
  count(Site, System, Placement, Session_date, Logger_id,
        Sunrise, Sunset, Sunrise_next, Sunset_next,
        datetime, species_assigned, name = "Number") |>
  pivot_wider(names_from = species_assigned, values_from = Number, values_fill = 0,
              names_prefix = "n_")

for (sp_col in c("n_GL", "n_PD", "n_MV", "n_PN")) {
  if (!sp_col %in% names(ts_new)) ts_new[[sp_col]] <- 0
}

ts_new$std_time_ss <- as.numeric(difftime(ts_new$datetime, ts_new$Sunset, units = "hours"))
ts_new$std_time_sr <- as.numeric(difftime(ts_new$datetime, ts_new$Sunrise_next, units = "hours"))
ts_new$Photoperiod <- ifelse(ts_new$std_time_ss >= 0 & ts_new$std_time_sr <= 0, "Night", "Day")


# 3. PLOTS -----------------------------------------------

# Time Series -------------------------------------------------------------

# summarize ts dataset per Site and per 30 min interval
# compute sum, mean, sd, and 95% CI of number of fish per species per 30 min interval
# 30 min interval = 0.5 hours, so we can round std_time to the nearest 0.5
ts_new$std_time_rounded <- round(ts_new$std_time_ss * 2) / 2

# Special dataset for LN to separate open water and shore loggers
ts_LN_stack <- ts_new %>%
  filter(System == "LN") |> 
  group_by(Site, System, Placement, std_time_rounded) %>%
  summarise(
    n_GL = sum(n_GL, na.rm = TRUE),
    n_PD = sum(n_PD, na.rm = TRUE),
    n_MV = sum(n_MV, na.rm = TRUE),
    n_PN = sum(n_PN, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = starts_with("n_"), names_to = "Species", values_to = "Number") %>%
  mutate(Species = str_remove(Species, "^n_"))

ts_stack <- ts_new %>%
  filter(System != "LN") |> 
  group_by(Site, System, std_time_rounded) %>%
  summarise(
    n_GL = sum(n_GL, na.rm = TRUE),
    n_PD = sum(n_PD, na.rm = TRUE),
    n_MV = sum(n_MV, na.rm = TRUE),
    n_PN = sum(n_PN, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = starts_with("n_"), names_to = "Species", values_to = "Number") %>%
  mutate(Species = str_remove(Species, "^n_"))

# Special subset for LN data because it has different sites and we want to combine the shore sites and the open water sites separately for overview plots
ts_LN_shore_stack <- ts_LN_stack %>%
  filter(System == "LN" & Placement == "Orthogonal") %>%
  group_by(System, std_time_rounded, Species) %>%
  summarise(Number = sum(Number, na.rm = TRUE), .groups = "drop") %>%
  mutate(Site = "LN_Shore")

ts_LN_open_water_stack <- ts_LN_stack %>%
  filter(System == "LN" & Placement == "Point") %>%
  group_by(System, std_time_rounded, Species) %>%
  summarise(Number = sum(Number, na.rm = TRUE), .groups = "drop") %>%
  mutate(Site = "LN_Open_Water")

PLP_fish_per_hour <- ggplot(subset(ts_stack, Site == "PLP"), aes(x = std_time_rounded, y = Number, fill = Species))+
  geom_vline(xintercept = -0.5, linewidth = .5, colour = "white", lty = 2)+
  geom_vline(xintercept = 12, linewidth = .5, colour = "white", lty = 2)+
  geom_area(alpha = .8, linewidth = .5, col = "white")+
  scale_fill_manual(values = SPECIES_PALETTE)+
  xlim(-4, 20)+
  labs(x = "Time relative to sunset (h)", y = "Number of fish detected")+
  theme_black(base_size = 16)

PLS_fish_per_hour <- ggplot(subset(ts_stack, Site == "PLS"), aes(x = std_time_rounded, y = Number, fill = Species))+
  geom_vline(xintercept = -0.5, linewidth = .5, colour = "white", lty = 2)+
  geom_vline(xintercept = 12, linewidth = .5, colour = "white", lty = 2)+
  geom_area(alpha = .8, linewidth = .5, col = "white")+
  scale_fill_manual(values = SPECIES_PALETTE)+
  xlim(-4, 20)+
  labs(x = "Time relative to sunset (h)", y = "Number of fish detected")+
  theme_black(base_size = 16)

LN_shore_fish_per_hour <- ggplot(ts_LN_shore_stack, aes(x = std_time_rounded, y = Number, fill = Species))+
  geom_vline(xintercept = -0.5, linewidth = .5, colour = "white", lty = 2)+
  geom_vline(xintercept = 12, linewidth = .5, colour = "white", lty = 2)+
  geom_area(alpha = .8, linewidth = .5, col = "white")+
  scale_fill_manual(values = SPECIES_PALETTE)+
  xlim(-4, 20)+
  labs(x = "Time relative to sunset (h)", y = "Number of fish detected")+
  theme_black(base_size = 16)

LN_open_water_fish_per_hour <- ggplot(ts_LN_open_water_stack, aes(x = std_time_rounded, y = Number, fill = Species))+
  geom_vline(xintercept = -0.5, linewidth = .5, colour = "white", lty = 2)+
  geom_vline(xintercept = 12, linewidth = .5, colour = "white", lty = 2)+
  geom_area(alpha = .8, linewidth = .5, col = "white")+
  scale_fill_manual(values = SPECIES_PALETTE)+
  xlim(-4, 20)+
  labs(x = "Time relative to sunset (h)", y = "Number of fish detected")+
  theme_black(base_size = 16)

JR_fish_per_hour <- ggplot(subset(ts_stack, Site == "JR1"), aes(x = std_time_rounded, y = Number, fill = Species))+
  geom_vline(xintercept = -0.5, linewidth = .5, colour = "white", lty = 2)+
  geom_vline(xintercept = 12, linewidth = .5, colour = "white", lty = 2)+
  geom_area(alpha = .8, linewidth = .5, col = "white")+
  scale_fill_manual(values = SPECIES_PALETTE)+
  xlim(-4, 20)+
  labs(x = "Time relative to sunset (h)", y = "Number of fish detected")+
  theme_black(base_size = 16)


# Boxplots ----------------------------------------------------------------
# Boxplots per photoperiod (day/night) for each species and site, showing number of fish and average location
PLP_location_boxplot <- ggplot(subset(fish_raw, Site == "PLP" & species_uncertain == "False"), aes(x = Photoperiod, y = mean_location))+
  geom_violin(aes(fill = Photoperiod), alpha = .8, scale = "width") +
  # geom_jitter(aes(col = Photoperiod), width = 0.15, size = 0.7, alpha = 0.25) +
  stat_summary(aes(col = Photoperiod), fun = median, geom = "crossbar",
               width = 0.5, linewidth = 0.5) + #color = "white", 
  scale_fill_manual(values = c("white", "darkgrey"))+
  scale_color_manual(values = c("black", "lightgrey"))+
  labs(x = "Photoperiod", y = "Fish location (electrode units)")+
  theme_black(base_size = 16)+
  theme(legend.position = "none")

PLS_location_boxplot <- ggplot(subset(fish_raw, Site == "PLS" & species_uncertain == "False"), aes(x = Photoperiod, y = mean_location))+
  geom_violin(aes(fill = Photoperiod), alpha = .8, scale = "width") +
  # geom_jitter(aes(col = Photoperiod), width = 0.15, size = 0.7, alpha = 0.25) +
  stat_summary(aes(col = Photoperiod), fun = median, geom = "crossbar",
               width = 0.5, linewidth = 0.5) + #color = "white", 
  scale_fill_manual(values = c("white", "darkgrey"))+
  scale_color_manual(values = c("black", "lightgrey"))+
  # scale_fill_manual(values = SPECIES_PALETTE)+
  labs(x = "Photoperiod", y = "Fish location (electrode units)")+
  theme_black(base_size = 16)+
  theme(legend.position = "none")

LN_shore_location_boxplot <- ggplot(subset(fish_raw, System == "LN" & Placement == "Orthogonal" & species_uncertain == "False" ), aes(x = Photoperiod, y = mean_location))+
  geom_violin(aes(fill = Photoperiod), alpha = .8, scale = "width") +
  # geom_jitter(aes(col = Photoperiod), width = 0.15, size = 0.7, alpha = 0.25) +
  stat_summary(aes(col = Photoperiod), fun = median, geom = "crossbar",
               width = 0.5, linewidth = 0.5) + #color = "white", 
  scale_fill_manual(values = c("white", "darkgrey"))+
  scale_color_manual(values = c("black", "lightgrey"))+
  labs(x = "Photoperiod", y = "Fish location (electrode units)")+
  theme_black(base_size = 16)+
  theme(legend.position = "none")

LN_open_water_location_boxplot <- ggplot(subset(fish_raw, System == "LN" & Placement == "Point" & species_uncertain == "False" ), aes(x = Photoperiod, y = mean_location))+
  geom_violin(aes(fill = Photoperiod), alpha = .8, scale = "width") +
  # geom_jitter(aes(col = Photoperiod), width = 0.15, size = 0.7, alpha = 0.25) +
  stat_summary(aes(col = Photoperiod), fun = median, geom = "crossbar",
               width = 0.5, linewidth = 0.5) + #color = "white", 
  scale_fill_manual(values = c("white", "darkgrey"))+
  scale_color_manual(values = c("black", "lightgrey"))+
  labs(x = "Photoperiod", y = "Fish location (electrode units)")+
  theme_black(base_size = 16)+
  theme(legend.position = "none")

JR_location_boxplot <- ggplot(subset(fish_raw, Site == "JR1" & species_uncertain == "False"), aes(x = Photoperiod, y = mean_location))+
  geom_violin(aes(fill = Photoperiod), alpha = .8, scale = "width") +
  # geom_jitter(aes(col = Photoperiod), width = 0.15, size = 0.7, alpha = 0.25) +
  stat_summary(aes(col = Photoperiod), fun = median, geom = "crossbar",
               width = 0.5, linewidth = 0.5) + #color = "white", 
  scale_fill_manual(values = c("white", "darkgrey"))+
  scale_color_manual(values = c("black", "lightgrey"))+
  labs(x = "Photoperiod", y = "Fish location (electrode units)")+
  theme_black(base_size = 16)+
  theme(legend.position = "none")



# Species composition donut charts ----------------------------------------

species_prop <- fish_raw |>
  filter(!Site == "PLS") |>
  filter(!species_uncertain=="True") |>
  group_by(System) |>
  count(species_assigned) |>
  mutate(pct   = n / sum(n) * 100,
         label = sprintf("%s\n%.1f%%", species_assigned, pct))

species_prop_LN <- fish_raw |>
  filter(System == "LN") |>
  filter(!species_uncertain=="True") |>
  group_by(System, Placement) |>
  count(species_assigned) |>
  mutate(pct   = n / sum(n) * 100,
         label = sprintf("%s\n%.1f%%", species_assigned, pct))

PL_spcomp <- ggplot(subset(species_prop, System == "PL"), aes(x = 2, y = n, fill = species_assigned)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5), size = 3.5) +
  scale_fill_manual(values = SPECIES_PALETTE, guide = "none") +
  theme_black(base_size = 18)

LN_shore_spcomp <- ggplot(subset(species_prop_LN, System == "LN" & Placement == "Orthogonal"), aes(x = 2, y = n, fill = species_assigned)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5), size = 3.5) +
  scale_fill_manual(values = SPECIES_PALETTE, guide = "none") +
  theme_black(base_size = 18)

LN_open_water_spcomp <- ggplot(subset(species_prop_LN, System == "LN" & Placement == "Point"), aes(x = 2, y = n, fill = species_assigned)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5), size = 3.5) +
  scale_fill_manual(values = SPECIES_PALETTE, guide = "none") +
  theme_black(base_size = 18)

JR_spcomp <- ggplot(subset(species_prop, System == "JR"), aes(x = 2, y = n, fill = species_assigned)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5), size = 3.5) +
  scale_fill_manual(values = SPECIES_PALETTE, guide = "none") +
  theme_black(base_size = 18)


# EOD width ---------------------------------------------------------------

temp_scale <- 20  # multiplier to map temperature onto the primary axis range

JR_width_plot <- ggplot(
  subset(fish_dat, System == "JR" & species_uncertain == "False"), aes(x = visit_midpoint, y = mean_width_us)) +
  geom_point(aes(color = species_assigned), alpha = 0.4) +
  stat_summary(
    aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
  scale_y_continuous(name = "Mean EOD width (µs)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (°C)")) +
  scale_x_datetime(name = "Date") +
  scale_color_manual(values = SPECIES_PALETTE) +
  theme_black(base_size = 18) +
  theme(axis.title.y.right = element_text(color = "#D55E00"),
        axis.text.y.right  = element_text(color = "#D55E00"))

temp_scale <- 4  # multiplier to map temperature onto the primary axis range

PLP_width_plot <- ggplot(
  subset(fish_dat, Site == "PLP" & species_uncertain == "False"), aes(x = visit_midpoint, y = mean_width_us)) +
  geom_point(aes(color = species_assigned), alpha = 0.4) +
  stat_summary(
    aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
  scale_y_continuous(name = "Mean EOD width (µs)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (°C)")) +
  scale_x_datetime(name = "Date") +
  scale_color_manual(values = SPECIES_PALETTE) +
  theme_black(base_size = 18) +
  theme(axis.title.y.right = element_text(color = "#D55E00"),
        axis.text.y.right  = element_text(color = "#D55E00"))

temp_scale <- 2  # multiplier to map temperature onto the primary axis range

PLS_width_plot <- ggplot(
  subset(fish_dat, Site == "PLS" & species_uncertain == "False"), aes(x = visit_midpoint, y = mean_width_us)) +
  geom_point(aes(color = species_assigned), alpha = 0.4) +
  stat_summary(
    aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
  scale_y_continuous(name = "Mean EOD width (µs)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (°C)")) +
  scale_x_datetime(name = "Date") +
  scale_color_manual(values = SPECIES_PALETTE) +
  theme_black(base_size = 18) +
  theme(axis.title.y.right = element_text(color = "#D55E00"),
        axis.text.y.right  = element_text(color = "#D55E00"))

LN_shore_width_plot <- ggplot(
  subset(fish_dat, System == "LN" & Placement == "Orthogonal" & species_uncertain == "False"), aes(x = visit_midpoint, y = mean_width_us)) +
  geom_point(aes(color = species_assigned), alpha = 0.4) +
  stat_summary(
    aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
  scale_y_continuous(name = "Mean EOD width (µs)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (°C)")) +
  scale_x_datetime(name = "Date") +
  scale_color_manual(values = SPECIES_PALETTE) +
  theme_black(base_size = 18) +
  theme(axis.title.y.right = element_text(color = "#D55E00"),
        axis.text.y.right  = element_text(color = "#D55E00"))

LN_open_water_width_plot <- ggplot(
  subset(fish_dat, System == "LN" & Placement == "Point" & species_uncertain == "False"), aes(x = visit_midpoint, y = mean_width_us)) +
  geom_point(aes(color = species_assigned), alpha = 0.4) +
  stat_summary(
    aes(y = T_best * temp_scale), fun = mean, geom = "line", linewidth = 0.7, color = "#D55E00") +
  scale_y_continuous(name = "Mean EOD width (µs)", sec.axis = sec_axis(transform = ~ . / temp_scale, name = "Temperature (°C)")) +
  scale_x_datetime(name = "Date") +
  scale_color_manual(values = SPECIES_PALETTE) +
  theme_black(base_size = 18) +
  theme(axis.title.y.right = element_text(color = "#D55E00"),
        axis.text.y.right  = element_text(color = "#D55E00"))


# Environmental Data Plots ------------------------------------------------

# Temperature -----------------------------------------------------------

PL_temp_boxplot <- ggplot(subset(temp_dat, Site == "PLP"), aes(x = factor(Photoperiod), y = T_C)) +
  # geom_jitter(width = 0.15, size = 0.7, alpha = 0.25, col = "white") +
  geom_boxplot(fill = "#D55E00", alpha = 1, col = "white") +
  ylim(20, 30)+
  labs(x = "Photoperiod", y = "Temperature (°C)") +
  theme_black(base_size = 18)

LN_temp_boxplot <- ggplot(subset(temp_dat, System == "LN"), aes(x = factor(Photoperiod), y = T_C)) +
  geom_boxplot(fill = "#D55E00", alpha = 1, col = "white") +
  ylim(20, 30)+
  labs(x = "Photoperiod", y = "Temperature (°C)") +
  theme_black(base_size = 18)

JR_temp_boxplot <- ggplot(subset(temp_dat, System == "JR"), aes(x = factor(Photoperiod), y = T_C)) +
  geom_boxplot(fill = "#D55E00", alpha = 1, col = "white") +
  ylim(20, 30)+
  labs(x = "Photoperiod", y = "Temperature (°C)") +
  theme_black(base_size = 18)

PL_DO_boxplot <- ggplot(subset(DO_combined, Site == "PLP"), aes(x = factor(Photoperiod), y = DO_AS)) +
  geom_boxplot(fill = "dodgerblue", alpha = 1, col = "white") +
  ylim(0, 120)+
  labs(x = "Photoperiod", y = "Air Saturation (%)") +
  theme_black(base_size = 18)

LN_DO_boxplot <- ggplot(subset(DO_combined, System == "LN"), aes(x = factor(Photoperiod), y = DO_AS)) +
  geom_boxplot(fill = "dodgerblue", alpha = 1, col = "white") +
  ylim(0, 120)+
  labs(x = "Photoperiod", y = "Air Saturation (%)") +
  theme_black(base_size = 18)

JR_DO_boxplot <- ggplot(subset(DO_combined, System == "JR"), aes(x = factor(Photoperiod), y = DO_AS)) +
  geom_boxplot(fill = "dodgerblue", alpha = 1, col = "white") +
  ylim(0, 120)+
  labs(x = "Photoperiod", y = "Air Saturation (%)") +
  theme_black(base_size = 18)


# Save all plots in a list for easy export, then export as pdfs ----------------

plot_list <- list(
  PLP_fish_per_hour = PLP_fish_per_hour,
  PLS_fish_per_hour = PLS_fish_per_hour,
  LN_shore_fish_per_hour = LN_shore_fish_per_hour,
  LN_open_water_fish_per_hour = LN_open_water_fish_per_hour,
  JR_fish_per_hour = JR_fish_per_hour,
  PLP_location_boxplot = PLP_location_boxplot,
  PLS_location_boxplot = PLS_location_boxplot,
  LN_shore_location_boxplot = LN_shore_location_boxplot,
  LN_open_water_location_boxplot = LN_open_water_location_boxplot,
  JR_location_boxplot = JR_location_boxplot,
  PL_spcomp = PL_spcomp,
  LN_shore_spcomp = LN_shore_spcomp,
  LN_open_water_spcomp = LN_open_water_spcomp,
  JR_spcomp = JR_spcomp,
  JR_width_plot = JR_width_plot,
  PLP_width_plot = PLP_width_plot,
  PLS_width_plot = PLS_width_plot,
  LN_shore_width_plot = LN_shore_width_plot,
  LN_open_water_width_plot = LN_open_water_width_plot,
  PL_temp_boxplot = PL_temp_boxplot,
  LN_temp_boxplot = LN_temp_boxplot,
  JR_temp_boxplot = JR_temp_boxplot,
  PL_DO_boxplot = PL_DO_boxplot,
  LN_DO_boxplot = LN_DO_boxplot,
  JR_DO_boxplot = JR_DO_boxplot
)

# export using ggexport
for (i in seq_along(plot_list)) {
  plot_name <- names(plot_list)[i]
  plot_file <- paste0(OUTPUT_DIR, "/", plot_name, ".pdf")
  ggexport(plot = plot_list[[i]], filename = plot_file, width = 8, height = 6)
}

ggexport(plotlist = plot_list, filename = paste0(OUTPUT_DIR, "/all_plots.pdf"))

