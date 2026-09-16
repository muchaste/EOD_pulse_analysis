library(openxlsx)
library(lubridate)
library(sf)
library(suncalc)
library(jsonlite)

rm(list = ls())

UTM_CRS <- 32736  # UTM zone 36S, matches GPS_UTM_36M_East/North columns
TZ <- "Africa/Nairobi"

# setwd("C:/Users/stefa/Seafile/Uni/01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/")
setwd("C:/Users/Admin/Projects/EOD_pulse_analysis/eod_logger_4/Uganda_2023/")

metadata <- read.csv2("Recordings_Sessions_Metadata.csv", sep = ",", dec = ".")
man_meas <- read.csv2("Manual Measurements.csv", sep = ",", dec = ".")

# add manual measurements from setting and retrieving the loggers to the metadata
metadata$Start_Date <- as.Date(metadata$Start_Date, format = "%d/%m/%Y")
metadata$End_Date <- as.Date(metadata$End_Date, format = "%d/%m/%Y")
man_meas$Date <- as.Date(man_meas$Date, format = "%d/%m/%Y")
man_meas$Time <- as.POSIXct(man_meas$Time, format = "%H:%M", tz = TZ)
date(man_meas$Time) <- man_meas$Date
man_meas$Location <- trimws(man_meas$Location)  # guard against stray whitespace from manual entry

# Parse time and shift if necessary (Check!!) (Excel-encoded UTC
# reinterpreted as Africa/Nairobi on read)
metadata$t_first_rec_start <- as.POSIXct(metadata$t_first_rec_start, format = "%d/%m/%Y %H:%M:%S", tz = TZ) # - hours(1)
metadata$t_last_rec_end    <- as.POSIXct(metadata$t_last_rec_end, format = "%d/%m/%Y %H:%M:%S", tz = TZ) # - hours(1)

# ---- Sunrise/sunset/moonrise/moonset from per-row GPS + session date ----------
# Computed once here (per site coordinate + Start_Date) instead of being looked
# up manually or recomputed with a single study-wide coordinate in script 3.
has_gps <- !is.na(metadata$GPS_UTM_36M_East) & !is.na(metadata$GPS_UTM_36M_North)
metadata$has_gps <- has_gps  # kept in output so missing sun/moon/weather can be traced to missing GPS
metadata$Site_lat <- NA_real_
metadata$Site_lon <- NA_real_

coords_utm <- st_as_sf(metadata[has_gps, ], coords = c("GPS_UTM_36M_East", "GPS_UTM_36M_North"), crs = UTM_CRS)
coords_wgs <- st_transform(coords_utm, crs = 4326)
metadata$Site_lon[has_gps] <- st_coordinates(coords_wgs)[, "X"]
metadata$Site_lat[has_gps] <- st_coordinates(coords_wgs)[, "Y"]

metadata$Sunrise  <- as.POSIXct(NA, tz = TZ)
metadata$Sunset   <- as.POSIXct(NA, tz = TZ)
metadata$Moonrise <- as.POSIXct(NA, tz = TZ)
metadata$Moonset  <- as.POSIXct(NA, tz = TZ)
metadata$Moon_illuminated_fraction <- NA_real_
metadata$Moon_phase_value <- NA_real_       # 0 = new moon, 0.5 = full moon, 1 = next new moon
metadata$Moon_phase_name  <- NA_character_

moon_phase_name <- function(phase) {
  if (is.na(phase)) return(NA_character_)
  breaks <- c(0, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16, 1)
  labels <- c("New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon",
              "Waning Gibbous", "Last Quarter", "Waning Crescent", "New Moon")
  labels[findInterval(phase, breaks, rightmost.closed = TRUE)]
}

# Many sessions span >24h and cross more than one sunrise/sunset/moonrise/moonset.
# For each session, only keep the first occurrence of each event that falls
# within the actual recording window (t_first_rec_start to t_last_rec_end).
for (i in which(has_gps)) {
  window_start <- metadata$t_first_rec_start[i]
  window_end   <- metadata$t_last_rec_end[i]
  candidate_dates <- seq(as.Date(window_start) - 1, as.Date(window_end) + 1, by = "day")
  position_data <- data.frame(date = candidate_dates, lat = metadata$Site_lat[i], lon = metadata$Site_lon[i])

  sun_candidates  <- getSunlightTimes(data = position_data, keep = c("sunrise", "sunset"), tz = TZ)
  moon_candidates <- getMoonTimes(data = position_data, keep = c("rise", "set"), tz = TZ)

  sunrise_in_window  <- sun_candidates$sunrise[sun_candidates$sunrise >= window_start & sun_candidates$sunrise <= window_end]
  sunset_in_window   <- sun_candidates$sunset[sun_candidates$sunset   >= window_start & sun_candidates$sunset  <= window_end]
  moonrise_in_window <- moon_candidates$rise[!is.na(moon_candidates$rise) & moon_candidates$rise >= window_start & moon_candidates$rise <= window_end]
  moonset_in_window  <- moon_candidates$set[!is.na(moon_candidates$set)   & moon_candidates$set  >= window_start & moon_candidates$set  <= window_end]

  if (length(sunrise_in_window)  > 0) metadata$Sunrise[i]  <- min(sunrise_in_window)
  if (length(sunset_in_window)   > 0) metadata$Sunset[i]   <- min(sunset_in_window)
  if (length(moonrise_in_window) > 0) metadata$Moonrise[i] <- min(moonrise_in_window)
  if (length(moonset_in_window)  > 0) metadata$Moonset[i]  <- min(moonset_in_window)

  moon_illum <- getMoonIllumination(date = as.Date(window_start), keep = c("fraction", "phase"))
  metadata$Moon_illuminated_fraction[i] <- moon_illum$fraction
  metadata$Moon_phase_value[i] <- moon_illum$phase
  metadata$Moon_phase_name[i]  <- moon_phase_name(moon_illum$phase)
}

# ---- Historical weather from Open-Meteo Archive API (free, no key needed) ----
# One hourly-resolution API call per unique GPS location, spanning the full date
# range recorded there, then each session summarizes only the hours that fall
# within its own [t_first_rec_start, t_last_rec_end] window.
weathercode_to_label <- function(code) {
  if (is.na(code)) return(NA_character_)
  if (code == 0) return("Sunny")
  if (code %in% c(1, 2, 3)) return("Cloudy")
  if (code %in% c(45, 48)) return("Fog")
  if (code %in% c(51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82)) return("Rainy")
  if (code %in% c(95, 96, 99)) return("Thunderstorm")
  NA_character_
}
WEATHER_LABELS <- c("Sunny", "Cloudy", "Fog", "Rainy", "Thunderstorm")

gps_groups <- unique(metadata[has_gps, c("GPS_UTM_36M_East", "GPS_UTM_36M_North")])

metadata$Rainfall_mm      <- NA_real_
metadata$Rain_computed    <- NA_character_
metadata$Cloudcover_mean_pct <- NA_real_
metadata$Weather_dominant <- NA_character_
metadata$Weather_n_hours  <- 0L  # number of hourly records found within the session window
for (label in WEATHER_LABELS) metadata[[paste0("pct_hours_", tolower(label))]] <- NA_real_

for (g in seq_len(nrow(gps_groups))) {
  group_rows <- which(has_gps &
    metadata$GPS_UTM_36M_East == gps_groups$GPS_UTM_36M_East[g] &
    metadata$GPS_UTM_36M_North == gps_groups$GPS_UTM_36M_North[g])

  lat <- metadata$Site_lat[group_rows[1]]
  lon <- metadata$Site_lon[group_rows[1]]
  date_from <- as.Date(min(metadata$t_first_rec_start[group_rows], na.rm = TRUE))
  date_to   <- as.Date(max(metadata$t_last_rec_end[group_rows],   na.rm = TRUE))

  url <- paste0(
    "https://archive-api.open-meteo.com/v1/archive?",
    "latitude=", lat, "&longitude=", lon,
    "&start_date=", date_from, "&end_date=", date_to,
    "&hourly=weathercode,precipitation,cloudcover&timezone=", URLencode(TZ, reserved = TRUE)
  )

  resp <- tryCatch(fromJSON(url), error = function(e) NULL)
  if (is.null(resp) || is.null(resp$hourly)) {
    warning(paste0("Weather API request failed for GPS ", lat, ",", lon))
    next
  }

  hourly <- data.frame(
    time          = as.POSIXct(resp$hourly$time, format = "%Y-%m-%dT%H:%M", tz = TZ),
    precipitation = resp$hourly$precipitation,
    cloudcover    = resp$hourly$cloudcover,
    weather_label = sapply(resp$hourly$weathercode, weathercode_to_label)
  )

  for (i in group_rows) {
    session_hours <- hourly[hourly$time >= metadata$t_first_rec_start[i] &
                             hourly$time <= metadata$t_last_rec_end[i], ]
    metadata$Weather_n_hours[i] <- nrow(session_hours)
    if (nrow(session_hours) == 0) next

    metadata$Rainfall_mm[i]         <- sum(session_hours$precipitation, na.rm = TRUE)
    metadata$Rain_computed[i]       <- ifelse(metadata$Rainfall_mm[i] > 0.1, "yes", "no")
    metadata$Cloudcover_mean_pct[i] <- mean(session_hours$cloudcover, na.rm = TRUE)

    label_pct <- table(factor(session_hours$weather_label, levels = WEATHER_LABELS)) / nrow(session_hours) * 100
    for (label in WEATHER_LABELS) metadata[[paste0("pct_hours_", tolower(label))]][i] <- unname(label_pct[label])
    metadata$Weather_dominant[i] <- names(which.max(label_pct))
  }
}

# Returns exactly one value for `mask`, or NA if no match, or the mean (with a
# warning identifying the session) if more than one row matches unexpectedly.
safe_extract <- function(vec, mask, label) {
  vals <- vec[mask]
  if (length(vals) == 0) return(NA)
  if (length(vals) > 1) {
    warning(paste0(label, ": ", length(vals), " matching rows, using mean"))
    return(mean(vals, na.rm = TRUE))
  }
  vals
}

# Combines readings across channels/timing into one mean/min/max/n (n = number of
# non-missing readings actually used, for quality control)
summarize_measurement <- function(vals) {
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0) return(c(mean = NA_real_, min = NA_real_, max = NA_real_, n = 0))
  c(mean = mean(vals), min = min(vals), max = max(vals), n = length(vals))
}

# ---- Logger-recorded temperature/light, pooled across all electrode locations ----
# Location (e.g. ortho_edge/ground_middle/vert_half) marks position along the cable,
# analogous to channel, and is pooled together per session. Uses the raw Datetime
# column (true calendar time); Datetime_fix/Datetime_fix_loop are deliberately
# rebased onto one fake shared reference date for overlay plots only (see
# "1_1_Env - Format and compile raw HOBO files.R") and must not be used for joins.
light_temp <- read.csv2("Temp_Light_All_Logger_Data.csv")
light_temp$Datetime <- as.POSIXct(light_temp$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = TZ)

metadata$T_log_mean_C <- NA_real_
metadata$T_log_min_C  <- NA_real_
metadata$T_log_max_C  <- NA_real_
metadata$T_log_n      <- 0L
metadata$Lux_log_mean <- NA_real_
metadata$Lux_log_min  <- NA_real_
metadata$Lux_log_max  <- NA_real_
metadata$Lux_log_n    <- 0L

for (i in seq_len(nrow(metadata))) {
  session_rows <- light_temp$System == metadata$System[i] & light_temp$Site == metadata$Site[i] &
    light_temp$Datetime >= metadata$t_first_rec_start[i] & light_temp$Datetime <= metadata$t_last_rec_end[i]
  session_rows[is.na(session_rows)] <- FALSE
  if (!any(session_rows)) next

  t_stats   <- summarize_measurement(light_temp$T_C[session_rows])
  lux_stats <- summarize_measurement(light_temp$Lux[session_rows])

  metadata$T_log_mean_C[i] <- t_stats["mean"]
  metadata$T_log_min_C[i]  <- t_stats["min"]
  metadata$T_log_max_C[i]  <- t_stats["max"]
  metadata$T_log_n[i]      <- t_stats["n"]
  metadata$Lux_log_mean[i] <- lux_stats["mean"]
  metadata$Lux_log_min[i]  <- lux_stats["min"]
  metadata$Lux_log_max[i]  <- lux_stats["max"]
  metadata$Lux_log_n[i]    <- lux_stats["n"]
}

full_metadata <- NULL

metadata$t_meas_set <- NA_real_
metadata$t_meas_retr <- NA_real_
metadata$n_man_meas_set  <- 0L  # total manual-measurement rows found for this logger/date, regardless of channel tag
metadata$n_man_meas_retr <- 0L
metadata$T_man_mean_C <- NA_real_
metadata$T_man_min_C <- NA_real_
metadata$T_man_max_C <- NA_real_
metadata$T_man_n <- 0L
# metadata$T_ch0_set_C <- NA
# metadata$T_mid_set_C <- NA
# metadata$T_ch7_set_C <- NA
metadata$DO_man_mean_AS <- NA_real_
metadata$DO_man_min_AS <- NA_real_
metadata$DO_man_max_AS <- NA_real_
metadata$DO_man_n <- 0L
# metadata$DO_ch0_set_AS <- NA
# metadata$DO_mid_set_AS <- NA
# metadata$DO_ch7_set_AS <- NA
metadata$pH_man_mean <- NA_real_
metadata$pH_man_min <- NA_real_
metadata$pH_man_max <- NA_real_
metadata$pH_man_n <- 0L
# metadata$pH_ch0_set <- NA
# metadata$pH_mid_set <- NA
# metadata$pH_ch7_set <- NA
metadata$Conductivity_man_mean_uS <- NA_real_
metadata$Conductivity_man_min_uS <- NA_real_
metadata$Conductivity_man_max_uS <- NA_real_
metadata$Conductivity_man_n <- 0L
# metadata$Conductivity_ch0_set_uS <- NA
# metadata$Conductivity_mid_set_uS <- NA
# metadata$Conductivity_ch7_set_uS <- NA
metadata$Depth_man_mean_cm <- NA_real_
metadata$Depth_man_min_cm <- NA_real_
metadata$Depth_man_max_cm <- NA_real_
metadata$Depth_man_n <- 0L
# metadata$Depth_ch0_set_cm <- NA
# metadata$Depth_mid_set_cm <- NA
# metadata$Depth_ch7_set_cm <- NA
metadata$Turbidity_mean_NTU <- NA_real_
metadata$Turbidity_min_NTU <- NA_real_
metadata$Turbidity_max_NTU <- NA_real_
metadata$Turbidity_n <- 0L
# metadata$Turbidity_set_NTU <- NA
# metadata$T_ch0_retr_C <- NA
# metadata$T_mid_retr_C <- NA
# metadata$T_ch7_retr_C <- NA
# metadata$DO_ch0_retr_AS <- NA
# metadata$DO_mid_retr_AS <- NA
# metadata$DO_ch7_retr_AS <- NA
# metadata$pH_ch0_retr <- NA
# metadata$pH_mid_retr <- NA
# metadata$pH_ch7_retr <- NA
# metadata$Conductivity_ch0_retr_uS <- NA
# metadata$Conductivity_mid_retr_uS <- NA
# metadata$Conductivity_ch7_retr_uS <- NA
# metadata$Depth_ch0_retr_cm <- NA
# metadata$Depth_mid_retr_cm <- NA
# metadata$Depth_ch7_retr_cm <- NA
# metadata$Turbidity_retr_NTU <- NA

n_sessions_skipped_duplicate <- 0L

for(logger_id in unique(metadata$Logger_ID)){
  logger_sub <- subset(metadata, Logger_ID == logger_id)
  for(session in unique(logger_sub$Start_Date)){
    session_sub <- subset(logger_sub, Start_Date == session)
    if (nrow(session_sub) > 1){
      warning(paste0("Multiple metadata rows for logger ", logger_id, " on ", session, " - skipping"))
      n_sessions_skipped_duplicate <- n_sessions_skipped_duplicate + 1L
      next
    }
    site <- session_sub$Site
    label <- paste(logger_id, site, session)

    # identify measurements from ch0, mid and ch7 from the location column:
    # ch0 = LX-edge
    # mid = LX-half
    # ch7 = LX-mid (I know, bad naming)
    ch0_id <- "edge"
    mid_id <- "half"
    ch7_id <- "mid"

    # anchor logger id at the start of "LX-tag" so e.g. L1 cannot match L10/L11
    logger_pattern <- paste0("^", logger_id, "-")
    mm_sub_set  <- subset(man_meas, Site == site & Date == session & grepl(logger_pattern, Location))
    mm_sub_retr <- subset(man_meas, Site == site & Date == session_sub$End_Date & grepl(logger_pattern, Location))
    session_sub$n_man_meas_set  <- nrow(mm_sub_set)
    session_sub$n_man_meas_retr <- nrow(mm_sub_retr)

    session_sub$t_meas_set  <- safe_extract(mm_sub_set$Time,  rep(TRUE, nrow(mm_sub_set)),  paste(label, "t_meas_set"))
    session_sub$t_meas_retr <- safe_extract(mm_sub_retr$Time, rep(TRUE, nrow(mm_sub_retr)), paste(label, "t_meas_retr"))

    # match the tag as the exact suffix after "LX-" so "mid" cannot match inside other text
    mask_ch0_set  <- endsWith(mm_sub_set$Location,  paste0("-", ch0_id))
    mask_mid_set  <- endsWith(mm_sub_set$Location,  paste0("-", mid_id))
    mask_ch7_set  <- endsWith(mm_sub_set$Location,  paste0("-", ch7_id))
    mask_ch0_retr <- endsWith(mm_sub_retr$Location, paste0("-", ch0_id))
    mask_mid_retr <- endsWith(mm_sub_retr$Location, paste0("-", mid_id))
    mask_ch7_retr <- endsWith(mm_sub_retr$Location, paste0("-", ch7_id))

    # unify all channels (ch0/mid/ch7) and both timings (set/retrieve) into one mean/min/max per session
    t_vals <- c(
      safe_extract(mm_sub_set$T_C,  mask_ch0_set,  paste(label, "T_ch0_set")),
      safe_extract(mm_sub_set$T_C,  mask_mid_set,  paste(label, "T_mid_set")),
      safe_extract(mm_sub_set$T_C,  mask_ch7_set,  paste(label, "T_ch7_set")),
      safe_extract(mm_sub_retr$T_C, mask_ch0_retr, paste(label, "T_ch0_retr")),
      safe_extract(mm_sub_retr$T_C, mask_mid_retr, paste(label, "T_mid_retr")),
      safe_extract(mm_sub_retr$T_C, mask_ch7_retr, paste(label, "T_ch7_retr"))
    )
    t_stats <- summarize_measurement(t_vals)
    session_sub$T_man_mean_C <- t_stats["mean"]
    session_sub$T_man_min_C  <- t_stats["min"]
    session_sub$T_man_max_C  <- t_stats["max"]
    session_sub$T_man_n      <- t_stats["n"]

    do_vals <- c(
      safe_extract(mm_sub_set$DO_AS,  mask_ch0_set,  paste(label, "DO_ch0_set")),
      safe_extract(mm_sub_set$DO_AS,  mask_mid_set,  paste(label, "DO_mid_set")),
      safe_extract(mm_sub_set$DO_AS,  mask_ch7_set,  paste(label, "DO_ch7_set")),
      safe_extract(mm_sub_retr$DO_AS, mask_ch0_retr, paste(label, "DO_ch0_retr")),
      safe_extract(mm_sub_retr$DO_AS, mask_mid_retr, paste(label, "DO_mid_retr")),
      safe_extract(mm_sub_retr$DO_AS, mask_ch7_retr, paste(label, "DO_ch7_retr"))
    )
    do_stats <- summarize_measurement(do_vals)
    session_sub$DO_man_mean_AS <- do_stats["mean"]
    session_sub$DO_man_min_AS  <- do_stats["min"]
    session_sub$DO_man_max_AS  <- do_stats["max"]
    session_sub$DO_man_n       <- do_stats["n"]

    cond_vals <- c(
      safe_extract(mm_sub_set$Cond_uS,  mask_ch0_set,  paste(label, "Cond_ch0_set")),
      safe_extract(mm_sub_set$Cond_uS,  mask_mid_set,  paste(label, "Cond_mid_set")),
      safe_extract(mm_sub_set$Cond_uS,  mask_ch7_set,  paste(label, "Cond_ch7_set")),
      safe_extract(mm_sub_retr$Cond_uS, mask_ch0_retr, paste(label, "Cond_ch0_retr")),
      safe_extract(mm_sub_retr$Cond_uS, mask_mid_retr, paste(label, "Cond_mid_retr")),
      safe_extract(mm_sub_retr$Cond_uS, mask_ch7_retr, paste(label, "Cond_ch7_retr"))
    )
    cond_stats <- summarize_measurement(cond_vals)
    session_sub$Conductivity_man_mean_uS <- cond_stats["mean"]
    session_sub$Conductivity_man_min_uS  <- cond_stats["min"]
    session_sub$Conductivity_man_max_uS  <- cond_stats["max"]
    session_sub$Conductivity_man_n       <- cond_stats["n"]

    pH_vals <- c(
      safe_extract(mm_sub_set$pH,  mask_ch0_set,  paste(label, "pH_ch0_set")),
      safe_extract(mm_sub_set$pH,  mask_mid_set,  paste(label, "pH_mid_set")),
      safe_extract(mm_sub_set$pH,  mask_ch7_set,  paste(label, "pH_ch7_set")),
      safe_extract(mm_sub_retr$pH, mask_ch0_retr, paste(label, "pH_ch0_retr")),
      safe_extract(mm_sub_retr$pH, mask_mid_retr, paste(label, "pH_mid_retr")),
      safe_extract(mm_sub_retr$pH, mask_ch7_retr, paste(label, "pH_ch7_retr"))
    )
    pH_stats <- summarize_measurement(pH_vals)
    session_sub$pH_man_mean <- pH_stats["mean"]
    session_sub$pH_man_min  <- pH_stats["min"]
    session_sub$pH_man_max  <- pH_stats["max"]
    session_sub$pH_man_n    <- pH_stats["n"]

    depth_vals <- c(
      safe_extract(mm_sub_set$h_cm,  mask_ch0_set,  paste(label, "Depth_ch0_set")),
      safe_extract(mm_sub_set$h_cm,  mask_mid_set,  paste(label, "Depth_mid_set")),
      safe_extract(mm_sub_set$h_cm,  mask_ch7_set,  paste(label, "Depth_ch7_set")),
      safe_extract(mm_sub_retr$h_cm, mask_ch0_retr, paste(label, "Depth_ch0_retr")),
      safe_extract(mm_sub_retr$h_cm, mask_mid_retr, paste(label, "Depth_mid_retr")),
      safe_extract(mm_sub_retr$h_cm, mask_ch7_retr, paste(label, "Depth_ch7_retr"))
    )
    depth_stats <- summarize_measurement(depth_vals)
    session_sub$Depth_man_mean_cm <- depth_stats["mean"]
    session_sub$Depth_man_min_cm  <- depth_stats["min"]
    session_sub$Depth_man_max_cm  <- depth_stats["max"]
    session_sub$Depth_man_n       <- depth_stats["n"]

    # turbidity is not measured per channel, so pool set and retrieve readings directly
    turb_stats <- summarize_measurement(c(mm_sub_set$Turbidity_NTU, mm_sub_retr$Turbidity_NTU))
    session_sub$Turbidity_mean_NTU <- turb_stats["mean"]
    session_sub$Turbidity_min_NTU  <- turb_stats["min"]
    session_sub$Turbidity_max_NTU  <- turb_stats["max"]
    session_sub$Turbidity_n        <- turb_stats["n"]

    full_metadata <- rbind(full_metadata, session_sub)

  }
}

write.csv2(full_metadata, "Recordings_Sessions_Metadata_Complete.csv", row.names = FALSE)
write.xlsx(full_metadata, "Recordings_Sessions_Metadata_Complete.xlsx", rowNames = FALSE)

# ---- Quality control summary --------------------------------------------------
cat("\n==== QC summary:", nrow(full_metadata), "sessions in output ====\n")
cat("Sessions skipped as duplicate logger/date rows:", n_sessions_skipped_duplicate, "\n")
cat("Sessions with GPS coordinates:", sum(full_metadata$has_gps), "/", nrow(full_metadata), "\n")
cat("Sessions with Sunrise/Sunset resolved:", sum(!is.na(full_metadata$Sunrise)), "/", nrow(full_metadata), "\n")
cat("Sessions with Moonrise/Moonset resolved:", sum(!is.na(full_metadata$Moonrise)), "/", nrow(full_metadata), "\n")
cat("Sessions with weather data (Weather_n_hours > 0):", sum(full_metadata$Weather_n_hours > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with logger temperature data (T_log_n > 0):", sum(full_metadata$T_log_n > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with logger light data (Lux_log_n > 0):", sum(full_metadata$Lux_log_n > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with any manual measurements taken (n_man_meas_set/retr > 0):",
    sum(full_metadata$n_man_meas_set > 0 | full_metadata$n_man_meas_retr > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with manual temperature reading (T_man_n > 0):", sum(full_metadata$T_man_n > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with manual DO reading (DO_man_n > 0):", sum(full_metadata$DO_man_n > 0), "/", nrow(full_metadata), "\n")
cat("Sessions with manual turbidity reading (Turbidity_n > 0):", sum(full_metadata$Turbidity_n > 0), "/", nrow(full_metadata), "\n")
