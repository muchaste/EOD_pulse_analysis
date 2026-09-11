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

metadata <- read.xlsx("Recordings_Sessions_Metadata.xlsx")
man_meas <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")

metadata$t_meas_set <- NA
metadata$t_meas_retr <- NA
metadata$T_ch0_set_C <- NA
metadata$T_mid_set_C <- NA
metadata$T_ch7_set_C <- NA
metadata$DO_ch0_set_AS <- NA
metadata$DO_mid_set_AS <- NA
metadata$DO_ch7_set_AS <- NA
metadata$pH_ch0_set <- NA
metadata$pH_mid_set <- NA
metadata$pH_ch7_set <- NA
metadata$Conductivity_ch0_set_uS <- NA
metadata$Conductivity_mid_set_uS <- NA
metadata$Conductivity_ch7_set_uS <- NA
metadata$Depth_ch0_set_cm <- NA
metadata$Depth_mid_set_cm <- NA
metadata$Depth_ch7_set_cm <- NA
metadata$Turbidity_set_NTU <- NA
metadata$T_ch0_retr_C <- NA
metadata$T_mid_retr_C <- NA
metadata$T_ch7_retr_C <- NA
metadata$DO_ch0_retr_AS <- NA
metadata$DO_mid_retr_AS <- NA
metadata$DO_ch7_retr_AS <- NA
metadata$pH_ch0_retr <- NA
metadata$pH_mid_retr <- NA
metadata$pH_ch7_retr <- NA
metadata$Conductivity_ch0_retr_uS <- NA
metadata$Conductivity_mid_retr_uS <- NA
metadata$Conductivity_ch7_retr_uS <- NA
metadata$Depth_ch0_retr_cm <- NA
metadata$Depth_mid_retr_cm <- NA
metadata$Depth_ch7_retr_cm <- NA
metadata$Turbidity_retr_NTU <- NA


# add manual measurements from setting and retrieving the loggers to the metadata

metadata$Start_Date <- convertToDate(metadata$Start_Date)
metadata$End_Date <- convertToDate(metadata$End_Date)
man_meas$Date <- convertToDate(man_meas$Date)
man_meas$Time <- convertToDateTime(man_meas$Time)
date(man_meas$Time) <- man_meas$Date
man_meas$Location <- trimws(man_meas$Location)  # guard against stray whitespace from manual entry

# Same +3h read-in shift as the sampling register in script 3 (Excel-encoded UTC
# reinterpreted as Africa/Nairobi on read) - subtract 3h to get true local time.
metadata$t_first_rec_start <- as.POSIXct(convertToDateTime(metadata$t_first_rec_start), tz = TZ) - hours(1)
metadata$t_last_rec_end    <- as.POSIXct(convertToDateTime(metadata$t_last_rec_end),    tz = TZ) - hours(1)

# ---- Sunrise/sunset/moonrise/moonset from per-row GPS + session date ----------
# Computed once here (per site coordinate + Start_Date) instead of being looked
# up manually or recomputed with a single study-wide coordinate in script 3.
has_gps <- !is.na(metadata$GPS_UTM_36M_East) & !is.na(metadata$GPS_UTM_36M_North)
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
# One API call per unique GPS location, spanning the full date range recorded
# there, to avoid one request per session. Classifies WMO weather codes into a
# simple Sunny/Cloudy/Fog/Rainy/Thunderstorm label and reports daily rainfall (mm).
weathercode_to_label <- function(code) {
  if (is.na(code)) return(NA_character_)
  if (code == 0) return("Sunny")
  if (code %in% c(1, 2, 3)) return("Cloudy")
  if (code %in% c(45, 48)) return("Fog")
  if (code %in% c(51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82)) return("Rainy")
  if (code %in% c(95, 96, 99)) return("Thunderstorm")
  NA_character_
}

gps_groups <- unique(metadata[has_gps, c("GPS_UTM_36M_East", "GPS_UTM_36M_North")])

weather_daily <- NULL
for (i in seq_len(nrow(gps_groups))) {
  group_rows <- has_gps &
    metadata$GPS_UTM_36M_East == gps_groups$GPS_UTM_36M_East[i] &
    metadata$GPS_UTM_36M_North == gps_groups$GPS_UTM_36M_North[i]

  lat <- metadata$Site_lat[group_rows][1]
  lon <- metadata$Site_lon[group_rows][1]
  date_from <- min(metadata$Start_Date[group_rows], na.rm = TRUE)
  date_to   <- max(metadata$End_Date[group_rows],   na.rm = TRUE)

  url <- paste0(
    "https://archive-api.open-meteo.com/v1/archive?",
    "latitude=", lat, "&longitude=", lon,
    "&start_date=", date_from, "&end_date=", date_to,
    "&daily=weathercode,precipitation_sum&timezone=", URLencode(TZ, reserved = TRUE)
  )

  resp <- tryCatch(fromJSON(url), error = function(e) NULL)
  if (is.null(resp) || is.null(resp$daily)) {
    warning(paste0("Weather API request failed for GPS ", lat, ",", lon))
    next
  }

  weather_daily <- rbind(weather_daily, data.frame(
    GPS_UTM_36M_East  = gps_groups$GPS_UTM_36M_East[i],
    GPS_UTM_36M_North = gps_groups$GPS_UTM_36M_North[i],
    Start_Date        = as.Date(resp$daily$time),
    Weather_computed  = sapply(resp$daily$weathercode, weathercode_to_label),
    Rainfall_mm       = resp$daily$precipitation_sum
  ))
}
weather_daily$Rain_computed <- ifelse(weather_daily$Rainfall_mm > 0.1, "yes", "no")

metadata <- merge(metadata, weather_daily,
                   by = c("GPS_UTM_36M_East", "GPS_UTM_36M_North", "Start_Date"), all.x = TRUE)

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

full_metadata <- NULL

for(logger_id in unique(metadata$Logger_ID)){
  logger_sub <- subset(metadata, Logger_ID == logger_id)
  for(session in unique(logger_sub$Start_Date)){
    session_sub <- subset(logger_sub, Start_Date == session)
    if (nrow(session_sub) > 1){
      warning(paste0("Multiple metadata rows for logger ", logger_id, " on ", session, " - skipping"))
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

    session_sub$t_meas_set  <- safe_extract(mm_sub_set$Time,  rep(TRUE, nrow(mm_sub_set)),  paste(label, "t_meas_set"))
    session_sub$t_meas_retr <- safe_extract(mm_sub_retr$Time, rep(TRUE, nrow(mm_sub_retr)), paste(label, "t_meas_retr"))

    # match the tag as the exact suffix after "LX-" so "mid" cannot match inside other text
    mask_ch0_set  <- endsWith(mm_sub_set$Location,  paste0("-", ch0_id))
    mask_mid_set  <- endsWith(mm_sub_set$Location,  paste0("-", mid_id))
    mask_ch7_set  <- endsWith(mm_sub_set$Location,  paste0("-", ch7_id))
    mask_ch0_retr <- endsWith(mm_sub_retr$Location, paste0("-", ch0_id))
    mask_mid_retr <- endsWith(mm_sub_retr$Location, paste0("-", mid_id))
    mask_ch7_retr <- endsWith(mm_sub_retr$Location, paste0("-", ch7_id))

    session_sub$T_ch0_set_C <- safe_extract(mm_sub_set$T_C, mask_ch0_set, paste(label, "T_ch0_set"))
    session_sub$T_mid_set_C <- safe_extract(mm_sub_set$T_C, mask_mid_set, paste(label, "T_mid_set"))
    session_sub$T_ch7_set_C <- safe_extract(mm_sub_set$T_C, mask_ch7_set, paste(label, "T_ch7_set"))

    session_sub$T_ch0_retr_C <- safe_extract(mm_sub_retr$T_C, mask_ch0_retr, paste(label, "T_ch0_retr"))
    session_sub$T_mid_retr_C <- safe_extract(mm_sub_retr$T_C, mask_mid_retr, paste(label, "T_mid_retr"))
    session_sub$T_ch7_retr_C <- safe_extract(mm_sub_retr$T_C, mask_ch7_retr, paste(label, "T_ch7_retr"))

    session_sub$DO_ch0_set_AS <- safe_extract(mm_sub_set$DO_AS, mask_ch0_set, paste(label, "DO_ch0_set"))
    session_sub$DO_mid_set_AS <- safe_extract(mm_sub_set$DO_AS, mask_mid_set, paste(label, "DO_mid_set"))
    session_sub$DO_ch7_set_AS <- safe_extract(mm_sub_set$DO_AS, mask_ch7_set, paste(label, "DO_ch7_set"))

    session_sub$DO_ch0_retr_AS <- safe_extract(mm_sub_retr$DO_AS, mask_ch0_retr, paste(label, "DO_ch0_retr"))
    session_sub$DO_mid_retr_AS <- safe_extract(mm_sub_retr$DO_AS, mask_mid_retr, paste(label, "DO_mid_retr"))
    session_sub$DO_ch7_retr_AS <- safe_extract(mm_sub_retr$DO_AS, mask_ch7_retr, paste(label, "DO_ch7_retr"))

    session_sub$Conductivity_ch0_set_uS <- safe_extract(mm_sub_set$Cond_uS, mask_ch0_set, paste(label, "Cond_ch0_set"))
    session_sub$Conductivity_mid_set_uS <- safe_extract(mm_sub_set$Cond_uS, mask_mid_set, paste(label, "Cond_mid_set"))
    session_sub$Conductivity_ch7_set_uS <- safe_extract(mm_sub_set$Cond_uS, mask_ch7_set, paste(label, "Cond_ch7_set"))

    session_sub$Conductivity_ch0_retr_uS <- safe_extract(mm_sub_retr$Cond_uS, mask_ch0_retr, paste(label, "Cond_ch0_retr"))
    session_sub$Conductivity_mid_retr_uS <- safe_extract(mm_sub_retr$Cond_uS, mask_mid_retr, paste(label, "Cond_mid_retr"))
    session_sub$Conductivity_ch7_retr_uS <- safe_extract(mm_sub_retr$Cond_uS, mask_ch7_retr, paste(label, "Cond_ch7_retr"))

    session_sub$pH_ch0_set <- safe_extract(mm_sub_set$pH, mask_ch0_set, paste(label, "pH_ch0_set"))
    session_sub$pH_mid_set <- safe_extract(mm_sub_set$pH, mask_mid_set, paste(label, "pH_mid_set"))
    session_sub$pH_ch7_set <- safe_extract(mm_sub_set$pH, mask_ch7_set, paste(label, "pH_ch7_set"))

    session_sub$pH_ch0_retr <- safe_extract(mm_sub_retr$pH, mask_ch0_retr, paste(label, "pH_ch0_retr"))
    session_sub$pH_mid_retr <- safe_extract(mm_sub_retr$pH, mask_mid_retr, paste(label, "pH_mid_retr"))
    session_sub$pH_ch7_retr <- safe_extract(mm_sub_retr$pH, mask_ch7_retr, paste(label, "pH_ch7_retr"))

    session_sub$Depth_ch0_set_cm <- safe_extract(mm_sub_set$h_cm, mask_ch0_set, paste(label, "Depth_ch0_set"))
    session_sub$Depth_mid_set_cm <- safe_extract(mm_sub_set$h_cm, mask_mid_set, paste(label, "Depth_mid_set"))
    session_sub$Depth_ch7_set_cm <- safe_extract(mm_sub_set$h_cm, mask_ch7_set, paste(label, "Depth_ch7_set"))

    session_sub$Depth_ch0_retr_cm <- safe_extract(mm_sub_retr$h_cm, mask_ch0_retr, paste(label, "Depth_ch0_retr"))
    session_sub$Depth_mid_retr_cm <- safe_extract(mm_sub_retr$h_cm, mask_mid_retr, paste(label, "Depth_mid_retr"))
    session_sub$Depth_ch7_retr_cm <- safe_extract(mm_sub_retr$h_cm, mask_ch7_retr, paste(label, "Depth_ch7_retr"))

    turb_set  <- mean(mm_sub_set$Turbidity_NTU, na.rm = TRUE)
    turb_retr <- mean(mm_sub_retr$Turbidity_NTU, na.rm = TRUE)
    session_sub$Turbidity_set_NTU  <- if (is.nan(turb_set))  NA else turb_set
    session_sub$Turbidity_retr_NTU <- if (is.nan(turb_retr)) NA else turb_retr

    full_metadata <- rbind(full_metadata, session_sub)

  }
}

write.csv2(full_metadata, "Recordings_Sessions_Metadata_Complete.csv", row.names = FALSE)
write.xlsx(full_metadata, "Recordings_Sessions_Metadata_Complete.xlsx", rowNames = FALSE)
