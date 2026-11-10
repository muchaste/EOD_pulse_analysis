library(openxlsx)
library(lubridate)
library(sf)
library(suncalc)
library(jsonlite)
library(dplyr)

rm(list = ls())

UTM_CRS <- 32736  # UTM zone 36S, matches GPS_UTM_36M_East/North columns
TZ <- "Africa/Nairobi"

# Read in metadata and environmental data
metadata <- read.csv2("01 Data/Recordings_Sessions_Metadata.csv", sep = ",", dec = ".")
env_summary <- read.csv2("01 Data/Environmental Data/Complete_env_data_summary_per_day.csv")

# Parse time and shift if necessary (Check!!) (Excel-encoded UTC
# reinterpreted as Africa/Nairobi on read)
metadata$t_first_rec_start <- as.POSIXct(metadata$t_first_rec_start, format = "%d.%m.%Y %H:%M", tz = TZ) # - hours(1)
metadata$t_last_rec_end    <- as.POSIXct(metadata$t_last_rec_end, format = "%d.%m.%Y %H:%M", tz = TZ) # - hours(1)
metadata$Start_Date <- as.Date(metadata$Start_Date, format = "%d.%m.%Y")
metadata$End_Date <- as.Date(metadata$End_Date, format = "%d.%m.%Y")
metadata$Session_date <- metadata$Start_Date
env_summary$Session_date <- as.Date(env_summary$Session_date)

# ---- Sunrise/sunset/moonrise/moonset from per-row GPS + session date ----------
# Computed once here (per site coordinate + Start_Date) instead of being looked
# up manually or recomputed with a single study-wide coordinate in script 3.
has_gps <- !is.na(metadata$GPS_UTM_36M_East) & !is.na(metadata$GPS_UTM_36M_North)
metadata$has_gps <- has_gps  # kept in output so missing sun/moon/weather can be traced to missing GPS
# rows with GPS but no recorded start/end (e.g. planned session that didn't happen) have no window to compute sun/moon/weather for
has_window <- !is.na(metadata$t_first_rec_start) & !is.na(metadata$t_last_rec_end)
# Sun/moon/weather are computed over a standardized 24h window from recording start (not the
# actual, variable recording duration) so every session yields exactly one sunrise-after-sunset
# pair, regardless of how long the EOD logger actually ran.
metadata$window_end_std <- metadata$t_first_rec_start + hours(24)
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

# Each session's window is standardized to 24h from t_first_rec_start (see window_end_std
# above), so exactly one sunrise/sunset/moonrise/moonset of each kind falls within it.
for (i in which(has_gps & has_window)) {
  window_start <- metadata$t_first_rec_start[i]
  window_end   <- metadata$window_end_std[i]
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

gps_groups <- unique(metadata[has_gps & has_window, c("GPS_UTM_36M_East", "GPS_UTM_36M_North")])

metadata$Rainfall_mm      <- NA_real_
metadata$Rain_computed    <- NA_character_
metadata$Cloudcover_mean_pct <- NA_real_
metadata$Weather_dominant <- NA_character_
metadata$Weather_n_hours  <- 0L  # number of hourly records found within the session window
for (label in WEATHER_LABELS) metadata[[paste0("pct_hours_", tolower(label))]] <- NA_real_

for (g in seq_len(nrow(gps_groups))) {
  group_rows <- which(has_gps & has_window &
    metadata$GPS_UTM_36M_East == gps_groups$GPS_UTM_36M_East[g] &
    metadata$GPS_UTM_36M_North == gps_groups$GPS_UTM_36M_North[g])

  lat <- metadata$Site_lat[group_rows[1]]
  lon <- metadata$Site_lon[group_rows[1]]
  date_from <- as.Date(min(metadata$t_first_rec_start[group_rows], na.rm = TRUE))
  date_to   <- as.Date(max(metadata$window_end_std[group_rows],   na.rm = TRUE))

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
                             hourly$time <= metadata$window_end_std[i], ]
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



# NEW - use env_summary from script 1_6 -----------------------------------
# env_summary's weather/moon/sun columns are calendar-day based (see 1_6) and would
# collide (.x/.y) with the session-precise versions computed above; its GPS/lat/lon
# columns (from 1_6's gps_key merge) likewise collide with metadata's own GPS fields.
# Keep only its sensor/manual-measurement aggregates here.
env_summary_sensors <- env_summary |>
  select(-any_of(c("Sunrise", "Sunset", "Moonrise", "Moonset",
                    "Moon_illuminated_fraction", "Moon_phase_value", "Moon_phase_name",
                    "Rainfall_mm", "Rain_computed", "Cloudcover_mean_pct",
                    "Weather_dominant", "Weather_n_hours",
                    "pct_hours_sunny", "pct_hours_cloudy", "pct_hours_fog",
                    "pct_hours_rainy", "pct_hours_thunderstorm",
                    "GPS_UTM_36M_East", "GPS_UTM_36M_North", "lat", "lon",
                    "Logger.and.electrode.placement")))

metadata_complete <- left_join(metadata, env_summary_sensors, by = c("Session_date", "System", "Site", "Logger_ID"))

write.csv2(metadata_complete, "01 Data/Recordings_Sessions_Environment_Complete.csv", row.names = F)
