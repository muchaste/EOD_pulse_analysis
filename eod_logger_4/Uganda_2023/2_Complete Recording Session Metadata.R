library(openxlsx)
library(lubridate)

rm(list = ls())


# setwd("C:/Users/stefa/Seafile/Uni/01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/")
setwd("C:/Users/Admin/Projects/EOD_pulse_analysis/eod_logger_4/Uganda_2023/")

metadata <- read.xlsx("Recordings_Sessions_Metadata.xlsx")
man_meas <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")

metadata$t_meas_set <- NA
metadata$t_meas_retr <- NA

# add manual measurements from setting and retrieving the loggers to the metadata

metadata$Start_Date <- convertToDate(metadata$Start_Date)
metadata$End_Date <- convertToDate(metadata$End_Date)
man_meas$Date <- convertToDate(man_meas$Date)
man_meas$Time <- convertToDateTime(man_meas$Time)
date(man_meas$Time) <- man_meas$Date
man_meas$Location <- trimws(man_meas$Location)  # guard against stray whitespace from manual entry

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
