library(openxlsx)
library(lubridate)

rm(list = ls())


setwd("C:/Users/stefa/Seafile/Uni/01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/")

metadata <- read.xlsx("Recordings_Sessions_Metadata.xlsx")
man_meas <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")

metadata$t_meas_set <- NA
metadata$t_meas_retr <- NA

# add manual measurements from setting and retrieving the loggers to the metadata

metadata$Start_Date <- convertToDate(metadata$Start_Date)
metadata$End_Date <- convertToDate(metadata$End_Date)
man_meas$Date <- convertToDate(man_meas$Date)

full_metadata <- NULL

for(logger_id in unique(metadata$Logger_ID)){
  logger_sub <- subset(metadata, Logger_ID == logger_id)
  for(session in unique(logger_sub$Start_Date)){
    session_sub <- subset(logger_sub, Start_Date == session)
    if (length(unique(session_sub$Site)) > 1){
      print("error - multiple sites with same logger id and date XXX")
      break
    }
    site <- session_sub$Site[1]
    mm_sub <- subset(man_meas, Site == site & Date == session &)
  }
}