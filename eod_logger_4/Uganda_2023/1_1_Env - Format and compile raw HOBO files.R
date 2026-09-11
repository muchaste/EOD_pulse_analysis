
rm(list = ls())

library(openxlsx)
library(readxl)
library(lubridate)
library(ggpubr)

setwd("./01 Data/HOBO loggers/")


# 1. Read in timetable
timetable <- read.xlsx("../Recordings and Measurements.xlsx", sheet = "Deployment_times")
timetable$Date <- convertToDate(timetable$Date)
timetable$Deployment <- convertToDateTime(timetable$Deployment)
timetable$Collection <- convertToDateTime(timetable$Collection)
tz(timetable$Deployment) <- "UTC"
tz(timetable$Collection) <- "UTC"

str(timetable)

# 2. Read in, format and concatenate raw files
raw_files <- list.files(pattern = ".xlsx", recursive = T)

# Remove files with a "faulty" somewhere in the name
raw_files <- raw_files[!grepl("faulty", raw_files)]

all_hobo_data <- NULL
for(i in 1:length(raw_files)){
  print(raw_files[i])
  d <- read_excel(raw_files[i], sheet = 1, skip=1)
  
  fname <- strsplit(raw_files[i], "/")[[1]][2]
  site <- strsplit(fname, "_")[[1]][1]
  system <- substr(site, 1, 2)
  location <- paste(strsplit(fname, "_")[[1]][2:3], collapse="_")
  
  d <- d[,2:4]
  colnames(d) <- c("Datetime", "T_C", "Lux")
  
  # Round Datetime to nearest 5min
  d$Datetime <- round_date(d$Datetime, unit = "5 minutes")
  
  d$System <- system
  d$Site <- site
  d$Location <- location
  d$Index <- i
  
  # Remove first 30 and last 5 minutes
  t_deploy <- timetable$Deployment[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] + minutes(30)
  t_collect <- timetable$Collection[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] - minutes(5)
  d <- subset(d, Datetime >= t_deploy & Datetime <= t_collect)
  
  all_hobo_data <- rbind.data.frame(all_hobo_data, d)
}

# 3. Add fix date timestamps for overlay plots ----------------------------

fixdate <- as.Date("2023-11-15") # my birthday, whatever

all_hobo_data$Datetime_fix <- all_hobo_data$Datetime # this one just starts at the fixdate
all_hobo_data$Datetime_fix_loop <- all_hobo_data$Datetime # this one starts at fixdate and loops when 24h are reached (to restart at the same date)

date(all_hobo_data$Datetime_fix) <- fixdate
date(all_hobo_data$Datetime_fix_loop) <- fixdate

sites <- unique(all_hobo_data$Site)
last_idx <- 0
for(i in 1:length(sites)){
  site_idc <- which(all_hobo_data$Site == sites[i])
  deployments <- unique(all_hobo_data$Index[site_idc])
  for(j in 1:length(deployments)){
    depl_idc <- which(all_hobo_data$Index[site_idc] == deployments[j]) + last_idx
    diff_days <- all_hobo_data$Datetime_fix[depl_idc][1] - all_hobo_data$Datetime[depl_idc][1]
    all_hobo_data$Datetime_fix[depl_idc] <- all_hobo_data$Datetime[depl_idc] + diff_days
  }
  last_idx <- max(site_idc)
}

# 4. Add daylight period
t_sunset <- as.POSIXct("2023-11-15 18:37", tz = "UTC")
t_sunrise <- as.POSIXct("2023-11-15 06:30", tz = "UTC")

all_hobo_data$Photoperiod <- "Day"
all_hobo_data$Photoperiod[all_hobo_data$Datetime_fix_loop < t_sunrise | all_hobo_data$Datetime_fix_loop > t_sunset] <- "Night"

# 5. Control plot
overlay_plot <- ggplot(all_hobo_data, aes(x = Datetime_fix, y = Lux, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

timeline_plot <- ggplot(all_hobo_data, aes(x = Datetime, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

combined_plot <- ggarrange(overlay_plot, timeline_plot, ncol = 2, common.legend = T, legend = "bottom")

# 5. Save data
write.csv2(all_hobo_data, "All_hobo_data.csv", row.names = F)
ggexport(combined_plot, filename = "HOBO overview plot.png", width = 3000, height = 2500, res = 200)
