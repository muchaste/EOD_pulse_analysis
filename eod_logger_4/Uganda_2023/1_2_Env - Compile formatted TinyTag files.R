
rm(list = ls())

library(openxlsx)
library(readxl)
library(lubridate)
library(ggpubr)

setwd("./01 Data/TinyTag/")


# 1. Read in timetable
timetable <- read.xlsx("../Recordings and Measurements.xlsx", sheet = "Deployment_times")
timetable$Date <- convertToDate(timetable$Date)
timetable$Deployment <- convertToDateTime(timetable$Deployment)
timetable$Collection <- convertToDateTime(timetable$Collection)
tz(timetable$Deployment) <- "UTC"
tz(timetable$Collection) <- "UTC"

str(timetable)

# 2. Read in, format and concatenate raw files
raw_files <- list.files(pattern = "*.xlsx", recursive = T)


all_tinytag_data <- NULL

for(i in 1:length(raw_files)){
  print(raw_files[i])
  d <- read_excel(raw_files[i], sheet = 1)
  
  # Round Datetime to nearest 5min
  d$Datetime <- round_date(d$Datetime, unit = "5 minutes")
  
  fname <- strsplit(raw_files[i], "/")[[1]][3]
  site <- strsplit(strsplit(fname, "_")[[1]][2], "\\.")[[1]][1]
  system <- substr(site, 1, 2)
  
  t_deploy <- timetable$Deployment[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] + minutes(30)
  t_collect <- timetable$Collection[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] - minutes(5)
  
  d <- subset(d, Datetime >= t_deploy & Datetime <= t_collect)
  
  # go through every column (minus datetime)
  for(j in 1:(ncol(d)-1)){
    d_sub <- d[,c(1,j+1)]
    location <- colnames(d_sub)[2]
    colnames(d_sub) <- c("Datetime", "T_C")
    d_sub$System <- system
    d_sub$Site <- site
    d_sub$Location <- location
    d_sub$Index <- i
    all_tinytag_data <- rbind.data.frame(all_tinytag_data, d_sub)
    
  }
  
}


# 3. Add fix date timestamps for overlay plots ----------------------------
fixdate <- as.Date("2023-11-15") # my birthday, whatever

all_tinytag_data$Datetime_fix <- all_tinytag_data$Datetime # this one just starts at the fixdate
all_tinytag_data$Datetime_fix_loop <- all_tinytag_data$Datetime # this one starts at fixdate and loops when 24h are reached (to restart at the same date)

date(all_tinytag_data$Datetime_fix) <- fixdate
date(all_tinytag_data$Datetime_fix_loop) <- fixdate

sites <- unique(all_tinytag_data$Site)

for(i in 1:length(sites)){
  locations <-  unique(all_tinytag_data$Location[all_tinytag_data$Site == sites[i]])
  for(j in 1:length(locations)){
    deployments <- unique(all_tinytag_data$Index[all_tinytag_data$Site == sites[i] & all_tinytag_data$Location == locations[j]])
    for(k in 1:length(deployments)){
      row_idc <- which(all_tinytag_data$Site == sites[i] & all_tinytag_data$Location == locations[j] & all_tinytag_data$Index == deployments[k])
      diff_days <- all_tinytag_data$Datetime_fix[row_idc][1] - all_tinytag_data$Datetime[row_idc][1]
      all_tinytag_data$Datetime_fix[row_idc] <- all_tinytag_data$Datetime[row_idc] + diff_days
    }
  }
}

# 4. Add daylight period
t_sunset <- as.POSIXct("2023-11-15 18:37", tz = "UTC")
t_sunrise <- as.POSIXct("2023-11-15 06:30", tz = "UTC")

all_tinytag_data$Photoperiod <- "Day"
all_tinytag_data$Photoperiod[all_tinytag_data$Datetime_fix_loop < t_sunrise | all_tinytag_data$Datetime_fix_loop > t_sunset] <- "Night"

# 5. Control plot
overlay_plot <- ggplot(all_tinytag_data, aes(x = Datetime_fix, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

timeline_plot <- ggplot(all_tinytag_data, aes(x = Datetime, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

combined_plot <- ggarrange(overlay_plot, timeline_plot, ncol = 2, common.legend = T, legend = "bottom")

# 5. Save data
write.csv2(all_tinytag_data, "All_tinytag_data.csv", row.names = F)
ggexport(combined_plot, filename = "TinyTag overview plot.png", width = 3000, height = 2500, res = 200)

