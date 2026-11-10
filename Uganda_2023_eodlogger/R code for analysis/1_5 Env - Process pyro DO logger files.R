library(openxlsx)
library(lubridate)
library(ggpubr)
library(marelac)

rm(list = ls())

O2_molw <- unname(molweight("O2"))

timetable <- read.xlsx("./01 Data/Recordings and Measurements.xlsx", sheet = "Deployment_times")
site_register <- read.xlsx("./01 Data/Recordings and Measurements.xlsx", sheet = "Sampling_trips_overview")
loc_register <- read.xlsx("./01 Data/Recordings and Measurements.xlsx", sheet = "Recordings_overview")

site_register$Date <- convertToDate(site_register$Date)
timetable$Date <- convertToDate(timetable$Date)
timetable$Deployment <- convertToDateTime(timetable$Deployment)
timetable$Collection <- convertToDateTime(timetable$Collection)
tz(timetable$Deployment) <- "UTC"
tz(timetable$Collection) <- "UTC"
# timetable$Site[timetable$Site == "JR1" | timetable$Site == "JR2"] <- "JR"

setwd("./01 Data/DO Logs PyroScience/1. Pyro Workbench Output/")

single_logs <- list.files(pattern = "*.txt", recursive = F)

all_PyroScience_data <- NULL

for(i in 1:length(single_logs)){
  # setwd("../1. Pyro Workbench Output/")
  d <- read.table(single_logs[i], skip = 35, header = T, sep = '\t', fileEncoding = "latin1") # had to add fileEncoding only on field laptop (?)
  d <- d[, c(1,9,6,8)]
  colnames(d) <- c("Datetime","T_C","DO_mg_l", "DO_AS")
  d$DO_mg_l <- (d$DO_mg_l/1000000)*O2_molw # convert from nmol to mg
  d$DO_AS <- d$DO_AS/1000
  d$T_C <- d$T_C/1000
  d$Datetime <- as.POSIXct(d$Datetime)
  tz(d$Datetime) <- "UTC" # set timezone to UTC
  d$Datetime <- round_date(d$Datetime, unit = "5 minutes") # round datetime to nearest 5 minutes
  
  system <- substr(single_logs[i], 1, 2)
  site <- site_register$DO_logger_PS_loc[site_register$Date == date(d$Datetime[1])]
  
  # Information about location has to be determined from the site
  if(site == "PLP"  | site == "SLP"){
    location <- "surface_edge"
  } else if(site == "PLS"){
    location <- "para_edge"
  } else if (site == "LN10_top"){
    location <- "vert_edge"
  } else if (site == "JR"){
    location <- "ortho_edge"
  } else {
    location <- loc_register$Placement[loc_register$Site == site]
    if (length(location) > 1){
      location <- location[1]
    }
    location <- switch(location,
                       "Orthogonal" = "ortho_edge",
                       "Parallel" = "para_edge",
                       "Point" = "vert_edge")
  }
  
  d$System <- system
  d$Site <- site
  d$Location <- location
  d$Index <- i
  
  # Remove first 30 and last 5 minutes
  t_deploy <- timetable$Deployment[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] + minutes(30)
  t_collect <- timetable$Collection[timetable$Date == date(d$Datetime[1]) & timetable$Site == site] - minutes(5)
  d <- subset(d, Datetime >= t_deploy & Datetime <= t_collect)
  
  all_PyroScience_data <- rbind.data.frame(all_PyroScience_data, d)
}

# 3. Add fix date timestamps for overlay plots ----------------------------

fixdate <- as.Date("2023-11-15") # my birthday, whatever

all_PyroScience_data$Datetime_fix <- all_PyroScience_data$Datetime # this one just starts at the fixdate
all_PyroScience_data$Datetime_fix_loop <- all_PyroScience_data$Datetime # this one starts at fixdate and loops when 24h are reached (to restart at the same date)

date(all_PyroScience_data$Datetime_fix) <- fixdate
date(all_PyroScience_data$Datetime_fix_loop) <- fixdate

sites <- unique(all_PyroScience_data$Site)
last_idx <- 0
for(i in 1:length(sites)){
  site_idc <- which(all_PyroScience_data$Site == sites[i])
  deployments <- unique(all_PyroScience_data$Index[site_idc])
  for(j in 1:length(deployments)){
    depl_idc <- which(all_PyroScience_data$Index[site_idc] == deployments[j]) + last_idx
    diff_days <- all_PyroScience_data$Datetime_fix[depl_idc][1] - all_PyroScience_data$Datetime[depl_idc][1]
    all_PyroScience_data$Datetime_fix[depl_idc] <- all_PyroScience_data$Datetime[depl_idc] + diff_days
  }
  last_idx <- max(site_idc)
}

# 4. Add daylight period
t_sunset <- as.POSIXct("2023-11-15 18:37", tz = "UTC")
t_sunrise <- as.POSIXct("2023-11-15 06:30", tz = "UTC")

all_PyroScience_data$Photoperiod <- "Day"
all_PyroScience_data$Photoperiod[all_PyroScience_data$Datetime_fix_loop < t_sunrise | all_PyroScience_data$Datetime_fix_loop > t_sunset] <- "Night"

# 5. Control plot
overlay_plot <- ggplot(all_PyroScience_data, aes(x = Datetime_fix, y = DO_AS, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

boxplot <- ggplot(all_PyroScience_data, aes(x = Photoperiod, y = DO_AS, fill = System))+
  geom_boxplot()+
  theme_classic()

timeline_plot <- ggplot(all_PyroScience_data, aes(x = Datetime, y = DO_AS, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

combined_plot <- ggarrange(overlay_plot, boxplot, ncol = 2, common.legend = T, legend = "bottom")


# 5. Save data
setwd("../")
write.csv2(all_PyroScience_data, "All_pyroscience_data.csv", row.names = F)
ggexport(combined_plot, filename = "PyroScience overview plot.png", width = 3000, height = 2500, res = 200)

