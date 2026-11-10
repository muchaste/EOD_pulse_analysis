
library(openxlsx)
library(lubridate)
library(marelac)
library(dplyr)
library(ggpubr)

rm(list = ls())

# 1. Read in Metadata
loc_register <- read.xlsx("./01 Data/Recordings and Measurements.xlsx", sheet = "Recordings_overview")
timetable <- read.xlsx("./01 Data/Recordings and Measurements.xlsx", sheet = "Deployment_times")
timetable$Date <- convertToDate(timetable$Date)
timetable$Deployment <- convertToDateTime(timetable$Deployment)
timetable$Collection <- convertToDateTime(timetable$Collection)
tz(timetable$Deployment) <- "UTC"
tz(timetable$Collection) <- "UTC"

setwd("./01 Data/DO Logs PME miniDOT/Raw Files/")

logs <- list.dirs(full.names = T)
logs <- logs[2:length(logs)] # eliminate "." entry


single_logs <- list.files(pattern = "*.txt", recursive = T)
O2_molw <- unname(molweight("O2"))

all_miniDOT_data <- NULL

for(i in 1:length(logs)){
  # setwd("../1. Raw Files/")
  setwd(logs[i])
  single_logs <- list.files(pattern = "*.txt", recursive = F)
  site <- strsplit(logs[i], "_")[[1]][2]
  system <- substr(site, 1, 2)
  
  # Information about location has to be determined from the site
  if(site == "PLP"  | site == "SLP"){
    location <- "surface_edge"
  } else if(site == "PLS"){
    location <- "para_edge"
  } else if (site == "LN10"){
    location <- "vert_middle"
  } else {
    location <- loc_register$Placement[loc_register$Site == site]
    location <- switch(location,
                       "Orthogonal" = "ortho_edge",
                       "Parallel" = "para_edge",
                       "Point" = "vert_edge")
  }
  
  # Nested loop to concatenate the files for each day
  logs_day <- NULL
  for(j in 1:length(single_logs)){
    d <- read.table(single_logs[j], skip = 2, header = T, sep = ',', fileEncoding = "latin1") # had to add fileEncoding only on field laptop (?)
    d <- d[, c(1,3,4)]
    colnames(d) <- c("Datetime", "T_C", "DO_mg_l")
    d$Datetime <- as.POSIXct(d$Datetime, origin = "1970-01-01") + hours(2) # add two hours to correct for wrong time zone (GMT+1 to GMT+3)
    tz(d$Datetime) <- "UTC" # set timezone to UTC
    d$Datetime <- round_date(d$Datetime, unit = "5 minutes") # round datetime to nearest 5 minutes
    d$DO_AS <- 999
    # convert DO to AS
    for(k in 1:length(d$T_C)){
      O2_satconc <- unname(gas_satconc(S=0, 
                                       t=d$T_C[k], 
                                       P=0.89, 
                                       species = "O2"))
      O2_airsat <- d$DO_mg_l[k]/(O2_satconc*O2_molw/1000)*100
      d$DO_AS[k] <- O2_airsat
    }
    logs_day <- rbind.data.frame(logs_day, d)
  }
  logs_day$System <- system
  logs_day$Site <- site
  logs_day$Location <- location
  logs_day$Index <- i
  
  # Remove first 30 and last 5 minutes
  t_deploy <- timetable$Deployment[timetable$Date == date(logs_day$Datetime[1]) & timetable$Site == site] + minutes(30)
  t_collect <- timetable$Collection[timetable$Date == date(logs_day$Datetime[1]) & timetable$Site == site] - minutes(5)
  logs_day <- subset(logs_day, Datetime >= t_deploy & Datetime <= t_collect)
  
  all_miniDOT_data <- rbind.data.frame(all_miniDOT_data, logs_day)
  setwd("../../Raw Files/")
  
}

# 3. Add fix date timestamps for overlay plots ----------------------------

fixdate <- as.Date("2023-11-15") # my birthday, whatever

all_miniDOT_data$Datetime_fix <- all_miniDOT_data$Datetime # this one just starts at the fixdate
all_miniDOT_data$Datetime_fix_loop <- all_miniDOT_data$Datetime # this one starts at fixdate and loops when 24h are reached (to restart at the same date)

date(all_miniDOT_data$Datetime_fix) <- fixdate
date(all_miniDOT_data$Datetime_fix_loop) <- fixdate

sites <- unique(all_miniDOT_data$Site)
last_idx <- 0
for(i in 1:length(sites)){
  site_idc <- which(all_miniDOT_data$Site == sites[i])
  deployments <- unique(all_miniDOT_data$Index[site_idc])
  for(j in 1:length(deployments)){
    depl_idc <- which(all_miniDOT_data$Index[site_idc] == deployments[j]) + last_idx
    diff_days <- all_miniDOT_data$Datetime_fix[depl_idc][1] - all_miniDOT_data$Datetime[depl_idc][1]
    all_miniDOT_data$Datetime_fix[depl_idc] <- all_miniDOT_data$Datetime[depl_idc] + diff_days
  }
  last_idx <- max(site_idc)
}

# 4. Add daylight period
t_sunset <- as.POSIXct("2023-11-15 18:37", tz = "UTC")
t_sunrise <- as.POSIXct("2023-11-15 06:30", tz = "UTC")

all_miniDOT_data$Photoperiod <- "Day"
all_miniDOT_data$Photoperiod[all_miniDOT_data$Datetime_fix_loop < t_sunrise | all_miniDOT_data$Datetime_fix_loop > t_sunset] <- "Night"

# 5. Control plot
overlay_plot <- ggplot(all_miniDOT_data, aes(x = Datetime_fix, y = DO_AS, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

boxplot <- ggplot(all_miniDOT_data, aes(x = Photoperiod, y = DO_AS, fill = System))+
  geom_boxplot()+
  theme_classic()

timeline_plot <- ggplot(all_miniDOT_data, aes(x = Datetime, y = DO_AS, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

combined_plot <- ggarrange(overlay_plot, boxplot, ncol = 2, common.legend = T, legend = "bottom")


# 5. Save data
setwd("../")
write.csv2(all_miniDOT_data, "All_miniDOT_data.csv", row.names = F)
ggexport(combined_plot, filename = "MiniDOT overview plot.png", width = 3000, height = 2500, res = 200)
