
### THIS IS NOW ALL IN THE "WORK WITH ALL ENVIRONMENTAL COMPILATIONS" FILE

library(openxlsx)
library(lubridate)
library(ggpubr)
library(tidyverse)
library(plotly)

rm(list = ls())

setwd("C:/Users/ShuttleBox//Seafile/Uni/01 Research Projects/03 Uganda 2023 EOD loggers/01 Data/DO Logs PyroScience/3. Curtailed Files/")


pyro_files <- list.files(pattern = "*.xlsx", recursive = F)

all_pyro_data <- NULL
for(i in 1:length(pyro_files)){
  d <- read.xlsx(pyro_files[i])
  
  d$Datetime <- convertToDateTime(d$Datetime)
  
  # Round Datetime to nearest 5min
  d$Datetime <- round_date(d$Datetime, unit = "5 minutes")
  
  d$Site <- strsplit(strsplit(pyro_files[i], "_")[[1]][2], "\\.")[[1]][1]
  d$System <- substr(d$Site[1], 1, 2)
  
  all_pyro_data <- rbind.data.frame(all_pyro_data, d)
}


# Add fixed Datetime for overlay plots
fixdate <- as.Date("2023-11-15")
all_pyro_data$Datetime_fix <- all_pyro_data$Datetime
all_pyro_data$Datetime_fix_loop <- all_pyro_data$Datetime
date(all_pyro_data$Datetime_fix) <- fixdate
date(all_pyro_data$Datetime_fix_loop) <- fixdate

deployments <- unique(all_pyro_data$Deployment)
for(i in 1:length(deployments)){
  diff_days <- all_pyro_data$Datetime_fix[all_pyro_data$Deployment == deployments[i]][1] - all_pyro_data$Datetime[all_pyro_data$Deployment == deployments[i]][1]
  all_pyro_data$Datetime_fix[all_pyro_data$Deployment == deployments[i]] <- all_pyro_data$Datetime[all_pyro_data$Deployment == deployments[i]] + diff_days
}


# Add day/night variable
t_sunset <- as.POSIXct("2023-11-15 18:37")
t_sunrise <- as.POSIXct("2023-11-15 06:30")

all_pyro_data$Photoperiod <- "Day"
all_pyro_data$Photoperiod[all_pyro_data$Datetime_fix_loop < t_sunrise | all_pyro_data$Datetime_fix_loop > t_sunset] <- "Night"

# plot
plt <- ggplot(all_pyro_data, aes(x = Datetime_fix, y = DO_AS, col = Site, group = as.factor(Deployment)))+
  geom_line()+
  labs(x = 'Time of Day', y = '% Air Saturation', title = 'Dissolved Oxygen in the Lake Nabugabo Area')+
  xlim(as.POSIXct("2023-11-15 09:30:00"), as.POSIXct("2023-11-17 10:00:00"))+
  theme_classic()+
  theme(legend.position = "bottom")

ggplotly(plt)  

# ggexport(plt, filename = "DO overview.png", width = 1800, height = 1200, res = 200)


# save
setwd("C:/Users/ShuttleBox//Seafile/Uni/01 Research Projects/03 Uganda 2023 EOD loggers/01 Data/")
write.csv2(all_pyro_data, "DO_Pyro_All_Data.csv", row.names = F)

