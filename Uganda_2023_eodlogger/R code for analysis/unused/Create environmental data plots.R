library(lubridate)
library(tidyverse)
library(ggpubr)
library(openxlsx)
library(plotly)
library(scales)

rm(list=ls())

# # PC <- "field laptop"
# PC <- "thinkpad"
# # PC <- "lab"
# 
# 
# # 0. Set working directories ----------------------------------------------
# if(PC == "field laptop"){
#   root <- "C:/Users/Stefan Mucha/Desktop/Nab_2023_Field_Data/HOBO loggers/"
# } else if(PC == "thinkpad"){
#   root <-"C:/Users/stefa/Seafile/Uni/"
# } else if(PC == "office"){
#   root <- "C:/Users/ShuttleBox/Seafile/Uni/"
# } else if(PC == "lab"){
#   root <- "D:/Seafile/Uni/"
# }



# 1. Read in data and format timestamp ------------------------------------

## 1.1 Light and temp data 

# setwd(root)
# if(PC != "field laptop"){
#   setwd("./01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/")
# }
setwd("./01 Data/")

light_temp <- read.csv2("Temp_Light_All_Logger_Data.csv")
light_temp <- subset(light_temp, System == "LN" | System == "PL" | System == "JR")
light_temp$Datetime <- as.POSIXct(light_temp$Datetime, format = "%Y-%m-%d %H:%M:%S")
light_temp$Datetime_fix <- as.POSIXct(light_temp$Datetime_fix, format = "%Y-%m-%d %H:%M:%S")
light_temp$Datetime_fix_loop <- as.POSIXct(light_temp$Datetime_fix_loop, format = "%Y-%m-%d %H:%M:%S")

## 1.2 DO PyroScience logger

# setwd(datadir)
DO_pyro <- read.csv2("DO_Pyro_All_Data.csv")
DO_pyro <- subset(DO_pyro, System == "LN" | System == "PL" | System == "JR")
DO_pyro$Datetime <- as.POSIXct(DO_pyro$Datetime, format = "%Y-%m-%d %H:%M:%S")
DO_pyro$Datetime_fix <- as.POSIXct(DO_pyro$Datetime_fix, format = "%Y-%m-%d %H:%M:%S")


# Manual Measurements
# setwd(datadir)
MM_dat <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")
MM_dat$Date <- convertToDate(MM_dat$Date)
MM_dat$Datetime <- convertToDateTime(MM_dat$Time)
date(MM_dat$Datetime) <- MM_dat$Date


# 2. Calculate mean values per site and location --------------------------
light_temp_means <- light_temp %>%
  group_by(System, Datetime_fix) %>%
  summarise(Lux_mean = mean(Lux, na.rm=T),
            Lux_sd = sd(Lux, na.rm=T),
            T_C_mean = mean(T_C, na.rm=T),
            T_C_sd = sd(T_C, na.rm=T))


DO_data_means <- DO_pyro %>%
  group_by(System, Datetime_fix) %>%
  summarise(DO_AS_mean = mean(DO_AS, na.rm=T),
            DO_AS_sd = sd(DO_AS, na.rm=T))

DO_pyro %>%
  group_by(System) %>%
  summarise(mean(DO_AS, na.rm=T),
            sd(DO_AS, na.rm=T),
            range(DO_AS, na.rm=T),
            mean(T_C),
            sd(T_C),
            range(T_C)
  )

# 3. Plot -----------------------------------------------------------------


## 3.1 Define time points -------------------------------------------------

t_p_start <- as.POSIXct("2023-11-15 06:00",format="%Y-%m-%d %H:%M")
t_p_end <- as.POSIXct("2023-11-17 12:00",format="%Y-%m-%d %H:%M")

t_p_vec <- c(t_p_start, t_p_start + hours(12), t_p_start + hours(24), t_p_start + hours(36), t_p_start + hours(48))

# t_times_LN <- matrix(nrow = length(unique(light_temp$Index)), ncol = 2)
# t_times_PL <- matrix(nrow = length(unique(light_temp$Index)), ncol = 2)

### Lake Nabugabo

light_temp_LN <- subset(light_temp, System == "LN")
T_start_times_LN <- NULL
T_end_times_LN <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_LN$Index))){
  sub <- subset(light_temp_LN, Index == unique(light_temp_LN$Index)[i])
  T_start_times_LN <- c(T_start_times_LN, min(sub$Datetime_fix))
  T_end_times_LN <- c(T_end_times_LN, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_LN_latest <- as.POSIXct(max(T_start_times_LN))
T_end_LN_earliest <- as.POSIXct(min(T_end_times_LN))

DO_pyro_LN <- subset(DO_pyro, System == "LN")

DO_start_times_LN <- NULL
DO_end_times_LN <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_pyro_LN$Deployment))){
  sub <- subset(DO_pyro_LN, Deployment == unique(DO_pyro_LN$Deployment)[i])
  DO_start_times_LN <- c(DO_start_times_LN, min(sub$Datetime_fix))
  DO_end_times_LN <- c(DO_end_times_LN, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_LN_latest <- as.POSIXct(max(DO_start_times_LN))
DO_end_LN_earliest <- as.POSIXct(min(DO_end_times_LN))

t_plt_LN <- ggplot(light_temp_LN, aes(x = Datetime_fix, y = T_C))+
  geom_line(aes(group = Index), col = "darkred", alpha = 0.2)+
  stat_summary(data = subset(light_temp_LN, Datetime_fix >= mean(T_start_times_LN) & Datetime_fix <= mean(T_end_times_LN)),
               fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Temperature (°C)")+
  ylim(20, 32)+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

DO_plt_LN <- ggplot(DO_pyro_LN, aes(x = Datetime_fix, y = DO_AS, group = Deployment))+
  geom_line(col = "darkblue", alpha = 0.8)+
  # stat_summary(data = subset(DO_pyro_LN, Datetime_fix >= mean(DO_start_times_LN) & Datetime_fix <= mean(DO_end_times_LN)),
  #              aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

LN_env_plot <- ggarrange(t_plt_LN, DO_plt_LN, ncol = 2, common.legend = T, legend = "bottom")


### Petro Lagoon
light_temp_PL <- subset(light_temp, System == "PL")
T_start_times_PL <- NULL
T_end_times_PL <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_PL$Index))){
  sub <- subset(light_temp_PL, Index == unique(light_temp_PL$Index)[i])
  T_start_times_PL <- c(T_start_times_PL, min(sub$Datetime_fix))
  T_end_times_PL <- c(T_end_times_PL, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_PL_latest <- as.POSIXct(max(T_start_times_PL))
T_end_PL_earliest <- as.POSIXct(min(T_end_times_PL))

DO_pyro_PL <- subset(DO_pyro, System == "PL")

DO_start_times_PL <- NULL
DO_end_times_PL <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_pyro_PL$Deployment))){
  sub <- subset(DO_pyro_PL, Deployment == unique(DO_pyro_PL$Deployment)[i])
  DO_start_times_PL <- c(DO_start_times_PL, min(sub$Datetime_fix))
  DO_end_times_PL <- c(DO_end_times_PL, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_PL_latest <- as.POSIXct(max(DO_start_times_PL))
DO_end_PL_earliest <- as.POSIXct(min(DO_end_times_PL))

t_plt_PL <- ggplot(light_temp_PL, aes(x = Datetime_fix, y = T_C))+
  geom_line(aes(group = Index), col = "darkred", alpha = 0.2)+
  stat_summary(data = subset(light_temp_PL, Datetime_fix >= mean(T_start_times_PL) & Datetime_fix <= mean(T_end_times_PL)),
               fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  ylim(20, 32)+
  labs(x = "Time of day", y = "Temperature (°C)")+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

DO_plt_PL <- ggplot(DO_pyro_PL, aes(x = Datetime_fix, y = DO_AS, group = Deployment))+
  geom_line(col = "darkblue", alpha = 0.4)+
  stat_summary(data = subset(DO_pyro_PL, Datetime_fix >= mean(DO_start_times_PL) & Datetime_fix <= mean(DO_end_times_PL)),
               aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

PL_env_plot <- ggarrange(t_plt_PL, DO_plt_PL, ncol = 2, common.legend = T, legend = "bottom")

### Juma River
light_temp_JR <- subset(light_temp, System == "JR")
T_start_times_JR <- NULL
T_end_times_JR <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_JR$Index))){
  sub <- subset(light_temp_JR, Index == unique(light_temp_JR$Index)[i])
  T_start_times_JR <- c(T_start_times_JR, min(sub$Datetime_fix))
  T_end_times_JR <- c(T_end_times_JR, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_JR_latest <- as.POSIXct(max(T_start_times_JR))
T_end_JR_earliest <- as.POSIXct(min(T_end_times_JR))

DO_pyro_JR <- subset(DO_pyro, System == "JR")

DO_start_times_JR <- NULL
DO_end_times_JR <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_pyro_JR$Deployment))){
  sub <- subset(DO_pyro_JR, Deployment == unique(DO_pyro_JR$Deployment)[i])
  DO_start_times_JR <- c(DO_start_times_JR, min(sub$Datetime_fix))
  DO_end_times_JR <- c(DO_end_times_JR, max(sub$Datetime_fix))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_JR_latest <- as.POSIXct(max(DO_start_times_JR))
DO_end_JR_earliest <- as.POSIXct(min(DO_end_times_JR))

t_plt_JR <- ggplot(light_temp_JR, aes(x = Datetime_fix, y = T_C))+
  geom_line(aes(group = Index), col = "darkred", alpha = 0.2)+
  stat_summary(data = subset(light_temp_JR, Datetime_fix >= mean(T_start_times_JR) & Datetime_fix <= mean(T_end_times_JR)),
               fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  ylim(20, 32)+
  labs(x = "Time of day", y = "Temperature (°C)")+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

DO_plt_JR <- ggplot(DO_pyro_JR, aes(x = Datetime_fix, y = DO_AS, group = Deployment))+
  geom_line(col = "darkblue", alpha = 0.4)+
  stat_summary(data = subset(DO_pyro_JR, Datetime_fix >= mean(DO_start_times_JR) & Datetime_fix <= mean(DO_end_times_JR)),
               aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_classic()+
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

JR_env_plot <- ggarrange(t_plt_JR, DO_plt_JR, ncol = 2, common.legend = T, legend = "bottom")

setwd("../02 Analysis Output/")

ggexport(LN_env_plot, filename = "LN_env_data.png", height = 1000, width = 3000, res = 300)
ggexport(PL_env_plot, filename = "PL_env_data.png", height = 1000, width = 3000, res = 300)
ggexport(JR_env_plot, filename = "JR_env_data.png", height = 1000, width = 3000, res = 300)
