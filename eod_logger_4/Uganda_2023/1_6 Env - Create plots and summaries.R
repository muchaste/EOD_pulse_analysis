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
setwd("./01 Data/")

light_temp <- read.csv2("Temp_Light_All_Logger_Data.csv")
light_temp$Datetime <- as.POSIXct(light_temp$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
light_temp$Datetime_fix <- as.POSIXct(light_temp$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
light_temp$Datetime_fix_loop <- as.POSIXct(light_temp$Datetime_fix_loop, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

## 1.2 DO PyroScience logger
DO_pyro <- read.csv2("DO Logs PyroScience/All_pyroscience_data.csv")
DO_pyro$Datetime <- as.POSIXct(DO_pyro$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
DO_pyro$Datetime_fix <- as.POSIXct(DO_pyro$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
DO_pyro$Source <- "Pyro"

## 1.2 DO MiniDOT logger
DO_miniDOT <- read.csv2("DO Logs PME miniDOT/All_miniDOT_data.csv")
DO_miniDOT$Datetime <- as.POSIXct(DO_miniDOT$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
DO_miniDOT$Datetime_fix <- as.POSIXct(DO_miniDOT$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
DO_miniDOT$Source <- "miniDOT"

# 1.3 Combined DO data
DO_combined <- rbind.data.frame(DO_pyro, DO_miniDOT)

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


DO_data_means <- DO_combined %>%
  group_by(System, Datetime_fix) %>%
  summarise(DO_AS_mean = mean(DO_AS, na.rm=T),
            DO_AS_sd = sd(DO_AS, na.rm=T))

DO_combined %>%
  group_by(System) %>%
  summarise(mean(DO_AS, na.rm=T),
            sd(DO_AS, na.rm=T),
            range(DO_AS, na.rm=T),
            mean(T_C),
            sd(T_C),
            range(T_C)
  )

# 3. Timeline plot ----------------------------------------------------------


## 3.1 Define time points -------------------------------------------------
t_p_start <- as.POSIXct("2023-11-15 06:00",format="%Y-%m-%d %H:%M", tz = "UTC")
t_p_end <- as.POSIXct("2023-11-17 12:00",format="%Y-%m-%d %H:%M", tz = "UTC")

t_p_vec <- c(t_p_start, t_p_start + hours(12), t_p_start + hours(24), t_p_start + hours(36), t_p_start + hours(48))

## 3.2 Lake Nabugabo -----------------------------------------------------------
light_temp_LN <- subset(light_temp, System == "LN")
T_start_times_LN <- NULL
T_end_times_LN <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_LN$Index))){
  sub <- subset(light_temp_LN, Index == unique(light_temp_LN$Index)[i])
  T_start_times_LN <- c(T_start_times_LN, min(sub$Datetime_fix, na.rm = T))
  T_end_times_LN <- c(T_end_times_LN, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_LN_latest <- as.POSIXct(max(T_start_times_LN))
T_end_LN_earliest <- as.POSIXct(min(T_end_times_LN))

DO_combined_LN <- subset(DO_combined, System == "LN")

DO_start_times_LN <- NULL
DO_end_times_LN <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_combined_LN$Index))){
  sub <- subset(DO_combined_LN, Index == unique(DO_combined_LN$Index)[i])
  DO_start_times_LN <- c(DO_start_times_LN, min(sub$Datetime_fix, na.rm = T))
  DO_end_times_LN <- c(DO_end_times_LN, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_LN_latest <- as.POSIXct(max(DO_start_times_LN))
DO_end_LN_earliest <- as.POSIXct(min(DO_end_times_LN))

t_plt_LN <- ggplot(light_temp_LN, aes(x = Datetime_fix, y = T_C, group = Index))+
  geom_line(aes(group = Index), col = "firebrick", alpha = .8)+
  # stat_summary(data = subset(light_temp_LN, Datetime_fix >= mean(T_start_times_LN) & Datetime_fix <= mean(T_end_times_LN)),
  #              fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M", tz="PST"))+
  labs(x = "Time of day", y = "Temperature (°C)")+
  ylim(20, 32)+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
        axis.text = element_text(size = 18, colour = "black"))

DO_plt_LN <- ggplot(DO_combined_LN, aes(x = Datetime_fix, y = DO_AS, group = Index))+
  geom_line(col = "dodgerblue")+
  # stat_summary(data = subset(DO_combined_LN, Datetime_fix >= mean(DO_start_times_LN) & Datetime_fix <= mean(DO_end_times_LN)),
  #              aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

LN_env_plot <- ggarrange(t_plt_LN, DO_plt_LN, ncol = 2, common.legend = T, legend = "bottom")

## 3.3 Petro Lagoon --------------------------------------------------------
light_temp_PL <- subset(light_temp, System == "PL")
T_start_times_PL <- NULL
T_end_times_PL <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_PL$Index))){
  sub <- subset(light_temp_PL, Index == unique(light_temp_PL$Index)[i])
  T_start_times_PL <- c(T_start_times_PL, min(sub$Datetime_fix, na.rm = T))
  T_end_times_PL <- c(T_end_times_PL, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_PL_latest <- as.POSIXct(max(T_start_times_PL))
T_end_PL_earliest <- as.POSIXct(min(T_end_times_PL))

DO_combined_PL <- subset(DO_combined, System == "PL")
DO_start_times_PL <- NULL
DO_end_times_PL <- NULL

for(i in 1:length(unique(DO_combined_PL$Index))){
  sub <- subset(DO_combined_PL, Index == unique(DO_combined_PL$Index)[i])
  DO_start_times_PL <- c(DO_start_times_PL, min(sub$Datetime_fix, na.rm = T))
  DO_end_times_PL <- c(DO_end_times_PL, max(sub$Datetime_fix, na.rm = T))
}

DO_start_PL_latest <- as.POSIXct(max(DO_start_times_PL))
DO_end_PL_earliest <- as.POSIXct(min(DO_end_times_PL))

t_plt_PL <- ggplot(light_temp_PL, aes(x = Datetime_fix, y = T_C, group = Index))+
  geom_line(aes(group = Index), col = "firebrick", alpha = .8)+
  # stat_summary(data = subset(light_temp_PL, Datetime_fix >= mean(T_start_times_PL) & Datetime_fix <= mean(T_end_times_PL)),
  #              fun=mean, color = "black", geom="line", linewidth = 1)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  ylim(20, 32)+
  labs(x = "Time of day", y = "Temperature (°C)")+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

DO_plt_PL <- ggplot(DO_combined_PL, aes(x = Datetime_fix, y = DO_AS, group = Index))+
  geom_line(col = "dodgerblue")+
  # stat_summary(data = subset(DO_combined_PL, Datetime_fix >= mean(DO_start_times_PL) & Datetime_fix <= mean(DO_end_times_PL)),
  #              aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

PL_env_plot <- ggarrange(t_plt_PL, DO_plt_PL, ncol = 2, common.legend = T, legend = "bottom")

## 3.4 Snake Lagoon --------------------------------------------------------
light_temp_SL <- subset(light_temp, System == "SL")
T_start_times_SL <- NULL
T_end_times_SL <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_SL$Index))){
  sub <- subset(light_temp_SL, Index == unique(light_temp_SL$Index)[i])
  T_start_times_SL <- c(T_start_times_SL, min(sub$Datetime_fix, na.rm = T))
  T_end_times_SL <- c(T_end_times_SL, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}


T_start_SL_latest <- as.POSIXct(max(T_start_times_SL))
T_end_SL_earliest <- as.POSIXct(min(T_end_times_SL))

DO_combined_SL <- subset(DO_combined, System == "SL")

DO_start_times_SL <- NULL
DO_end_times_SL <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_combined_SL$Index))){
  sub <- subset(DO_combined_SL, Index == unique(DO_combined_SL$Index)[i])
  DO_start_times_SL <- c(DO_start_times_SL, min(sub$Datetime_fix, na.rm = T))
  DO_end_times_SL <- c(DO_end_times_SL, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_SL_latest <- as.POSIXct(max(DO_start_times_SL))
DO_end_SL_earliest <- as.POSIXct(min(DO_end_times_SL))

t_plt_SL <- ggplot(light_temp_SL, aes(x = Datetime_fix, y = T_C, group = Index))+
  geom_line(aes(group = Index), col = "firebrick", alpha = .8)+
  # stat_summary(data = subset(light_temp_SL, Datetime_fix >= mean(T_start_times_SL) & Datetime_fix <= mean(T_end_times_SL)),
  #              fun=mean, color = "black", geom="line", linewidth = 1)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  ylim(20, 32)+
  labs(x = "Time of day", y = "Temperature (°C)")+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

DO_plt_SL <- ggplot(DO_combined_SL, aes(x = Datetime_fix, y = DO_AS, group = Index))+
  geom_line(col = "dodgerblue")+
  # stat_summary(data = subset(DO_combined_SL, Datetime_fix >= mean(DO_start_times_SL) & Datetime_fix <= mean(DO_end_times_SL)),
  #              aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

SL_env_plot <- ggarrange(t_plt_SL, DO_plt_SL, ncol = 2, common.legend = T, legend = "bottom")



## 3.5 Juma River ----------------------------------------------------------

light_temp_JR <- subset(light_temp, System == "JR")
T_start_times_JR <- NULL
T_end_times_JR <- NULL
# durations <- NULL
for(i in 1:length(unique(light_temp_JR$Index))){
  sub <- subset(light_temp_JR, Index == unique(light_temp_JR$Index)[i])
  T_start_times_JR <- c(T_start_times_JR, min(sub$Datetime_fix, na.rm = T))
  T_end_times_JR <- c(T_end_times_JR, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

T_start_JR_latest <- as.POSIXct(max(T_start_times_JR))
T_end_JR_earliest <- as.POSIXct(min(T_end_times_JR))

DO_combined_JR <- subset(DO_combined, System == "JR")

DO_start_times_JR <- NULL
DO_end_times_JR <- NULL
# durations <- NULL
for(i in 1:length(unique(DO_combined_JR$Index))){
  sub <- subset(DO_combined_JR, Index == unique(DO_combined_JR$Index)[i])
  DO_start_times_JR <- c(DO_start_times_JR, min(sub$Datetime_fix, na.rm = T))
  DO_end_times_JR <- c(DO_end_times_JR, max(sub$Datetime_fix, na.rm = T))
  # durations <- c(durations, (max(sub$std_time)-min(sub$std_time)))
}

DO_start_JR_latest <- as.POSIXct(max(DO_start_times_JR))
DO_end_JR_earliest <- as.POSIXct(min(DO_end_times_JR))

t_plt_JR <- ggplot(light_temp_JR, aes(x = Datetime_fix, y = T_C, group = Index))+
  geom_line(aes(group = Index), col = "firebrick", alpha = 0.8)+
  # stat_summary(data = subset(light_temp_JR, Datetime_fix >= mean(T_start_times_JR) & Datetime_fix <= mean(T_end_times_JR)),
  #              fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  ylim(20, 32)+
  labs(x = "Time of day", y = "Temperature (°C)")+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

DO_plt_JR <- ggplot(DO_combined_JR, aes(x = Datetime_fix, y = DO_AS, group = Index))+
  geom_line(col = "dodgerblue")+
  # stat_summary(data = subset(DO_combined_JR, Datetime_fix >= mean(DO_start_times_JR) & Datetime_fix <= mean(DO_end_times_JR)),
  #              aes(group = 1), fun=mean, color = "black", geom="line", linewidth = 1)+
  # facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   # breaks = date_breaks("12 hours"),
                   breaks = t_p_vec+hours(1),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")+
  ylim(-5,125)+
  theme_bw(base_size = 18)+
  theme(#axis.title = element_blank(),
    axis.text = element_text(size = 18, colour = "black"))

JR_env_plot <- ggarrange(t_plt_JR, DO_plt_JR, ncol = 2, common.legend = T, legend = "bottom")

setwd("../03 Analysis Output/")

ggexport(LN_env_plot, filename = "LN_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(PL_env_plot, filename = "PL_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(SL_env_plot, filename = "SL_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(JR_env_plot, filename = "JR_env_data.png", height = 1400, width = 3000, res = 300)

# 4. Boxplots -------------------------------------------------------------
# One Boxplot per system and daylight period (day/night) for temperature and DO saturation
t_box_all <- ggplot(light_temp, aes(x = System, y = T_C, fill = Photoperiod))+
  geom_boxplot()+
  labs(x = "System", y = "Temperature (°C)")+
  theme_classic()

DO_box_all <- ggplot(DO_combined, aes(x = System, y = DO_AS, fill = Photoperiod))+
  geom_boxplot()+
  labs(x = "System", y = "Air saturation (%)")+
  theme_classic()

pH_box_all <- ggplot(MM_dat, aes(x = System, y = pH))+
  geom_boxplot(fill = "lightpink")+
  labs(x = "System", y = "pH")+
  theme_classic()

cond_box_all <- ggplot(MM_dat, aes(x = System, y = Cond_uS))+
  geom_boxplot(fill = "lightpink")+
  labs(x = "System", y = "Conductivity (µS)")+
  theme_classic()

turb_box_all <- ggplot(MM_dat, aes(x = System, y = Turbidity_NTU))+
  geom_boxplot(fill = "lightpink")+
  labs(x = "System", y = "Turbidity (NTU)")+
  theme_classic()

t_box_LN <- ggplot(light_temp_LN, aes(x = Photoperiod, y = T_C, fill = Photoperiod))+
  geom_boxplot()+
  labs(x = "Photoperiod", y = "Temperature (°C)")+
  theme_classic()

DO_box_LN <- ggplot(DO_combined_LN, aes(x = Photoperiod, y = DO_AS, fill = Photoperiod))+
  geom_boxplot()+
  labs(x = "Photoperiod", y = "Air saturation (%)")+
  theme_classic()

overview_box <- ggarrange(t_box_all, DO_box_all, pH_box_all, cond_box_all, turb_box_all,
 ncol = 2, nrow = 3, common.legend = T, legend = "bottom")

ggexport(overview_box, filename = "Overview_boxplots.png", height = 3000, width = 3000, res = 300)

# 5. Summary statistics -------------------------------------------------------------
# Add session date to light_temp and DO_combined per Index
light_temp <- light_temp %>%
  group_by(System, Site, Index) %>%
  mutate(Session_date = min(date(Datetime), na.rm=T))

DO_combined <- DO_combined %>%
  group_by(System, Site, Index) %>%
  mutate(Session_date = min(date(Datetime), na.rm=T))

MM_dat$Session_date <- MM_dat$Date

# Summarize per site and deployment/index/session
light_temp_summary <- light_temp %>%
  group_by(System, Site, Session_date) %>%
  summarise(Lux_mean = mean(Lux, na.rm=T),
            Lux_sd = sd(Lux, na.rm=T),
            Lux_min = min(Lux, na.rm=T),
            Lux_max = max(Lux, na.rm=T),
            T_C_mean = mean(T_C, na.rm=T),
            T_C_sd = sd(T_C, na.rm=T),
            T_C_min = min(T_C, na.rm=T),
            T_C_max = max(T_C, na.rm=T))

DO_summary <- DO_combined %>%
  group_by(System, Site, Index) %>%
  summarise(Session_date = min(date(Datetime), na.rm=T),
            DO_AS_mean = mean(DO_AS, na.rm=T),
            DO_AS_sd = sd(DO_AS, na.rm=T),
            DO_AS_min = min(DO_AS, na.rm=T),
            DO_AS_max = max(DO_AS, na.rm=T),
            T_C_DO_mean = mean(T_C, na.rm=T),
            T_C_DO_sd = sd(T_C, na.rm=T),
            T_C_DO_min = min(T_C, na.rm=T),
            T_C_DO_max = max(T_C, na.rm=T))

MM_summary <- MM_dat %>%
  group_by(System, Site, Session_date) %>%
  summarise(pH_mean = mean(pH, na.rm=T),
            pH_sd = sd(pH, na.rm=T),
            pH_min = min(pH, na.rm=T),
            pH_max = max(pH, na.rm=T),
            Cond_mean = mean(Cond_uS, na.rm=T),
            Cond_sd = sd(Cond_uS, na.rm=T),
            Cond_min = min(Cond_uS, na.rm=T),
            Cond_max = max(Cond_uS, na.rm=T),
            Turbidity_mean = mean(Turbidity_NTU, na.rm=T),
            Turbidity_sd = sd(Turbidity_NTU, na.rm=T),
            Turbidity_min = min(Turbidity_NTU, na.rm=T),
            Turbidity_max = max(Turbidity_NTU, na.rm=T),
            T_C_MM_mean = mean(T_C, na.rm=T),
            T_C_MM_sd = sd(T_C, na.rm=T),
            T_C_MM_min = min(T_C, na.rm=T),
            T_C_MM_max = max(T_C, na.rm=T),
            DO_AS_MM_mean = mean(DO_AS, na.rm=T),
            DO_AS_MM_sd = sd(DO_AS, na.rm=T),
            DO_AS_MM_min = min(DO_AS, na.rm=T),
            DO_AS_MM_max = max(DO_AS, na.rm=T))

complete_summary <- merge(light_temp_summary, DO_summary, by = c("System", "Site", "Session_date"), all = T)
complete_summary <- merge(complete_summary, MM_summary, by = c("System", "Site", "Session_date"), all = T)


write.csv2(complete_summary, "Complete_env_data_summary_per_day.csv", row.names = F)
