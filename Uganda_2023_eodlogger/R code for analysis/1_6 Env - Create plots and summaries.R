library(lubridate)
library(tidyverse)
library(ggpubr)
library(openxlsx)
library(plotly)
library(scales)
library(suncalc)
library(jsonlite)

rm(list=ls())

TZ <- "Africa/Nairobi"

# 1. Read in data and format timestamp ------------------------------------

## 1.1 Light and temp data 
light_temp <- read.csv2("01 Data/Environmental Data/Temp_Light_All_Logger_Data.csv")
light_temp$Datetime <- as.POSIXct(light_temp$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
light_temp$Datetime_fix <- as.POSIXct(light_temp$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
light_temp$Datetime_fix_loop <- as.POSIXct(light_temp$Datetime_fix_loop, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")

## 1.2 DO PyroScience logger
DO_pyro <- read.csv2("01 Data/Environmental Data/DO Logs PyroScience/All_pyroscience_data.csv")
DO_pyro$Datetime <- as.POSIXct(DO_pyro$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_pyro$Datetime_fix <- as.POSIXct(DO_pyro$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_pyro$Source <- "Pyro"

## 1.2 DO MiniDOT logger
DO_miniDOT <- read.csv2("01 Data/Environmental Data/DO Logs PME miniDOT/All_miniDOT_data.csv")
DO_miniDOT$Datetime <- as.POSIXct(DO_miniDOT$Datetime, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_miniDOT$Datetime_fix <- as.POSIXct(DO_miniDOT$Datetime_fix, format = "%Y-%m-%d %H:%M:%S", tz = "Africa/Nairobi")
DO_miniDOT$Source <- "miniDOT"

# 1.3 Combined DO data
DO_combined <- rbind.data.frame(DO_pyro, DO_miniDOT)

# 1.4 Manual Measurements
MM_dat <- read.csv2("01 Data/Environmental Data/Manual Measurements.csv", sep = ",", dec = ".", quote = "\"")
# MM_dat <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")
MM_dat$Date <- as.Date(MM_dat$Date, format = "%d.%m.%Y")
MM_dat$Session_date <- as.Date(MM_dat$Session_date, format = "%d.%m.%Y")
MM_dat$Time <- as.POSIXct(MM_dat$Time, format = "%H:%M", tz = "Africa/Nairobi")
date(MM_dat$Time) <- MM_dat$Date
MM_dat$Location <- trimws(MM_dat$Location)  # guard against stray whitespace from manual entry
MM_dat$Logger_ID <- strsplit(MM_dat$Location, "-") %>% sapply(`[`, 1)  # Extract logger ID from Location column


# 1.5 Logger ID and Location Key
ID_loc_key <- read.csv2("01 Data/Logger_ID_Location_Key.csv", sep = ",", dec = ".", quote = "\"")
ID_loc_key$Session_date <- as.Date(ID_loc_key$Session_date, format = "%d.%m.%Y")
# rows without a Session_date must be applied before dated rows so that dated rows (PLS) can override them
ID_loc_key <- ID_loc_key[order(!is.na(ID_loc_key$Session_date)), ]

# 1.6 GPS key: one coordinate pair per site (mean of deployment/retrieval if the logger moved) ----
gps_key <- read.csv("01 Data/Logger_Placement_GPS_Key.csv", sep = ",", dec = ".", quote = "\"")
gps_key <- gps_key[gps_key$Site != "", ]

# 2 Assign Logger_ID to light_temp and DO_combined based on the key table ----
light_temp$Logger_ID <- NA
DO_combined$Logger_ID <- NA

for(i in 1:nrow(ID_loc_key)){
  key_row <- ID_loc_key[i, ]

  if(key_row$System == "LN"){
    # LN: Logger_ID determined by Site alone
    idx_lt <- which(light_temp$System == key_row$System & light_temp$Site == key_row$Site)
    idx_do <- which(DO_combined$System == key_row$System & DO_combined$Site == key_row$Site)
  } else if(key_row$Site == "PLS"){
    # PLS: Logger_ID determined by Session_date (blank date = all remaining, un-dated sessions)
    if(is.na(key_row$Session_date)){
      idx_lt <- which(light_temp$System == key_row$System & light_temp$Site == key_row$Site)
      idx_do <- which(DO_combined$System == key_row$System & DO_combined$Site == key_row$Site)
    } else {
      idx_lt <- which(light_temp$System == key_row$System & light_temp$Site == key_row$Site &
                         date(light_temp$Datetime_fix) == key_row$Session_date)
      idx_do <- which(DO_combined$System == key_row$System & DO_combined$Site == key_row$Site &
                         date(DO_combined$Datetime_fix) == key_row$Session_date)
    }
  } else {
    # JR, SL, PLP: Logger_ID determined by Location (grepl match against Location_key)
    idx_lt <- which(light_temp$System == key_row$System & light_temp$Site == key_row$Site &
                       grepl(key_row$Location_key, light_temp$Location))
    idx_do <- which(DO_combined$System == key_row$System & DO_combined$Site == key_row$Site &
                       grepl(key_row$Location_key, DO_combined$Location))
  }

  light_temp$Logger_ID[idx_lt] <- key_row$Logger_ID
  DO_combined$Logger_ID[idx_do] <- key_row$Logger_ID
}

# 3. Timeline plot ----------------------------------------------------------


## 3.1 Define time points -------------------------------------------------
t_p_start <- as.POSIXct("2023-11-15 06:00",format="%Y-%m-%d %H:%M", tz = "Africa/Nairobi")
t_p_end <- as.POSIXct("2023-11-17 12:00",format="%Y-%m-%d %H:%M", tz = "Africa/Nairobi")

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


ggexport(LN_env_plot, filename = "03 Analysis Output/LN_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(PL_env_plot, filename = "03 Analysis Output/PL_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(SL_env_plot, filename = "03 Analysis Output/SL_env_data.png", height = 1400, width = 3000, res = 300)
ggexport(JR_env_plot, filename = "03 Analysis Output/JR_env_data.png", height = 1400, width = 3000, res = 300)

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

ggexport(overview_box, filename = "03 Analysis Output/Overview_boxplots.png", height = 3000, width = 3000, res = 300)

# 5. Summary statistics -------------------------------------------------------------
# Add session date to light_temp and DO_combined per Index
light_temp <- light_temp %>%
  group_by(System, Site, Index, Logger_ID) %>%
  mutate(Session_date = min(date(Datetime), na.rm=T))

DO_combined <- DO_combined %>%
  group_by(System, Site, Index, Logger_ID) %>%
  mutate(Session_date = min(date(Datetime), na.rm=T))

# MM_dat$Session_date <- MM_dat$Date

# Summarize per site and deployment/index/session
light_temp_summary <- light_temp %>%
  group_by(System, Site, Session_date, Logger_ID) %>%
  summarise(Lux_mean = mean(Lux, na.rm=T),
            Lux_sd = sd(Lux, na.rm=T),
            Lux_min = min(Lux, na.rm=T),
            Lux_max = max(Lux, na.rm=T),
            T_C_mean = mean(T_C, na.rm=T),
            T_C_sd = sd(T_C, na.rm=T),
            T_C_min = min(T_C, na.rm=T),
            T_C_max = max(T_C, na.rm=T),
            T_logger_t_start = min(Datetime, na.rm=T),
            T_logger_t_end = max(Datetime, na.rm=T))

DO_summary <- DO_combined %>%
  group_by(System, Site, Index, Logger_ID) %>%
  summarise(Session_date = min(date(Datetime), na.rm=T),
            DO_AS_mean = mean(DO_AS, na.rm=T),
            DO_AS_sd = sd(DO_AS, na.rm=T),
            DO_AS_min = min(DO_AS, na.rm=T),
            DO_AS_max = max(DO_AS, na.rm=T),
            T_C_DO_mean = mean(T_C, na.rm=T),
            T_C_DO_sd = sd(T_C, na.rm=T),
            T_C_DO_min = min(T_C, na.rm=T),
            T_C_DO_max = max(T_C, na.rm=T),
            DO_logger_t_start = min(Datetime, na.rm=T),
            DO_logger_t_end = max(Datetime, na.rm=T))

MM_summary <- MM_dat %>%
  group_by(System, Site, Session_date, Logger_ID) %>%
  summarise(MM_n = n(),
            MM_time_mean = mean(Time, na.rm=T),
            pH_mean = mean(pH, na.rm=T),
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
            DO_AS_MM_max = max(DO_AS, na.rm=T),
            Depth_mean = mean(h_cm, na.rm=T),
            Depth_sd = sd(h_cm, na.rm=T),
            Depth_min = min(h_cm, na.rm=T),
            Depth_max = max(h_cm, na.rm=T))

complete_summary <- merge(light_temp_summary, DO_summary, by = c("System", "Site", "Session_date", "Logger_ID"), all = T)
complete_summary <- merge(complete_summary, MM_summary, by = c("System", "Site", "Session_date", "Logger_ID"), all = T)

# 6. Add GPS coordinates and weather information analogous to Script 2 -----

complete_summary <- merge(complete_summary, gps_key, by = "Site", all.x = T)

## 6.2 Sunrise/sunset/moonrise/moonset per site and Session_date ----------
has_gps <- !is.na(complete_summary$lat) & !is.na(complete_summary$lon)

complete_summary$Sunrise  <- as.POSIXct(NA, tz = TZ)
complete_summary$Sunset   <- as.POSIXct(NA, tz = TZ)
complete_summary$Moonrise <- as.POSIXct(NA, tz = TZ)
complete_summary$Moonset  <- as.POSIXct(NA, tz = TZ)
complete_summary$Moon_illuminated_fraction <- NA_real_
complete_summary$Moon_phase_value <- NA_real_       # 0 = new moon, 0.5 = full moon, 1 = next new moon
complete_summary$Moon_phase_name  <- NA_character_

moon_phase_name <- function(phase) {
  if (is.na(phase)) return(NA_character_)
  breaks <- c(0, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16, 1)
  labels <- c("New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon",
              "Waning Gibbous", "Last Quarter", "Waning Crescent", "New Moon")
  labels[findInterval(phase, breaks, rightmost.closed = TRUE)]
}

# Each row is a full calendar day (Session_date) rather than a recording window, so
# only the first sunrise/sunset/moonrise/moonset within that day is kept
for (i in which(has_gps)) {
  day_start <- as.POSIXct(complete_summary$Session_date[i], tz = TZ)
  day_end <- day_start + hours(24) - seconds(1)
  candidate_dates <- seq(as.Date(day_start) - 1, as.Date(day_end) + 1, by = "day")
  position_data <- data.frame(date = candidate_dates, lat = complete_summary$lat[i], lon = complete_summary$lon[i])

  sun_candidates  <- getSunlightTimes(data = position_data, keep = c("sunrise", "sunset"), tz = TZ)
  moon_candidates <- getMoonTimes(data = position_data, keep = c("rise", "set"), tz = TZ)

  sunrise_in_window  <- sun_candidates$sunrise[sun_candidates$sunrise >= day_start & sun_candidates$sunrise <= day_end]
  sunset_in_window   <- sun_candidates$sunset[sun_candidates$sunset   >= day_start & sun_candidates$sunset  <= day_end]
  moonrise_in_window <- moon_candidates$rise[!is.na(moon_candidates$rise) & moon_candidates$rise >= day_start & moon_candidates$rise <= day_end]
  moonset_in_window  <- moon_candidates$set[!is.na(moon_candidates$set)   & moon_candidates$set  >= day_start & moon_candidates$set  <= day_end]

  if (length(sunrise_in_window)  > 0) complete_summary$Sunrise[i]  <- min(sunrise_in_window)
  if (length(sunset_in_window)   > 0) complete_summary$Sunset[i]   <- min(sunset_in_window)
  if (length(moonrise_in_window) > 0) complete_summary$Moonrise[i] <- min(moonrise_in_window)
  if (length(moonset_in_window)  > 0) complete_summary$Moonset[i]  <- min(moonset_in_window)

  moon_illum <- getMoonIllumination(date = as.Date(day_start), keep = c("fraction", "phase"))
  complete_summary$Moon_illuminated_fraction[i] <- moon_illum$fraction
  complete_summary$Moon_phase_value[i] <- moon_illum$phase
  complete_summary$Moon_phase_name[i]  <- moon_phase_name(moon_illum$phase)
}

## 6.3 Historical weather from Open-Meteo Archive API (free, no key needed) ----
# One hourly-resolution API call per unique GPS location, spanning the full date
# range recorded there, then each row summarizes only the hours within its own day
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

gps_groups <- unique(complete_summary[has_gps, c("lat", "lon")])

complete_summary$Rainfall_mm      <- NA_real_
complete_summary$Rain_computed    <- NA_character_
complete_summary$Cloudcover_mean_pct <- NA_real_
complete_summary$Weather_dominant <- NA_character_
complete_summary$Weather_n_hours  <- 0L  # number of hourly records found within the day
for (label in WEATHER_LABELS) complete_summary[[paste0("pct_hours_", tolower(label))]] <- NA_real_

for (g in seq_len(nrow(gps_groups))) {
  group_rows <- which(has_gps &
    complete_summary$lat == gps_groups$lat[g] &
    complete_summary$lon == gps_groups$lon[g])

  lat <- complete_summary$lat[group_rows[1]]
  lon <- complete_summary$lon[group_rows[1]]
  date_from <- min(complete_summary$Session_date[group_rows], na.rm = TRUE)
  date_to   <- max(complete_summary$Session_date[group_rows], na.rm = TRUE)

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
    day_start <- as.POSIXct(complete_summary$Session_date[i], tz = TZ)
    day_end <- day_start + hours(24) - seconds(1)
    day_hours <- hourly[hourly$time >= day_start & hourly$time <= day_end, ]
    complete_summary$Weather_n_hours[i] <- nrow(day_hours)
    if (nrow(day_hours) == 0) next

    complete_summary$Rainfall_mm[i]         <- sum(day_hours$precipitation, na.rm = TRUE)
    complete_summary$Rain_computed[i]       <- ifelse(complete_summary$Rainfall_mm[i] > 0.1, "yes", "no")
    complete_summary$Cloudcover_mean_pct[i] <- mean(day_hours$cloudcover, na.rm = TRUE)

    label_pct <- table(factor(day_hours$weather_label, levels = WEATHER_LABELS)) / nrow(day_hours) * 100
    for (label in WEATHER_LABELS) complete_summary[[paste0("pct_hours_", tolower(label))]][i] <- unname(label_pct[label])
    complete_summary$Weather_dominant[i] <- names(which.max(label_pct))
  }
}

write.csv2(complete_summary, "01 Data/Environmental Data/Complete_env_data_summary_per_day.csv", row.names = F)

# 7. Create plots for the complete environmental data summary per day -----

## 7.1 Long-format table of all variables, only for the repeatedly sampled systems ----
env_summary_sub <- subset(complete_summary, System %in% c("PL", "SL", "JR"))
env_summary_sub <- subset(env_summary_sub, Site != ("JR2"))
env_summary_sub <- subset(env_summary_sub, Session_date > as.Date("05.10.2023", format = "%d.%m.%Y")) # Remove first measurements from Petro Lagoon, they were taken at different sites
env_summary_sub$Session_date <- as.Date(env_summary_sub$Session_date)

var_T_logger <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                            Session_date = env_summary_sub$Session_date,
                            Variable = "Temperature (°C)", Source = "Logger",
                            value = env_summary_sub$T_C_mean, val_min = env_summary_sub$T_C_min, val_max = env_summary_sub$T_C_max)

var_T_manual <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                            Session_date = env_summary_sub$Session_date,
                            Variable = "Temperature (°C)", Source = "Manual",
                            value = env_summary_sub$T_C_MM_mean, val_min = env_summary_sub$T_C_MM_min, val_max = env_summary_sub$T_C_MM_max)

var_DO_logger <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                             Session_date = env_summary_sub$Session_date,
                             Variable = "DO (%)", Source = "Logger",
                             value = env_summary_sub$DO_AS_mean, val_min = env_summary_sub$DO_AS_min, val_max = env_summary_sub$DO_AS_max)

var_DO_manual <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                             Session_date = env_summary_sub$Session_date,
                             Variable = "DO (%)", Source = "Manual",
                             value = env_summary_sub$DO_AS_MM_mean, val_min = env_summary_sub$DO_AS_MM_min, val_max = env_summary_sub$DO_AS_MM_max)

var_pH <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                      Session_date = env_summary_sub$Session_date,
                      Variable = "pH", Source = "Manual",
                      value = env_summary_sub$pH_mean, val_min = env_summary_sub$pH_min, val_max = env_summary_sub$pH_max)

var_Cond <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                        Session_date = env_summary_sub$Session_date,
                        Variable = "Conductivity (µS)", Source = "Manual",
                        value = env_summary_sub$Cond_mean, val_min = env_summary_sub$Cond_min, val_max = env_summary_sub$Cond_max)

var_Turb <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                        Session_date = env_summary_sub$Session_date,
                        Variable = "Turbidity (NTU)", Source = "Manual",
                        value = env_summary_sub$Turbidity_mean, val_min = env_summary_sub$Turbidity_min, val_max = env_summary_sub$Turbidity_max)

var_Depth <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                         Session_date = env_summary_sub$Session_date,
                         Variable = "Depth (cm)", Source = "Manual",
                         value = env_summary_sub$Depth_mean, val_min = env_summary_sub$Depth_min, val_max = env_summary_sub$Depth_max)

var_Rain <- data.frame(System = env_summary_sub$System, Site = env_summary_sub$Site,
                        Session_date = env_summary_sub$Session_date,
                        Variable = "Rainfall (mm)", Source = "Open-Meteo",
                        value = env_summary_sub$Rainfall_mm, val_min = env_summary_sub$Rainfall_mm, val_max = env_summary_sub$Rainfall_mm)

env_long <- rbind(var_T_logger, var_T_manual, var_DO_logger, var_DO_manual, var_pH, var_Cond, var_Turb, var_Depth, var_Rain)
env_long <- env_long[!is.na(env_long$value), ]
env_long$Variable <- factor(env_long$Variable, levels = c("Temperature (°C)", "DO (%)", "pH", "Conductivity (µS)",
                                                           "Turbidity (NTU)", "Depth (cm)", "Rainfall (mm)"))

## 7.2 Stacked per-variable panels, raw values, systems distinguished by colour ----
# one row per variable (free y-scales, since units/ranges differ hugely), shared x-axis across all rows
x_range <- range(env_long$Session_date, na.rm = TRUE)

env_multivar_plot <- ggplot(env_long, aes(x = Session_date, y = value, colour = System,
                                           shape = Source, group = System)) + #, #linetype = Site,
                                          # group = interaction(System, Source, Site))) +
                                          # group = interaction(System, Source))) +
  # geom_line(position = position_dodge(width = 0.6), alpha = 0.5) +
  geom_smooth(se = FALSE, linewidth = .5, linetype = 2)+
  geom_pointrange(aes(ymin = val_min, ymax = val_max), position = position_dodge(width = 0.6), size = 0.4) +
  facet_wrap(~ Variable, ncol = 1, scales = "free_y", strip.position = "left") +
  scale_x_date(limits = x_range, date_breaks = "1 week", date_labels = "%d %b") +
  labs(x = "Date", y = NULL, colour = "System") + # , shape = "Source", linetype = "Site") +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom", strip.placement = "outside", strip.background = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggexport(env_multivar_plot, filename = "03 Analysis Output/Multivariable_env_timeseries.png", height = 4200, width = 2400, res = 300)

