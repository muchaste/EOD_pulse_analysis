library(lubridate)
library(tidyverse)
library(ggpubr)
library(openxlsx)
library(plotly)
library(scales)

rm(list=ls())

# PC <- "field laptop"
# PC <- "thinkpad"
PC <- "thinkpad"


# 0. Set working directories ----------------------------------------------
if(PC == "field laptop"){
  datadir <- "C:/Users/Stefan Mucha/Desktop/Nab_2023_Field_Data/HOBO loggers/"
} else if(PC == "thinkpad"){
  datadir <-"C:/Users/stefa/Seafile/Uni/01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/"
} else if(PC == "office"){
  datadir <- "C:/Users/ShuttleBox/Seafile/Uni/01 Research Projects/04 Uganda 2023 EOD loggers/01 Data/"
}



# 1. Read in data and format timestamp ------------------------------------

## 1.1 Light and temp data 

setwd(datadir)
light_temp <- read.csv2("Temp_Light_All_Logger_Data.csv")
light_temp$Datetime <- as.POSIXct(light_temp$Datetime)
light_temp$Datetime_fix <- as.POSIXct(light_temp$Datetime_fix)
light_temp$Datetime_fix_loop <- as.POSIXct(light_temp$Datetime_fix_loop)

## 1.2 DO PyroScience logger

setwd(datadir)
DO_pyro <- read.csv2("DO_Pyro_All_Data.csv")
DO_pyro$Datetime <- as.POSIXct(DO_pyro$Datetime)
DO_pyro$Datetime_fix <- as.POSIXct(DO_pyro$Datetime_fix)


# Manual Measurements
setwd(datadir)
MM_dat <- read.xlsx("Recordings and Measurements.xlsx", sheet = "Manual_measurements")
MM_dat$Date <- convertToDate(MM_dat$Date)
MM_dat$Datetime <- convertToDateTime(MM_dat$Time)
date(MM_dat$Datetime) <- MM_dat$Date


# 2. Calculate mean values per site and location --------------------------
light_temp_means <- light_temp %>%
  group_by(System, Location, Datetime_fix) %>%
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

t_p_start <- as.POSIXct("2023-11-15 06:00",format="%Y-%m-%d %H:%M")
t_p_end <- as.POSIXct("2023-11-17 12:00",format="%Y-%m-%d %H:%M")

l_plt <- ggplot(light_temp, aes(x = Datetime_fix, y = Lux, col = Location, group = interaction(Index, Location)))+
  geom_line(alpha = 0.2)+
  geom_line(data = light_temp_means, aes(y = Lux_mean, group = Location), size = 1)+
  facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   breaks = date_breaks("12 hours"),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Light intensity (Lux)")

t_plt <- ggplot(light_temp, aes(x = Datetime_fix, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line(alpha = 0.2)+
  geom_line(data = light_temp_means, aes(y = T_C_mean, group = Location), size = 1)+
  facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   breaks = date_breaks("12 hours"),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Temperature (°C)")

DO_plt <- ggplot(DO_pyro, aes(x = Datetime_fix, y = DO_AS, group = Deployment))+
  geom_line(col = "darkblue", alpha = 0.2)+
  geom_line(data = DO_data_means, aes(y = DO_AS_mean, group=System), col = "darkblue", size = 1)+
  facet_wrap(~System, ncol = 4)+
  scale_x_datetime(limits = c(t_p_start, t_p_end),
                   breaks = date_breaks("12 hours"),
                   labels = date_format("%H:%M"))+
  labs(x = "Time of day", y = "Air saturation (%)")

all_plot <- ggarrange(DO_plt, l_plt, t_plt, nrow = 3, common.legend = T, legend = "bottom",
                      align = "hv")

setwd(datadir)
setwd("../02 Analysis Output/")

ggexport(all_plot, filename = "Environmental logger data.png", width = 2000, height = 1600, res = 200)


# 4. Daily Data with Man Measurements ------------------------------

daily_light_temp <- light_temp %>% group_by(Index, System, Photoperiod) %>%
  summarise(Lux_mean = mean(Lux, na.rm = T),
            Lux_max = max(Lux),
            Lux_min = min(Lux),
            Lux_sd = sd(Lux, na.rm = T),
            T_C_mean = mean(T_C, na.rm = T),
            T_C_max = max(T_C),
            T_C_min = min(T_C),
            Datetime = mean(Datetime))

daily_DO <- DO_pyro %>% group_by(Deployment, System, Photoperiod) %>%
  summarise(DO_AS_mean = mean(DO_AS, na.rm = T),
            DO_AS_max = max(DO_AS),
            DO_AS_min = min(DO_AS),
            Datetime = mean(Datetime))

daily_mm <- MM_dat %>% group_by(Date, System) %>%
  summarise(T_C_mean = mean(T_C, na.rm = T),
            T_C_max = max(T_C),
            T_C_min = min(T_C),
            DO_AS_mean = mean(DO_AS, na.rm = T),
            DO_AS_max = max(DO_AS),
            DO_AS_min = min(DO_AS),
            pH_mean = mean(pH, na.rm = T),
            pH_max = max(pH),
            pH_min = min(pH),
            Cond_uS_mean = mean(Cond_uS, na.rm = T),
            Cond_uS_max = max(Cond_uS),
            Cond_uS_min = min(Cond_uS),
            h_cm_mean = mean(h_cm, na.rm = T),
            h_cm_max = max(h_cm),
            h_cm_min = min(h_cm),
            Turbidity_NTU_mean = mean(Turbidity_NTU, na.rm = T),
            Turbidity_NTU_max = max(Turbidity_NTU),
            Turbidity_NTU_min = min(Turbidity_NTU),
            Datetime = mean(Datetime))


# 5. Plot Daily variation -------------------------------------------------

DO_plt_daily <- ggplot(daily_DO[daily_DO$System != "LN",], aes(x = Datetime, y = DO_AS_mean))+
  geom_point(aes(col = Photoperiod))+
  geom_errorbar(aes(ymin = DO_AS_min, ymax = DO_AS_max, col = Photoperiod))+
  geom_point(data = daily_mm[daily_mm$System != "LN",])+
  geom_smooth(data = daily_mm[daily_mm$System != "LN",], se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Air saturation (%)")+
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank())

l_plt_daily <- ggplot(daily_light_temp[daily_light_temp$System != "LN",], aes(x = Datetime, y = Lux_mean, col = Photoperiod))+
  geom_point()+
  geom_errorbar(aes(ymin = Lux_min, ymax = Lux_max))+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Light intensity (Lux)")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

t_plt_daily <- ggplot(daily_light_temp[daily_light_temp$System != "LN",], aes(x = Datetime, y = T_C_mean))+
  geom_point(aes(col = Photoperiod))+
  geom_errorbar(aes(ymin = T_C_min, ymax = T_C_max, col = Photoperiod))+
  geom_point(data = daily_mm[daily_mm$System != "LN",])+
  geom_smooth(se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Temperature (°C)")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

pH_plt_daily <- ggplot(data = daily_mm[daily_mm$System != "LN",], aes(x = Datetime, y = pH_mean))+
  geom_point()+
  geom_errorbar(aes(ymin = pH_min, ymax = pH_max))+
  geom_smooth(se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "pH")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

cond_plt_daily <- ggplot(data = daily_mm[daily_mm$System != "LN",], aes(x = Datetime, y = Cond_uS_mean))+
  geom_point()+
  geom_errorbar(aes(ymin = Cond_uS_min, ymax = Cond_uS_max))+
  geom_smooth(se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Conductivity (uS)")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

h_plt_daily <- ggplot(data = daily_mm[daily_mm$System != "LN",], aes(x = Datetime, y = h_cm_mean))+
  geom_point()+
  geom_errorbar(aes(ymin = h_cm_min, ymax = h_cm_max))+
  geom_smooth(se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Water height (cm)")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank())

turb_plt_daily <- ggplot(data = daily_mm[daily_mm$System != "LN",], aes(x = Datetime, y = Turbidity_NTU_mean))+
  geom_point()+
  geom_errorbar(aes(ymin = Turbidity_NTU_min, ymax = Turbidity_NTU_max))+
  geom_smooth(se=F, col = "black", size = 0.5, method = "lm")+
  facet_wrap(~System, ncol = 3)+
  labs(x = "Date", y = "Water turbidity (NTU)")+
  theme(strip.background = element_blank(),
        strip.text.x = element_blank())

all_plt_daily <- ggarrange(DO_plt_daily, l_plt_daily, t_plt_daily, pH_plt_daily,
                           cond_plt_daily, h_plt_daily, turb_plt_daily,
                           nrow = 7, common.legend = T, legend = "bottom",
                           heights = c(5,5,5,5,5,5,6), align = "v")


setwd(datadir)
setwd("../02 Analysis Output/")

ggexport(all_plt_daily, filename = "Environmental data daily no LN.png", width = 2000, height = 3000, res = 200)


# Boxplots across all days ------------------------------------------------
do_box <- ggplot(DO_pyro, aes(x = Photoperiod, y = DO_AS, fill = Photoperiod))+
  geom_boxplot()+
  facet_wrap(~System, ncol = 4)+
  labs(x = "Photoperiod", y = "Air saturation (%)")+
  theme_classic()
  

