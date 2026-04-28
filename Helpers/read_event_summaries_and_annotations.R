
library(jsonlite)
library(tidyverse)
library(ggpubr)

annotations <- fromJSON("F:/Juma_L5/20231011/04 As above but with locations/annotations.json")
event_summary <- read.csv2("F:/Juma_L5/20231011/04 As above but with locations/all_event_summaries.csv", sep=",", dec=".")

fish_counts <- as.data.frame(annotations$fish_counts)
fish_counts <- pivot_longer(fish_counts, cols = everything(), names_to="event", values_to = "fish_n")
fish_counts$event_id <- NA
for (i in 1:length(fish_counts$event)){
  fish_counts$event_id[i] <- as.numeric(strsplit(fish_counts$event[i], '_')[[1]][3])
}

# Recalculate IPIs (within each channel event)
setwd("F:/Juma_L5/20231011/04 As above but with locations/")
event_summary$ipi_cv <- 0
eod_tables <- list.files(pattern="*eod_table.csv")

for (i in 1:length(eod_tables)){
  d <- read.csv2(eod_tables[i], sep=",",dec=".")
  ipi_list <- NULL
  for (ch_event in unique(d$channel_event_id)){
    ch_event_dat <- subset(d, channel_event_id == ch_event)
    ch_event_ipis <- diff(ch_event_dat$midpoint_idx)/96000
    ipi_list <- c(ipi_list, ch_event_ipis)
  }
  event_summary$mean_ipi_recalc[i] <- mean(ipi_list)
  event_summary$median_ipi_recalc[i] <- median(ipi_list)
  event_summary$ipi_cv[i] <- sd(ipi_list)/mean(ipi_list)
}


# Compile with fish counts
all_dat <- left_join(fish_counts, event_summary, by="event_id")

p1 <- ggplot(all_dat, aes(x = as.factor(fish_n), y = mean_ipi_seconds))+
  geom_boxplot()
p2 <- ggplot(all_dat, aes(x = as.factor(fish_n), y = mean_ipi_recalc))+
  geom_boxplot()
p3 <- ggplot(all_dat, aes(x = as.factor(fish_n), y = median_ipi_seconds))+
  geom_boxplot()
p4 <- ggplot(all_dat, aes(x = as.factor(fish_n), y = median_ipi_recalc))+
  geom_boxplot()
p5 <- ggplot(all_dat, aes(x = as.factor(fish_n), y = ipi_cv))+
  geom_boxplot()

ggarrange(p1, p2, p3, p4, p5, ncol=3, nrow = 2)

all_dat %>% group_by(fish_n) %>% summarise(mean(mean_ipi_seconds),
                                           range(mean_ipi_seconds),
                                           mean(median_ipi_seconds),
                                           range(median_ipi_seconds))
subset(all_dat, fish_n == 2)$mean_ipi_seconds
