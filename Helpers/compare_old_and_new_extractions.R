# R script for reading in event_summaries from 03_1_REF to compare different extractions


# Load necessary libraries
library(tidyverse)
library(ggpubr)
library(lubridate)

rm(list = ls())


# Choose file 1
setwd("E:/Juma_L5/20231011/Head-to-tail Extraction classified 0.002/Re-classified LDA features/")
event_summary_old <- read.csv2("event_list.csv", sep=",", dec=".")

# Choose file 2
setwd("E:/Juma_L5/20231011/03_thresh_0.005_event_0.01_ipi_0.5_merge_2_nogmm_bp_flexible_newcalib_newdedup/")
event_summary_new <- read.csv2("all_event_summaries.csv", sep=",", dec=".")

# Compare the two dataframes
# Compare timeline of events between the two extractions
event_summary_old$event_start_time <- as.POSIXct(event_summary_old$t_start, format="%Y-%m-%d %H:%M:%S")
event_summary_new$event_start_time <- as.POSIXct(event_summary_new$event_start_time, format="%Y-%m-%d %H:%M:%S")
event_summary_old$extraction_method <- "01 old"
event_summary_new$extraction_method <- "02 new"
old_sub <- subset(event_summary_old, select = c(event_start_time, pulses, mean_f, extraction_method))
new_sub <- subset(event_summary_new, select = c(event_start_time, n_eods, mean_ipi_seconds, extraction_method))
old_sub$mean_f <- 1/old_sub$mean_f
colnames(old_sub) <- c("event_start_time", "n_eods", "mean_ipi_seconds", "extraction_method")
event_summary_combined <- rbind.data.frame(old_sub, new_sub)

plot_labels = c("old extraction", "new extraction")

ggplot(event_summary_combined, aes(x = event_start_time, y = n_eods, color = extraction_method)) +
  geom_point() +
  labs(title = "Timeline of Events: Reference vs Location Extraction",
       x = "Start Time",
       y = "Number of EODs") +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "1 hour") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "topright") +
  theme_classic()+
  scale_color_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(color = guide_legend(title = "Extraction Method"))


# Compare distributions of mean_ipi_seconds between the two extractions
ggplot(event_summary_combined, aes(x = mean_ipi_seconds, fill = extraction_method)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Mean IPI Seconds: Reference vs Location Extraction",
       x = "Mean IPI Seconds",
       y = "Density") +
  theme_classic() +
  scale_fill_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(fill = guide_legend(title = "Extraction Method"))


# Group and summarize by hour
event_summary_combined_hourly <- event_summary_combined %>%
  group_by(hour_date = floor_date(event_start_time, unit = "hour"), extraction_method) %>%
  summarise(total_eods = sum(n_eods),
            hour_num = hour(hour_date[1]))

# event_summary_combined_hourly$hour_num <- as.numeric(event_summary_combined_hourly$hour)

# Plot the combined data
ggplot(event_summary_combined_hourly, aes(x = hour_date, y = total_eods, color = extraction_method)) +
  geom_line() +
  geom_point() +
  labs(title = "Total EODs per Hour: Reference vs Location Extraction",
       x = "Hour",
       y = "Total EODs") +
  scale_x_datetime(date_labels = "%H:%M", date_breaks = "1 hour") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "bottom") + 
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(color = guide_legend(title = "Extraction Method"))

ggplot(event_summary_combined_hourly, aes(x = hour_num, y = total_eods, color = extraction_method)) +
  geom_line() +
  geom_point() +
  labs(title = "Total EODs per Hour: Reference vs Location Extraction",
       x = "Hour",
       y = "Total EODs") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "bottom") + 
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(color = guide_legend(title = "Extraction Method"))
