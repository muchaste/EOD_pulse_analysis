# R script for reading in event_summaries from 03_1_REF to compare different extractions


# Load necessary libraries
library(tidyverse)
library(ggpubr)
library(lubridate)

rm(list = ls())


# Choose file 1
setwd("E:/SL_L1_surface/20231115/01_thresh_0.005_splitevent_0.01_ipi_0.5_merge_2_bp13kHz_variable_newcalib_newdedup/")
event_summary_1 <- read.csv2("all_event_summaries.csv", sep=",", dec=".")

# Choose file 2
setwd("E:/SL_L2_ground/20231115/01_thresh_0.005_splitevent_0.01_ipi_0.5_merge_2_bp13kHz_variable_newcalib_newdedup/")
event_summary_2 <- read.csv2("all_event_summaries.csv", sep=",", dec=".")

irreg_colname <- "dead_channels_file"
if (irreg_colname %in% colnames(event_summary_1)){
  event_summary_1 <- subset(event_summary_1, select = -c(dead_channels_file))
}
if (irreg_colname %in% colnames(event_summary_2)){
  event_summary_2 <- subset(event_summary_2, select = -c(dead_channels_file))
}

# Compare the two dataframes
# Compare timeline of events between the two extractions
event_summary_1$event_start_time <- as.POSIXct(event_summary_1$event_start_time, format="%Y-%m-%d %H:%M:%S")
event_summary_2$event_start_time <- as.POSIXct(event_summary_2$event_start_time, format="%Y-%m-%d %H:%M:%S")
event_summary_1$context <- "01 old"
event_summary_2$context <- "02 new"
event_summary_combined <- rbind.data.frame(event_summary_1, event_summary_2)

plot_labels = c("old", "new")

ggplot(event_summary_combined, aes(x = event_start_time, y = n_eods, color = context)) +
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
ggplot(event_summary_combined, aes(x = mean_ipi_seconds, fill = context)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Mean IPI Seconds: Reference vs Location Extraction",
       x = "Mean IPI Seconds",
       y = "Density") +
  theme_classic() +
  scale_fill_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(fill = guide_legend(title = "Extraction Method"))

# Compare distributions of mean_width_ms between the two extractions
ggplot(event_summary_combined, aes(x = mean_width_ms, fill = context)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Mean EOD Width (ms): Reference vs Location Extraction",
       x = "Mean EOD Width (ms)",
       y = "Density") +
  theme_classic() +
  scale_fill_manual(values = c("blue", "red"), labels = plot_labels) +
  guides(fill = guide_legend(title = "Extraction Method"))

# Group and summarize by hour
event_summary_combined_hourly <- event_summary_combined %>%
  group_by(hour_date = floor_date(event_start_time, unit = "hour"), context) %>%
  summarise(total_eods = sum(n_eods),
            hour_num = hour(hour_date[1]))

# event_summary_combined_hourly$hour_num <- as.numeric(event_summary_combined_hourly$hour)

# Plot the combined data
ggplot(event_summary_combined_hourly, aes(x = hour_date, y = total_eods, color = context)) +
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

ggplot(event_summary_combined_hourly, aes(x = hour_num, y = total_eods, color = context)) +
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

