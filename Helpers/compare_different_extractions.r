# R script for reading in event_summaries from 03_1_REF to compare different extractions


# Load necessary libraries
library(tidyverse)
library(ggpubr)
library(lubridate)

rm(list = ls())


# Choose file 1
setwd("F:/Juma_L5/20231011/02_thresh_0.002_event_0.01_ipi_0.5_merge_2_nogmm_bp_flexible/")
event_summary_low <- read.csv2("all_event_summaries.csv", sep=",", dec=".")

# Choose file 2
setwd("F:/Juma_L5/20231011/03_thresh_0.005_event_0.01_ipi_0.5_merge_2_nogmm_bp_flexible/")
event_summary_high <- read.csv2("all_event_summaries.csv", sep=",", dec=".")

# Compare the two dataframes
# Compare timeline of events between the two extractions
event_summary_low$event_start_time <- as.POSIXct(event_summary_low$event_start_time, format="%Y-%m-%d %H:%M:%S")
event_summary_high$event_start_time <- as.POSIXct(event_summary_high$event_start_time, format="%Y-%m-%d %H:%M:%S")
event_summary_low$extraction_method <- "Event thresh 0.02"
event_summary_high$extraction_method <- "Event thresh 0.04"
event_summary_combined <- rbind.data.frame(event_summary_low, event_summary_high)

ggplot(event_summary_combined, aes(x = event_start_time, y = n_eods, color = extraction_method)) +
  geom_point() +
  labs(title = "Timeline of Events: Reference vs Location Extraction",
       x = "Start Time",
       y = "Number of EODs") +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "1 hour") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "topright") +
  theme_classic()+
  scale_color_manual(values = c("blue", "red"), labels = c("Low Event Thresh", "High Event Thresh")) +
  guides(color = guide_legend(title = "Extraction Method"))


# Compare distributions of mean_ipi_seconds between the two extractions
ggplot(event_summary_combined, aes(x = mean_ipi_seconds, fill = extraction_method)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Mean IPI Seconds: Reference vs Location Extraction",
       x = "Mean IPI Seconds",
       y = "Density") +
  theme_classic() +
  scale_fill_manual(values = c("blue", "red"), labels = c("Low Event Thresh", "High Event Thresh")) +
  guides(fill = guide_legend(title = "Extraction Method"))

# Compare distributions of mean_width_ms between the two extractions
ggplot(event_summary_combined, aes(x = mean_width_ms, fill = extraction_method)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Mean EOD Width (ms): Reference vs Location Extraction",
       x = "Mean EOD Width (ms)",
       y = "Density") +
  theme_classic() +
  scale_fill_manual(values = c("blue", "red"), labels = c("Low Event Thresh", "High Event Thresh")) +
  guides(fill = guide_legend(title = "Extraction Method"))

# Group and summarize by hour
event_summary_combined_hourly <- event_summary_combined %>%
  group_by(hour = floor_date(event_start_time, unit = "hour"), extraction_method) %>%
  summarise(total_eods = sum(n_eods))

# Plot the combined data
ggplot(event_summary_combined_hourly, aes(x = hour, y = total_eods, color = extraction_method)) +
  geom_line() +
  geom_point() +
  labs(title = "Total EODs per Hour: Reference vs Location Extraction",
       x = "Hour",
       y = "Total EODs") +
  scale_x_datetime(date_labels = "%H:%M", date_breaks = "1 hour") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "bottom") + 
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"), labels = c("Low Event Thresh", "High Event Thresh")) +
  guides(color = guide_legend(title = "Extraction Method"))

