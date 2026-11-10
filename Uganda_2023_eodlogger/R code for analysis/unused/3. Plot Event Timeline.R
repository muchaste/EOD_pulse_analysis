# Load necessary libraries
library(tidyverse)
library(lubridate)
library(ggpubr)

rm(list = ls())

# Define the root directory
root_dir <- "I:/PL_L9_L10_stream/"
setwd(root_dir)

# List all event_list.csv files in the 'events_waveform' subdirectory
event_list_file_paths <- list.files(root_dir, pattern = "event_list.csv", recursive = TRUE, full.names = TRUE) %>%
  str_subset("events_waveform")
eod_list_file_paths <- list.files(root_dir, pattern = "eods_by_event.csv", recursive = TRUE, full.names = TRUE) %>%
  str_subset("events_waveform")

# Initialize an empty dataframe to store the combined data
eod_data <- data.frame()

# Loop through each file, read it, and combine it with the location information
counter <- 1
for (file_path in eod_list_file_paths) {
  temp_data <- read_csv(file_path)
  location_name <- strsplit(strsplit(file_path, "/")[[1]][2], "_")[[1]][1]
  temp_data <- temp_data %>%
    mutate(location = location_name,
           sampling_day = counter)
  
  eod_data <- bind_rows(eod_data, temp_data)
  counter <- counter + 1
}

# Convert timestamp to POSIXct
eod_data <- eod_data %>%
  mutate(timestamp = ymd_hms(timestamp))

# Create a reference start date and end date
reference_start_date <- as.Date("2024-02-01")
reference_end_date <- as.Date("2024-02-02")
t_start <- as.POSIXct(paste(reference_start_date, "12:00:00"), tz = "UTC")
t_end <- as.POSIXct(paste(reference_end_date, "11:59:00"), tz = "UTC")

# Adjust the dates to the reference dates while keeping the time information intact
eod_data$t_fixed <- eod_data$timestamp
for (day in unique(eod_data$sampling_day)) {
  idc <- which(eod_data$sampling_day == day)
  t_temp <- eod_data$t_fixed[idc]
  t_ref <- as.POSIXct(paste(reference_start_date, format(t_temp[1], "%H:%M:%S")), tz = "UTC")
  t_diff <- t_ref - t_temp[1]
  t_temp <- t_temp + t_diff
  eod_data$t_fixed[idc] <- t_temp
}

# Create a summary of EODs and events
summary_data_species <- eod_data %>%
  group_by(t_fixed = floor_date(t_fixed, "hour"), channel, species_pred) %>%
  summarize(n_eods = n(), n_events = n_distinct(event)) %>%
  ungroup()

# Define dusk and night periods (example times, adjust as necessary)
dusk_start <- as.POSIXct("2024-02-01 18:00:00", tz = "UTC")
night_start <- as.POSIXct("2024-02-01 18:30:00", tz = "UTC")
night_end <- as.POSIXct("2024-02-02 06:00:00", tz = "UTC")
dawn_end <- as.POSIXct("2024-02-02 06:30:00", tz = "UTC")

# Plot the data using ggplot2
tl_together <- ggplot(summary_data_species, aes(x = t_fixed, fill = species_pred)) +
  geom_rect(aes(xmin = dusk_start, xmax = night_start, ymin = -Inf, ymax = Inf),
            fill = "lightgrey", alpha = 0.05) +
  geom_rect(aes(xmin = night_start, xmax = night_end, ymin = -Inf, ymax = Inf),
            fill = "darkgrey", alpha = 0.05) +
  geom_rect(aes(xmin = night_end, xmax = dawn_end, ymin = -Inf, ymax = Inf),
            fill = "lightgrey", alpha = 0.05) +
  geom_col(aes(y = n_eods), size = 1) +
  xlim(t_start, t_end) +
  labs(title = "EODs and Events Over Time",
       x = "Time",
       y = "Count",
       fill = "Species") +
  theme_minimal() +
  theme(legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

chan_order <- c("7", "6", "5", "4", "3", "2", "1")

summary_data_species$channel <- factor(summary_data_species$channel, levels = chan_order)

tl_by_channel <- ggplot(summary_data_species, aes(x = t_fixed, y = n_eods, fill = species_pred)) +
  geom_col() +
  xlim(t_start, t_end) +
  facet_wrap(~ channel, nrow = 7, scales = "fixed") +
  labs(x = "Time", y = "# EODs", fill = "Species") +
  theme_classic() +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

# Pie Chart
# Compute the position of labels
eod_dat_species <- eod_data %>%
  group_by(species_pred) %>%
  summarise(n_eods = n())

eod_pie_dat <- eod_dat_species %>% 
  arrange(desc(species_pred)) %>%
  mutate(prop = n_eods / sum(eod_dat_species$n_eods) * 100) %>%
  mutate(ypos = cumsum(prop) - 0.5 * prop)

# Basic piechart
pie_chart <- ggplot(eod_pie_dat, aes(x = "", y = prop, fill = species_pred)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() + 
  theme(legend.position = "none") +
  geom_text(aes(y = ypos, label = species_pred), color = "white", size = 6)

# Export plots as PDFs
ggexport(tl_together, filename = "EOD_timeline_together.pdf", width = 8, height = 6)
ggexport(tl_by_channel, filename = "EOD_timeline_by_channel.pdf", width = 8, height = 12)
ggexport(pie_chart, filename = "EOD_species_pie_chart.pdf", width = 6, height = 6)
