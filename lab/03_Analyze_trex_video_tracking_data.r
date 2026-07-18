# Load video tracking data and create summary datasets, plots, and statistical analyses
# Uses output from trex tracking

library(MoveR)
library(tidyverse)
library(ggpubr)


rm(list = ls())

# Set root directory (output root of trex tracking) and subdirectories for processed data and plots
root_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/trex_tracking"
processed_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/trex_tracking/processed"
plot_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/trex_tracking/plots/"

setwd(root_dir)

tracking_file_list <- list.files(pattern = "*.npz", recursive = TRUE)

# Infer id and trial from folder structure (assumes folder structure is trial/id/filename)
# parse file start time from file name (example filename: FishID_PD1_2022_11_15_8_39_24_24_id0.npz)
tracking_file_list <- tibble(
  file_path = tracking_file_list,
  trial = strsplit(tracking_file_list, "/") %>% sapply("[", 1),
  id = strsplit(tracking_file_list, "/") %>% sapply("[", 2),
  file_time = as.POSIXct(strptime(sapply(strsplit(basename(tracking_file_list), "_"), function(x) paste(x[3:10], collapse = "_")), format = "%Y_%m_%d_%H_%M_%OS"))
)

track_1s_all <- NULL
track_1m_all <- NULL
track_10m_all <- NULL

for(sub_trial in unique(tracking_file_list$trial)) {
    for(sub_id in unique(tracking_file_list$id[tracking_file_list$trial == sub_trial])) {
      tryCatch({
        tracking_file_list_subset <- tracking_file_list %>% filter(str_equal(trial, sub_trial) & str_equal(id, sub_id))



      }, error = function(e){
        message(paste0("Skipping ", sub_id, "/", sub_trial, ": ", conditionMessage(e)))
      })
    }
}