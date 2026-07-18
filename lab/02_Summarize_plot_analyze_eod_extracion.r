# Load eod extraction results and create summary datasets, plots, and statistical analyses
# Uses output eod tables from lab/01_Shuttle_box_Matlab_pulse_extraction.py

library(tidyverse)
library(ggpubr)
library(roll)
library(lubridate)
library(scales)

rm(list = ls())

normalize_mean <- function(vector){
  norm_vector <- vector/mean(vector, na.rm = T)
  return(norm_vector)
}

# Set root directory (output root of 01_Shuttle_box_Matlab_pulse_extraction.py)
# first_round_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/first_round"
# second_round_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/second_round"

root_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction"
processed_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/processed"
plot_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/plots/"

setwd(root_dir)

# Analysis start time
t_s_analysis <- as.POSIXct("2024-06-01 08:40:00", format="%Y-%m-%d %H:%M:%OS", tz="CET")

# List eod extraction results
eod_file_list <- list.files(pattern="*_pulse_extraction_results.csv", recursive = TRUE)

# Infer id and trial from folder structure (assumes folder structure is trial/id/filename)
eod_file_list <- tibble(
  file_path = eod_file_list,
  trial = strsplit(eod_file_list, "/") %>% sapply("[", 1),
  id = strsplit(eod_file_list, "/") %>% sapply("[", 2)
)

eods_1s_all <- NULL
eods_1m_all <- NULL
eods_10m_all <- NULL

for(sub_trial in unique(eod_file_list$trial)) {
    for(sub_id in unique(eod_file_list$id[eod_file_list$trial == sub_trial])) {
        eod_file_list_subset <- eod_file_list %>% filter(str_equal(trial, sub_trial) & str_equal(id, sub_id))
        eods_1s_id <- NULL
        for(file in eod_file_list_subset$file_path) {
            eod_data <- read.csv2(file, sep=",", dec=".", header=TRUE)
            metadata_file <- str_replace(basename(file), "_pulse_extraction_results.csv", "_metadata.csv")
            file_metadata <- read.csv2(file.path(dirname(file), metadata_file), sep=",", dec=".", header=TRUE)

            file_start <- as.POSIXct(file_metadata$file_timestamp[1], format="%Y-%m-%d %H:%M:%OS", tz="CET")
            file_end <- as.POSIXct(file_metadata$file_end_timestamp[1], format="%Y-%m-%d %H:%M:%OS", tz="CET")

            # Format timestamp column to POSIXct and cut to round 1s start time
            eod_data$timestamp <- as.POSIXct(eod_data$timestamp, format="%Y-%m-%d %H:%M:%OS", tz="CET")

            # sort by timestamp
            eod_data <- eod_data[order(eod_data$timestamp, decreasing = FALSE), ]
            
            # Add side column based on eod_channel column
            eod_data$side <- "left"
            eod_data$side <- ifelse(eod_data$eod_channel > 1, "right", "left")

            # Orientation column (if side == left: left_h (chan 1), if side == right: right_h (chan 3))
            eod_data$orientation <- eod_data$orient_left_h
            eod_data$orientation[eod_data$side == "right"] <- eod_data$orient_right_h[eod_data$side == "right"]

            # Use amplitude also only from horizontal channels to get an estimate of movement-based amplitude variation
            eod_data$amp_movement <- eod_data$amp_left_h
            eod_data$amp_movement[eod_data$side == "right"] <- eod_data$amp_right_h[eod_data$side == "right"]

            # Add IPI column (inter-pulse interval in seconds)
            eod_data$IPI <- c(NA, as.numeric(diff(eod_data$relative_time_s)))

            # Create summary dataset for this file
            # per second
            eods_1s_file <- eod_data %>% 
                group_by(time = cut(timestamp, breaks = "1 sec")) %>%
                summarize(
                eod_rate_hz = n(),
                IPI_cv = sd(IPI, na.rm=T)/mean(IPI, na.rm=T),
                IPI_s = mean(IPI, na.rm=T),
                pp_dur_us = mean(eod_width_us, na.rm=T),
                amp_cv = sd(amp_movement, na.rm=T)/mean(amp_movement, na.rm=T),
                pp_amp_v = mean(amp_movement, na.rm=T),
                pp_ratio = mean(eod_amplitude_ratio, na.rm=T),
                peak_fft_hz = mean(fft_freq_max, na.rm=T),
                n_sidechange = sum(diff(as.numeric(as.factor(side))) != 0, na.rm=T),
                n_turns = sum(diff(as.numeric(as.factor(orientation))) != 0, na.rm=T)
                )
            eods_1s_file$time <- as.POSIXct(eods_1s_file$time)
            eods_1s_id <- rbind(eods_1s_id, eods_1s_file)
        }

        # Sort eods_1s by time
        eods_1s_id <- eods_1s_id[order(eods_1s_id$time, decreasing = FALSE), ]

        # Fill in missing seconds with NA values
        eods_1s_id <- eods_1s_id %>%
          complete(time = seq(min(time), max(time), by = "1 sec"))

        # Common analysis start time is always 08:40:00, so cut to that time or fill with NA if the first timestamp is later than that
        t_s_analysis <- as.POSIXct(paste(format(min(eods_1s_id$time), "%Y-%m-%d"), "08:40:00"), format="%Y-%m-%d %H:%M:%S", tz="CET")
        if(min(eods_1s_id$time) > t_s_analysis){
          na_rows <- data.frame(time = seq(from = t_s_analysis, to = min(eods_1s_id$time) - 1, by = "1 sec"))
          eods_1s_id <- rbind(na_rows, eods_1s_id)
          rownames(eods_1s_id) <- NULL
        } else {
          eods_1s_id <- eods_1s_id[eods_1s_id$time >= t_s_analysis, ]
        }

        # Common analysis end time is always 08:40:00 + 200 minutes, so cut to that time or fill with NA if the last timestamp is earlier than that
        t_e_analysis <- t_s_analysis + 200*60
        if(max(eods_1s_id$time) < t_e_analysis){
          na_rows <- data.frame(time = seq(from = max(eods_1s_id$time) + 1, to = t_e_analysis, by = "1 sec"))
          eods_1s_id <- rbind(eods_1s_id, na_rows)
          rownames(eods_1s_id) <- NULL
        } else {
          eods_1s_id <- eods_1s_id[eods_1s_id$time <= t_e_analysis, ]
        }
        # ## Cut to nearest min
        # if(second(min(eod_data$timestamp)) != 0){
        #     t_cut <- ceiling_date(min(eod_data$timestamp), unit = "min")
        #     eod_data_cut <- eod_data[eod_data$timestamp >= t_cut, ]
        #     # Change timestamp in first row to round second timestamp
        #     eod_data_cut$timestamp[1] <- floor_date(eod_data_cut$timestamp[1], unit = "seconds")
        # } else {
        #     eod_data_cut <- eod_data
        # }

        eods_1m_id <- eods_1s_id %>%
            group_by(time = cut(time, breaks = "1 min")) %>%
            summarize(
                eod_rate_hz = mean(eod_rate_hz, na.rm=T),
                IPI_cv = sd(IPI_s, na.rm=T)/mean(IPI_s, na.rm=T),
                IPI_s = mean(IPI_s, na.rm=T),
                pp_dur_us = mean(pp_dur_us, na.rm=T),
                amp_cv = mean(amp_cv, na.rm=T),
                pp_amp_v = mean(pp_amp_v, na.rm=T),
                pp_ratio = mean(pp_ratio, na.rm=T),
                peak_fft_hz = mean(peak_fft_hz, na.rm=T),
                n_sidechange = sum(n_sidechange, na.rm=T),
                n_turns = sum(n_turns, na.rm=T)
            )
        eods_1m_id$time <- as.POSIXct(eods_1m_id$time)
  
        #   ## Cut to nearest 10 min
        #   if(minute(min(eod_data$timestamp))%%10 != 0){
        #     t_cut <- ceiling_date(min(eod_data$timestamp), unit = "10 min")
        #     eod_data_cut <- eod_data[eod_data$timestamp >= t_cut, ]
        #     # Change timestamp in first row to round 10 min timestamp
        #     eod_data_cut$timestamp[1] <- floor_date(eod_data_cut$timestamp[1], unit = "10 min")
        #   } else {
        #     eod_data_cut <- eod_data
        #     eod_data_cut$timestamp[1] <- floor_date(eod_data_cut$timestamp[1], unit = "10 min")
        #   }

        eods_10m_id <- eods_1s_id %>%
            group_by(time = cut(time, breaks = "10 min")) %>%
            summarize(
                eod_rate_hz = mean(eod_rate_hz, na.rm=T),
                IPI_cv = sd(IPI_s, na.rm=T)/mean(IPI_s, na.rm=T),
                IPI_s = mean(IPI_s, na.rm=T),
                pp_dur_us = mean(pp_dur_us, na.rm=T),
                amp_cv = mean(amp_cv, na.rm=T),
                pp_amp_v = mean(pp_amp_v, na.rm=T),
                pp_ratio = mean(pp_ratio, na.rm=T),
                peak_fft_hz = mean(peak_fft_hz, na.rm=T),
                n_sidechange = sum(n_sidechange, na.rm=T),
                n_turns = sum(n_turns, na.rm=T)
            )
        eods_10m_id$time <- as.POSIXct(eods_10m_id$time)
  


        # Add normalized values ---------------------------------------------------
        # CHANGE: only normalize eod rate and IPI (replaced 2:ncol(eods_1s) with 2:3)
        eods_1s_id <- eods_1s_id %>%
            mutate(across(2:3,
                        ~normalize_mean(.x),
                        .names = "norm_{.col}"))
        
        eods_1m_id <- eods_1m_id %>%
            mutate(across(2:3,
                        ~normalize_mean(.x),
                        .names = "norm_{.col}"))
        
        eods_10m_id <- eods_10m_id %>%
            mutate(across(2:3,
                        ~normalize_mean(.x),
                        .names = "norm_{.col}"))
  

        # Calculate rolling means -------------------------------------------------
        eods_1s_id <- eods_1s_id %>%
            mutate(across(2:ncol(eods_1s_id),
                        ~roll_mean(.x, width = 10),
                        .names = "rm_{.col}"))
        
        eods_1m_id <- eods_1m_id %>%
            mutate(across(2:ncol(eods_1m_id),
                        ~roll_mean(.x, width = 5),
                        .names = "rm_{.col}"))
        
        eods_10m_id <- eods_10m_id %>%
            mutate(across(2:ncol(eods_10m_id),
                        ~roll_mean(.x, width = 3),
                        .names = "rm_{.col}"))

        
        # Add fix timestamp
        date_start <- as.POSIXct("2020-11-15")

        eods_1s_id$fixstamp <- eods_1s_id$time
        date(eods_1s_id$fixstamp) <- date(date_start) + as.numeric((date(eods_1s_id$time) - date(eods_1s_id$time[1])))
        
        eods_1m_id$fixstamp <- eods_1m_id$time
        date(eods_1m_id$fixstamp) <- date(date_start) + as.numeric((date(eods_1m_id$time) - date(eods_1m_id$time[1])))
        
        eods_10m_id$fixstamp <- eods_10m_id$time
        date(eods_10m_id$fixstamp) <- date(date_start) + as.numeric((date(eods_10m_id$time) - date(eods_10m_id$time[1])))
        
        # Add id
        eods_1s_id$id <- sub_id
        eods_1m_id$id <- sub_id
        eods_10m_id$id <- sub_id

        # Add trial info
        eods_1s_id$trial <- sub_trial
        eods_1m_id$trial <- sub_trial
        eods_10m_id$trial <- sub_trial
        
        setwd(processed_dir)
        write.csv2(eods_1s_id, file = paste(sub_id, "_", sub_trial, "_eods_1s.csv", sep = ""))
        write.csv2(eods_1m_id, file = paste(sub_id, "_", sub_trial, "_eods_1min.csv", sep = ""))
        write.csv2(eods_10m_id, file = paste(sub_id, "_", sub_trial, "_eods_10min.csv", sep = ""))
        print("CSV saved")
        
        # Plots -------------------------------------------------------------------
        
        eods_10m_id$time <- as.POSIXct(eods_10m_id$time)
        eodr_plot <- ggplot(eods_10m_id, aes(x = time, y = eod_rate_hz))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_eod_rate_hz))+
            xlab("Daytime")+
            ylab("EOD rate (Hz)")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        intcv_plot <- ggplot(eods_10m_id, aes(x = time, y = IPI_cv))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_IPI_cv))+
            xlab("Daytime")+
            ylab("Inter-pulse-interval CV")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        amp_plot <- ggplot(eods_10m_id, aes(x = time, y = pp_amp_v))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_pp_amp_v))+
            xlab("Daytime")+
            ylab("Mean EOD amplitude (V)")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        ampcv_plot <- ggplot(eods_10m_id, aes(x = time, y = amp_cv))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_amp_cv))+
            xlab("Daytime")+
            ylab("Amplitude CV")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        fft_plot <- ggplot(eods_10m_id, aes(x = time, y = peak_fft_hz))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_peak_fft_hz))+
            xlab("Daytime")+
            ylab("Mean peak fft freq. (Hz)")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        pp_ratio_plot <- ggplot(eods_10m_id, aes(x = time, y = pp_ratio))+
            geom_point(alpha = .5)+
            geom_line(aes(x = time, y = rm_pp_ratio))+
            xlab("Daytime")+
            ylab("Mean peak-to-peak ratio")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)
        
        sidechange_plot <- ggplot(eods_10m_id, aes(x = time, y = n_sidechange))+
            geom_line(lty = 2)+
            xlab("Daytime")+
            ylab("Shuttles between sides")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)

        turns_plot <- ggplot(eods_10m_id, aes(x = time, y = n_turns))+
            geom_line(lty = 2)+
            xlab("Daytime")+
            ylab("Turns (orientation changes)")+
            scale_x_datetime(labels = date_format("%H:%M", tz = Sys.timezone()), name = NULL)+
            theme_bw(base_size = 14)

        
        comb_plot <- ggarrange(eodr_plot, amp_plot, intcv_plot,
                                ampcv_plot, pp_ratio_plot, sidechange_plot, 
                                fft_plot, turns_plot, nrow = 3, ncol = 3, align = "hv",
                                common.legend = T, legend = "bottom")
        comb_plot <- annotate_figure(comb_plot, 
                        top = text_grob(paste(sub_id, " EOD overview", sep = ""), size = 14)
        )
        
        setwd(plot_dir)
        ggexport(comb_plot, filename = paste(sub_id,"_", sub_trial, "_eod_overview.png", sep = ""), width = 1700, height = 1200, res = 120)
        print("Overview plot saved")

        # Add summaries to all datasets
        eods_1s_all <- rbind(eods_1s_all, eods_1s_id)
        eods_1m_all <- rbind(eods_1m_all, eods_1m_id)
        eods_10m_all <- rbind(eods_10m_all, eods_10m_id)
    }
}
  
# Save all datasets --------------------------------------------------------
setwd(processed_dir)
write.csv2(eods_1s_all, file = "all_eods_1s.csv")
write.csv2(eods_1m_all, file = "all_eods_1min.csv")
write.csv2(eods_10m_all, file = "all_eods_10min.csv")
