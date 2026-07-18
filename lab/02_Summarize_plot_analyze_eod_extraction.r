
# Load eod extraction results and create summary datasets, plots, and statistical analyses
# Uses output eod tables from lab/01_Shuttle_box_Matlab_pulse_extraction.py

library(tidyverse)
library(ggpubr)
library(roll)
library(lubridate)
library(scales)

rm(list = ls())

normalize_baseline <- function(vector, time, baseline_start, baseline_end){
  baseline_mean <- mean(vector[time >= baseline_start & time < baseline_end], na.rm = T)
  norm_vector <- vector/baseline_mean
  return(norm_vector)
}

# Set root directory (output root of 01_Shuttle_box_Matlab_pulse_extraction.py)
root_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction"
processed_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/processed"
plot_dir <- "F:/P degeni responses to hypoxia/02 Analysis Output/eod_extraction/plots/"

setwd(root_dir)

# Analysis window and processing constants -----------------------------------
analysis_start_tod <- "08:40:00"    # time of day the analysis window starts, for every trial
analysis_duration_min <- 200        # length of the analysis window in minutes
baseline_duration_min <- 40         # length of the baseline window (from analysis start) used for normalization
roll_width_1s <- 10
roll_width_1m <- 5
roll_width_10m <- 3

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
      tryCatch({
        eod_file_list_subset <- eod_file_list %>% filter(str_equal(trial, sub_trial) & str_equal(id, sub_id))

        # Read all metadata up front so files can be sorted chronologically and checked for overlaps
        file_metadata_list <- lapply(eod_file_list_subset$file_path, function(file){
            metadata_file <- str_replace(basename(file), "_pulse_extraction_results.csv", "_metadata.csv")
            read.csv(file.path(dirname(file), metadata_file), header=TRUE)
        })
        file_starts <- as.POSIXct(sapply(file_metadata_list, function(m) m$file_timestamp[1]), format="%Y-%m-%d %H:%M:%OS", tz="CET")
        file_ends <- as.POSIXct(sapply(file_metadata_list, function(m) m$file_end_timestamp[1]), format="%Y-%m-%d %H:%M:%OS", tz="CET")

        file_order <- order(file_starts)
        eod_file_list_subset <- eod_file_list_subset[file_order, ]
        file_starts <- file_starts[file_order]
        file_ends <- file_ends[file_order]

        if(length(file_starts) > 1 && any(file_starts[-1] < file_ends[-length(file_ends)])){
          warning(paste0(sub_id, "/", sub_trial, ": overlapping file time ranges detected"))
        }

        # Build one raw pulse table and one recording-coverage table across all files -----
        pulses_id <- NULL
        coverage_id <- NULL
        for(i in seq_along(eod_file_list_subset$file_path)) {
            file <- eod_file_list_subset$file_path[i]
            file_start <- file_starts[i]
            file_end <- file_ends[i]

            # Seconds actually covered by this file's recording, regardless of whether any EODs were detected
            coverage_id <- bind_rows(coverage_id, tibble(time = seq(floor_date(file_start, "sec"), floor_date(file_end, "sec"), by = "1 sec")))

            eod_data <- read.csv(file, header=TRUE)
            if(nrow(eod_data) == 0) next  # no detections in this file; its coverage is already recorded above

            eod_data$timestamp <- as.POSIXct(eod_data$timestamp, format="%Y-%m-%d %H:%M:%OS", tz="CET")
            eod_data <- eod_data[order(eod_data$timestamp, decreasing = FALSE), ]

            # Add side column based on eod_channel column
            eod_data$side <- ifelse(eod_data$eod_channel > 1, "right", "left")

            # Orientation column (if side == left: left_h (chan 1), if side == right: right_h (chan 3))
            eod_data$orientation <- eod_data$orient_left_h
            eod_data$orientation[eod_data$side == "right"] <- eod_data$orient_right_h[eod_data$side == "right"]

            # Use amplitude also only from horizontal channels to get an estimate of movement-based amplitude variation
            eod_data$amp_movement <- eod_data$amp_left_h
            eod_data$amp_movement[eod_data$side == "right"] <- eod_data$amp_right_h[eod_data$side == "right"]

            # Add IPI column (inter-pulse interval in seconds); kept file-scoped since relative_time_s resets per file
            eod_data$IPI <- c(NA, as.numeric(diff(eod_data$relative_time_s)))

            pulses_id <- bind_rows(pulses_id, eod_data)
        }

        if(is.null(pulses_id) || nrow(pulses_id) == 0){
          stop(paste0(sub_id, "/", sub_trial, ": no EODs detected in any file, skipping"))
        }

        pulses_id <- pulses_id[order(pulses_id$timestamp, decreasing = FALSE), ]
        coverage_id <- coverage_id[order(coverage_id$time, decreasing = FALSE), ] %>% distinct(time, .keep_all = TRUE)
        coverage_id$recorded <- TRUE

        # Global side/orientation change flags, computed once over the full sorted sequence
        # (not reset per time bin, so transitions across bin boundaries are not lost)
        pulses_id$side_change <- c(FALSE, diff(as.numeric(as.factor(pulses_id$side))) != 0)
        pulses_id$turn_change <- c(FALSE, diff(as.numeric(as.factor(pulses_id$orientation))) != 0)

        # Build the fixed analysis-window backbone directly on the coverage table ---------
        t_s_analysis <- as.POSIXct(paste(format(min(coverage_id$time), "%Y-%m-%d"), analysis_start_tod), format="%Y-%m-%d %H:%M:%S", tz="CET")
        t_e_analysis <- t_s_analysis + analysis_duration_min*60 - 1

        backbone <- tibble(time = seq(t_s_analysis, t_e_analysis, by = "1 sec"))
        coverage_id <- backbone %>%
            left_join(coverage_id %>% select(time, recorded), by = "time") %>%
            mutate(recorded = replace_na(recorded, FALSE))

        # Unified aggregation: identical formulas at every resolution, computed from pooled
        # raw pulses + the actual recorded duration (not from an already-aggregated finer table) --

        # per second
        cov_1s <- coverage_id %>%
            group_by(time = floor_date(time, "1 sec")) %>%
            summarize(recorded_seconds = sum(recorded), .groups = "drop") %>%
            mutate(frac_recorded = recorded_seconds/1)

        pulse_1s <- pulses_id %>%
            group_by(time = floor_date(timestamp, "1 sec")) %>%
            summarize(
                n_pulses = n(),
                IPI_cv = sd(IPI, na.rm=T)/mean(IPI, na.rm=T),
                IPI_s = mean(IPI, na.rm=T),
                pp_dur_us = mean(eod_width_us, na.rm=T),
                amp_cv = sd(amp_movement, na.rm=T)/mean(amp_movement, na.rm=T),
                pp_amp_v = mean(amp_movement, na.rm=T),
                pp_ratio = mean(eod_amplitude_ratio, na.rm=T),
                peak_fft_hz = mean(fft_freq_max, na.rm=T),
                n_sidechange = sum(side_change, na.rm=T),
                n_turns = sum(turn_change, na.rm=T),
                .groups = "drop"
            )

        eods_1s_id <- cov_1s %>%
            left_join(pulse_1s, by = "time") %>%
            mutate(
                n_pulses = replace_na(n_pulses, 0),
                eod_rate_hz = ifelse(recorded_seconds > 0, n_pulses/recorded_seconds, NA_real_)
            ) %>%
            select(time, eod_rate_hz, IPI_cv, IPI_s, pp_dur_us, amp_cv, pp_amp_v,
                   pp_ratio, peak_fft_hz, n_sidechange, n_turns, frac_recorded)

        # per minute
        cov_1m <- coverage_id %>%
            group_by(time = floor_date(time, "1 min")) %>%
            summarize(recorded_seconds = sum(recorded), .groups = "drop") %>%
            mutate(frac_recorded = recorded_seconds/60)

        pulse_1m <- pulses_id %>%
            group_by(time = floor_date(timestamp, "1 min")) %>%
            summarize(
                n_pulses = n(),
                IPI_cv = sd(IPI, na.rm=T)/mean(IPI, na.rm=T),
                IPI_s = mean(IPI, na.rm=T),
                pp_dur_us = mean(eod_width_us, na.rm=T),
                amp_cv = sd(amp_movement, na.rm=T)/mean(amp_movement, na.rm=T),
                pp_amp_v = mean(amp_movement, na.rm=T),
                pp_ratio = mean(eod_amplitude_ratio, na.rm=T),
                peak_fft_hz = mean(fft_freq_max, na.rm=T),
                n_sidechange = sum(side_change, na.rm=T),
                n_turns = sum(turn_change, na.rm=T),
                .groups = "drop"
            )

        eods_1m_id <- cov_1m %>%
            left_join(pulse_1m, by = "time") %>%
            mutate(
                n_pulses = replace_na(n_pulses, 0),
                eod_rate_hz = ifelse(recorded_seconds > 0, n_pulses/recorded_seconds, NA_real_)
            ) %>%
            select(time, eod_rate_hz, IPI_cv, IPI_s, pp_dur_us, amp_cv, pp_amp_v,
                   pp_ratio, peak_fft_hz, n_sidechange, n_turns, frac_recorded)

        # per 10 minutes
        cov_10m <- coverage_id %>%
            group_by(time = floor_date(time, "10 min")) %>%
            summarize(recorded_seconds = sum(recorded), .groups = "drop") %>%
            mutate(frac_recorded = recorded_seconds/600)

        pulse_10m <- pulses_id %>%
            group_by(time = floor_date(timestamp, "10 min")) %>%
            summarize(
                n_pulses = n(),
                IPI_cv = sd(IPI, na.rm=T)/mean(IPI, na.rm=T),
                IPI_s = mean(IPI, na.rm=T),
                pp_dur_us = mean(eod_width_us, na.rm=T),
                amp_cv = sd(amp_movement, na.rm=T)/mean(amp_movement, na.rm=T),
                pp_amp_v = mean(amp_movement, na.rm=T),
                pp_ratio = mean(eod_amplitude_ratio, na.rm=T),
                peak_fft_hz = mean(fft_freq_max, na.rm=T),
                n_sidechange = sum(side_change, na.rm=T),
                n_turns = sum(turn_change, na.rm=T),
                .groups = "drop"
            )

        eods_10m_id <- cov_10m %>%
            left_join(pulse_10m, by = "time") %>%
            mutate(
                n_pulses = replace_na(n_pulses, 0),
                eod_rate_hz = ifelse(recorded_seconds > 0, n_pulses/recorded_seconds, NA_real_)
            ) %>%
            select(time, eod_rate_hz, IPI_cv, IPI_s, pp_dur_us, amp_cv, pp_amp_v,
                   pp_ratio, peak_fft_hz, n_sidechange, n_turns, frac_recorded)


        # Add normalized values ---------------------------------------------------
        # Normalize relative to the mean over the first `baseline_duration_min` minutes
        # of the trial (not the whole-window mean, which would include the treatment period)
        baseline_end <- t_s_analysis + baseline_duration_min*60

        eods_1s_id <- eods_1s_id %>%
            mutate(across(2:3,
                        ~normalize_baseline(.x, time, t_s_analysis, baseline_end),
                        .names = "norm_{.col}"))
        
        eods_1m_id <- eods_1m_id %>%
            mutate(across(2:3,
                        ~normalize_baseline(.x, time, t_s_analysis, baseline_end),
                        .names = "norm_{.col}"))
        
        eods_10m_id <- eods_10m_id %>%
            mutate(across(2:3,
                        ~normalize_baseline(.x, time, t_s_analysis, baseline_end),
                        .names = "norm_{.col}"))
  

        # Calculate rolling means -------------------------------------------------
        # min_obs = 1 so rolling means don't propagate NA further than the actual gaps
        eods_1s_id <- eods_1s_id %>%
            mutate(across(2:ncol(eods_1s_id),
                        ~roll_mean(.x, width = roll_width_1s, min_obs = 1),
                        .names = "rm_{.col}"))
        
        eods_1m_id <- eods_1m_id %>%
            mutate(across(2:ncol(eods_1m_id),
                        ~roll_mean(.x, width = roll_width_1m, min_obs = 1),
                        .names = "rm_{.col}"))
        
        eods_10m_id <- eods_10m_id %>%
            mutate(across(2:ncol(eods_10m_id),
                        ~roll_mean(.x, width = roll_width_10m, min_obs = 1),
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
        write.csv(eods_1s_id, file = paste(sub_id, "_", sub_trial, "_eods_1s.csv", sep = ""), row.names = FALSE)
        write.csv(eods_1m_id, file = paste(sub_id, "_", sub_trial, "_eods_1min.csv", sep = ""), row.names = FALSE)
        write.csv(eods_10m_id, file = paste(sub_id, "_", sub_trial, "_eods_10min.csv", sep = ""), row.names = FALSE)
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
        eods_1s_all <- bind_rows(eods_1s_all, eods_1s_id)
        eods_1m_all <- bind_rows(eods_1m_all, eods_1m_id)
        eods_10m_all <- bind_rows(eods_10m_all, eods_10m_id)

      }, error = function(e){
        message(paste0("Skipping ", sub_id, "/", sub_trial, ": ", conditionMessage(e)))
      })
    }
}
  
# Save all datasets --------------------------------------------------------
setwd(processed_dir)
write.csv(eods_1s_all, file = "all_eods_1s.csv", row.names = FALSE)
write.csv(eods_1m_all, file = "all_eods_1min.csv", row.names = FALSE)
write.csv(eods_10m_all, file = "all_eods_10min.csv", row.names = FALSE)
