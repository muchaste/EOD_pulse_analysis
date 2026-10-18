# R script for selection of electric eel events from extracted data
# 1. Load all_event_summaries.csv files from all recording sessions
# 2. Subset events by duration and mean amplitude (remove outliers, create histograms, ggplot2)
# 3. Load pulse-level eod tables from subsetted events to compute median IPI, IPI cv, amplitude CV (from real amplitudes, not absolute!)
# 4. Aggregate computed metrics across sessions and generate summary statistics and histograms
# 5. Select events based on computed metrics based on histograms:
# 5.1 Criteria for all events:
#      - minimum number of pulses: 30
#      - good signal-to-noise ratio: Top 20-30% mean amplitude
#      - remove events with extreme outliers in duration or mean amplitude
#      - isolated events: 1 min before and after no other events
# 5.2 Transient electric eel movements:
#      - high amplitude CV
#      - moderate median IPI
#      - brief duration
#      - moderate-high mean amplitude
# 5.3 Sustained electric eel activity:
#      - low amplitude CV
#      - high median IPI
#      - long duration
#      - moderate mean amplitude
# 5.4 Hunting electric eel activity:
#      - very low median IPI
#      - short duration
#      - very high amplitude
# 
# 6. Review and manually curate selected events if necessary
# 7. Save curated events to a new CSV file for further analysis
# Use ggplot2+ggpubr for visualizations, tidyverse for data manipulation

library(tidyverse)
library(ggpubr)
library(lubridate)

rm(list = ls())

# Root folder to scan recursively for all_event_summaries.csv
root_dir <- "F:/First Trip/Raleighvallen/"
output_dir <- "F:/First Trip/Raleighvallen/eel_event_subsetting"
dir.create(output_dir, showWarnings = FALSE)

# Assumes these extractions used extraction_method='single_channel' with ignore_channel=TRUE
# (events clustered by time only) - so n_eods/duration_seconds/mean_amplitude/max_amplitude/
# median_ipi_seconds/ipi_cv in the summary are already correct whole-event, cross-file values.
# Only amplitude CV (needs signed pulse_orientation, not in the summary) requires re-reading
# the per-event eod_table.csv files, done in Phase C below for pre-filtered candidates only.

# ==============================================================================
# PHASE A - Discover and aggregate all_event_summaries.csv files across sessions
# ==============================================================================

summary_files <- list.files(root_dir, pattern = "all_event_summaries\\.csv$",
                            recursive = TRUE, full.names = TRUE)
cat(sprintf("Found %d all_event_summaries.csv files\n", length(summary_files)))

all_events <- NULL
for (f in summary_files) {
  session_events <- read.csv(f, sep = ",", dec = ".", stringsAsFactors = FALSE)
  session_events$session_path <- dirname(f)
  # Folder structure: .../{Trip}/{Location}/{EOD N}/recordings/{extraction}/all_event_summaries.csv
  session_events$eod_recorder <- basename(dirname(dirname(dirname(f))))
  session_events$location <- basename(dirname(dirname(dirname(dirname(f)))))
  session_events$trip <- basename(dirname(dirname(dirname(dirname(dirname(f))))))
  all_events <- bind_rows(all_events, session_events)
}

all_events$event_start_time <- ymd_hms(all_events$event_start_time, quiet = TRUE)
all_events$event_end_time <- ymd_hms(all_events$event_end_time, quiet = TRUE)
# Unique key: event_id is only unique WITHIN a session, not across all sessions
all_events$session_event_key <- paste(all_events$session_path, all_events$event_id, sep = "::")

cat(sprintf("Total events across all sessions: %d\n", nrow(all_events)))

# ==============================================================================
# PHASE B - Pre-filter on summary metrics alone (no per-event files needed)
# ==============================================================================

# --- Isolation check: run on the FULL per-session event list, before any other
# filtering removes potential neighbours ---
isolation_min_gap_s <- 60

all_events <- all_events %>%
  group_by(session_path) %>%
  arrange(event_start_time, .by_group = TRUE) %>%
  mutate(
    gap_before_s = as.numeric(difftime(event_start_time, lag(event_end_time), units = "secs")),
    gap_after_s = as.numeric(difftime(lead(event_start_time), event_end_time, units = "secs")),
    gap_before_s = ifelse(is.na(gap_before_s), Inf, gap_before_s),
    gap_after_s = ifelse(is.na(gap_after_s), Inf, gap_after_s),
    is_isolated = gap_before_s >= isolation_min_gap_s & gap_after_s >= isolation_min_gap_s
  ) %>%
  ungroup()

cat(sprintf("Isolated events: %d / %d\n", sum(all_events$is_isolated), nrow(all_events)))

# filter out non-isolated events before proceeding with other pre-filters
all_events <- all_events %>%
  filter(is_isolated)

# --- Duration outlier removal (IQR on log-duration, pooled globally) ---
log_dur <- log(all_events$duration_seconds)
dur_q1 <- quantile(log_dur, 0.25, na.rm = TRUE)
dur_q3 <- quantile(log_dur, 0.75, na.rm = TRUE)
dur_iqr <- dur_q3 - dur_q1
dur_lower <- exp(dur_q1 - 1.5 * dur_iqr)
dur_upper <- exp(dur_q3 + 1.5 * dur_iqr)

# alternative: pre-set durations based on heuristics
dur_lower <- 10
dur_upper <- 120

# --- Amplitude pre-filter: top 20-30% mean_amplitude, pooled globally ---
amp_cutoff <- quantile(all_events$mean_amplitude, 0.75, na.rm = TRUE)

# Histograms to validate cutoffs before locking them in
p_dur <- ggplot(all_events, aes(x = duration_seconds)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  geom_vline(xintercept = c(dur_lower, dur_upper), color = "red", linetype = "dashed") +
  labs(title = "Event duration (log scale)", x = "Duration (s)")

p_amp <- ggplot(all_events, aes(x = mean_amplitude)) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = amp_cutoff, color = "red", linetype = "dashed") +
  labs(title = "Mean event amplitude", x = "Mean amplitude")

p_n_eods <- ggplot(all_events, aes(x = n_eods)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  geom_vline(xintercept = 30, color = "red", linetype = "dashed") +
  labs(title = "Pulses per event", x = "n_eods")

ggarrange(p_dur, p_amp, p_n_eods, ncol = 3)
ggsave(file.path(output_dir, "prefilter_histograms.png"), width = 15, height = 5)

candidate_events <- all_events %>%
  filter(
    n_eods >= 30,
    is_isolated,
    duration_seconds >= dur_lower, duration_seconds <= dur_upper,
    mean_amplitude >= amp_cutoff
  )

cat(sprintf("Candidate events after summary-level pre-filter: %d / %d\n",
            nrow(candidate_events), nrow(all_events)))

# ==============================================================================
# PHASE C - Pulse-level signed amplitude CV (only for pre-filtered candidates)
# ==============================================================================

candidate_events$amp_cv <- NA_real_

for (i in seq_len(nrow(candidate_events))) {
  session_path_i <- candidate_events$session_path[i]
  event_id_i <- candidate_events$event_id[i]

  eod_table_file <- list.files(
    session_path_i,
    pattern = paste0("_event_", event_id_i, "_eod_table\\.csv$"),
    full.names = TRUE
  )
  if (length(eod_table_file) == 0) {
    cat(sprintf("  Warning: no eod_table found for event %d in %s\n", event_id_i, session_path_i))
    next
  }
  # Multi-file/part events can save more than one eod_table (same event_id, part suffix) -
  # the whole-event table always carries no part suffix, so prefer the shortest match.
  eod_table_file <- eod_table_file[which.min(nchar(eod_table_file))]

  pulses <- read.csv(eod_table_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  signed_amplitude <- ifelse(pulses$pulse_orientation == "HN", -pulses$eod_amplitude, pulses$eod_amplitude)
  candidate_events$amp_cv[i] <- sd(signed_amplitude) / abs(mean(signed_amplitude))
}

cat(sprintf("Signed amplitude CV computed for %d / %d candidates\n",
            sum(!is.na(candidate_events$amp_cv)), nrow(candidate_events)))


# ==============================================================================
# INTERLUDE - copy pngs of candidate events into output folder
# naming convention: <session_path>event_<event_id>_detection.png
# ==============================================================================
for (i in seq_len(nrow(candidate_events))) {
  session_path_i <- candidate_events$session_path[i]
  event_id_i <- candidate_events$event_id[i]

  if (event_id_i < 10) {
    id_pattern <- paste0("event_00", event_id_i, "_detection\\.png$")
  } else if (event_id_i < 100) {
    id_pattern <- paste0("event_0", event_id_i, "_detection\\.png$")
  } else {
    id_pattern <- paste0("event_", event_id_i, "_detection\\.png$")
  }

  detection_png_file <- list.files(
    session_path_i,
    pattern = id_pattern,
    full.names = TRUE
  )
  if (length(detection_png_file) == 0) {
    cat(sprintf("  Warning: no detection PNG found for event %d in %s\n", event_id_i, session_path_i))
    next
  }
  # Multi-file/part events can save more than one detection PNG (same event_id, part suffix) -
  # the whole-event PNG always carries no part suffix, so prefer the shortest match.
  detection_png_file <- detection_png_file[which.min(nchar(detection_png_file))]

  file.copy(detection_png_file, file.path(output_dir, basename(detection_png_file)), overwrite = TRUE)
}


# ==============================================================================
# PHASE D - Classify into behavioral categories
# ==============================================================================

# Quantile cutoffs computed WITHIN the candidate pool (not the full dataset), since this
# pool is already restricted to decent signal-to-noise, isolated, non-outlier events.
q_amp_cv  <- quantile(candidate_events$amp_cv, c(0.33, 0.67), na.rm = TRUE)
q_ipi     <- quantile(candidate_events$median_ipi_seconds, c(0.1, 0.33, 0.67), na.rm = TRUE)
q_dur     <- quantile(candidate_events$duration_seconds, c(0.33, 0.67), na.rm = TRUE)
q_amp     <- quantile(candidate_events$mean_amplitude, c(0.33, 0.9), na.rm = TRUE)

candidate_events <- candidate_events %>%
  mutate(
    is_transient = amp_cv >= q_amp_cv[2] &
      median_ipi_seconds > q_ipi[2] & median_ipi_seconds < q_ipi[3] &
      duration_seconds <= q_dur[1] &
      mean_amplitude >= q_amp[1],
    is_resting = amp_cv <= q_amp_cv[1] &
      median_ipi_seconds >= q_ipi[3] &
      duration_seconds >= q_dur[2] &
      mean_amplitude > q_amp[1] & mean_amplitude < q_amp[2],
    is_hunting = median_ipi_seconds <= q_ipi[1] &
      duration_seconds <= q_dur[1] &
      mean_amplitude >= q_amp[2],
    n_categories_matched = is_transient + is_resting + is_hunting,
    category = case_when(
      n_categories_matched == 0 ~ "unclassified",
      n_categories_matched > 1 ~ "ambiguous",
      is_transient ~ "transient",
      is_resting ~ "resting",
      is_hunting ~ "hunting"
    )
  )

cat("Category counts:\n")
print(table(candidate_events$category))

# Diagnostic scatter: duration vs IPI, colored by category, sized by amplitude
p_classify <- ggplot(candidate_events, aes(x = median_ipi_seconds, y = duration_seconds,
                                           color = category, size = mean_amplitude)) +
  geom_point(alpha = 0.7) +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Candidate events by behavioral category",
       x = "Median IPI (s)", y = "Duration (s)")

p_classify_cv <- ggplot(candidate_events, aes(x = category, y = amp_cv, fill = category)) +
  geom_boxplot() +
  labs(title = "Amplitude CV by category")

ggarrange(p_classify, p_classify_cv, ncol = 2)
ggsave(file.path(output_dir, "classification_diagnostics.png"), width = 14, height = 6)

# ==============================================================================
# PHASE E - Manual curation & export
# ==============================================================================

write.csv(candidate_events, file.path(output_dir, "candidate_events_for_review.csv"), row.names = FALSE)
cat(sprintf("Saved %d candidate events for manual review to: %s\n",
            nrow(candidate_events), file.path(output_dir, "candidate_events_for_review.csv")))

# After manual review (step 6), load the curated subset (e.g. with a manually added
# 'keep' column) and save the final curated set here:
# curated_events <- read.csv(file.path(output_dir, "candidate_events_for_review.csv")) %>%
#   filter(keep == TRUE)
# write.csv(curated_events, file.path(output_dir, "curated_events_final.csv"), row.names = FALSE)

