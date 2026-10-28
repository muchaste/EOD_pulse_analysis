# R script for selection of electric eel events from extracted data
# 1. Load all_event_summaries.csv files of the envelope extractions (path contains "Envelope") from all recording sessions
# 2. Time criteria on whole events of the low-threshold extraction: isolated (no other event within
#    isolation_min_gap_s before/after), minimum pulses, duration range
# 3. Load pulse-level eod tables of these events; require min_strong_pulses pulses >= pulse_amp_cutoff
#    (signal-to-noise gate), detect volley groups and compute event metrics (amplitude, IPI, amplitude CV,
#    polarity flips) from all pulses and from the volley pulses; these metrics go into the review table
# 4. Copy event files and save candidate table + pulses for visual review
# 5. Manual review: assign category (r = re-classify, v = volley, others excluded) in the 'category' column
# 6. Re-classify the r events into moving (m) / stationary (s) by absolute amplitude CV and median IPI, then select
#    the top n events per category ranked by the P90 amplitude of the metric pulses
#    (volley pulses for volleys, all pulses for the other categories)
# Use ggplot2+ggpubr for visualizations, tidyverse for data manipulation

library(tidyverse)
library(ggpubr)
library(lubridate)

rm(list = ls())

# Root folder to scan recursively for all_event_summaries.csv
root_dir <- "G:/First Trip/Avanavero/"
output_dir <- "G:/First Trip/Avanavero/eel_event_subsetting_win_8000"
dir.create(output_dir, showWarnings = FALSE)

# Assumes these extractions used extraction_method='single_channel' with ignore_channel=TRUE
# (events clustered by time only). Isolation and duration use the whole-event times of the
# summary; all amplitude/IPI metrics are recomputed from the pulse tables (eod_table.csv).

# ==============================================================================
# PHASE A - Discover and aggregate all_event_summaries.csv files across sessions
# ==============================================================================

summary_files <- list.files(root_dir, pattern = "all_event_summaries\\.csv$",
                            recursive = TRUE, full.names = TRUE)
cat(sprintf("Found %d all_event_summaries.csv files\n", length(summary_files)))

# Only the envelope-method extractions
summary_files <- summary_files[grepl("Envelope", summary_files, fixed = TRUE)]
summary_files <- summary_files[grepl("win_8000", summary_files, fixed = TRUE)]
cat(sprintf("Kept %d envelope extractions\n", length(summary_files)))

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
# PHASE B - Time criteria on whole events, then event metrics from the pulse tables
# ==============================================================================

pulse_amp_cutoff <- 0.005
min_pulses <- 30
min_strong_pulses <- 10
dur_lower <- 10
dur_upper <- 100
isolation_min_gap_s <- 60
# Volley pulses: groups of >= min_volley_pulses consecutive pulses (any amplitude) with IPI <= volley_max_ipi_s
volley_max_ipi_s <- 0.05
min_volley_pulses <- 5

# --- Isolation check: run on the FULL per-session event list, before any other
# filtering removes potential neighbours ---
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

# Histograms of the isolated events to validate cutoffs
isolated_events <- all_events %>%
  filter(is_isolated)

p_dur <- ggplot(isolated_events, aes(x = duration_seconds)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  geom_vline(xintercept = c(dur_lower, dur_upper), color = "red", linetype = "dashed") +
  labs(title = "Isolated event duration (log scale)", x = "Duration (s)")

p_n_eods <- ggplot(isolated_events, aes(x = n_eods)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  geom_vline(xintercept = min_pulses, color = "red", linetype = "dashed") +
  labs(title = "Pulses per isolated event", x = "n_eods")

time_candidates <- isolated_events %>%
  filter(
    n_eods >= min_pulses,
    duration_seconds >= dur_lower, duration_seconds <= dur_upper
  )

cat(sprintf("Events after time criteria: %d / %d\n", nrow(time_candidates), nrow(all_events)))

# --- Load pulse tables of the time candidates (every amplitude) ---
pulse_list <- list()
for (session_path_i in unique(time_candidates$session_path)) {
  table_files <- list.files(session_path_i, pattern = "_eod_table\\.csv$", full.names = TRUE)
  session_event_ids <- time_candidates$event_id[time_candidates$session_path == session_path_i]

  for (event_id_i in session_event_ids) {
    event_files <- table_files[grepl(paste0("_event_", event_id_i, "_eod_table.csv"),
                                     basename(table_files), fixed = TRUE)]
    if (length(event_files) == 0) {
      cat(sprintf("  Warning: no eod_table found for event %d in %s\n", event_id_i, session_path_i))
      next
    }
    # Multi-file/part events can save more than one eod_table (same event_id, part suffix) -
    # the whole-event table always carries no part suffix, so prefer the shortest match.
    event_file <- event_files[which.min(nchar(event_files))]

    pulses <- read_csv(event_file, show_col_types = FALSE,
                       col_types = cols_only(timestamp = col_character(),
                                             eod_amplitude = col_double(),
                                             pulse_orientation = col_character()))
    pulses$session_path <- session_path_i
    pulses$event_id <- event_id_i
    pulse_list[[length(pulse_list) + 1]] <- pulses
  }
}

all_pulses <- bind_rows(pulse_list) %>%
  mutate(
    time = ymd_hms(timestamp, quiet = TRUE),
    signed_amplitude = ifelse(pulse_orientation == "HN", -eod_amplitude, eod_amplitude)
  ) %>%
  select(-timestamp)

cat(sprintf("Loaded %d pulses from %d events\n", nrow(all_pulses), nrow(time_candidates)))

# --- Signal-to-noise gate: at least min_strong_pulses pulses >= pulse_amp_cutoff ---
strong_counts <- all_pulses %>%
  group_by(session_path, event_id) %>%
  summarise(n_strong_eods = sum(eod_amplitude >= pulse_amp_cutoff), .groups = "drop") %>%
  filter(n_strong_eods >= min_strong_pulses)

# --- Pulses of the candidate events with strong / volley flags ---
# Stray electrolocation pulses before/between/after volleys are not part of a volley group.
candidate_pulses <- all_pulses %>%
  semi_join(strong_counts, by = c("session_path", "event_id")) %>%
  group_by(session_path, event_id) %>%
  arrange(time, .by_group = TRUE) %>%
  mutate(volley_group = cumsum(c(TRUE, diff(as.numeric(time)) > volley_max_ipi_s))) %>%
  group_by(session_path, event_id, volley_group) %>%
  mutate(in_volley = n() >= min_volley_pulses) %>%
  ungroup() %>%
  mutate(is_strong = eod_amplitude >= pulse_amp_cutoff)

# --- Event metrics from all pulses (pulse_set "all") and from the volley pulses (pulse_set "volley") ---
# IPIs are only taken between consecutive pulses of the same IPI group (all pulses: one group; volley
# pulses: the volley group), so gaps between volleys do not enter. The P90 amplitude is the proximity
# proxy for ranking, independent of the share of weak pulses. metric_amp_cv_signed is kept next to
# metric_amp_cv_abs for comparison: the signed version explodes when the polarity inverts.
metric_pulses <- bind_rows(
  candidate_pulses %>% mutate(pulse_set = "all", ipi_group = 1L),
  candidate_pulses %>% filter(in_volley) %>% mutate(pulse_set = "volley", ipi_group = volley_group)
)

event_metrics <- metric_pulses %>%
  group_by(session_path, event_id, pulse_set) %>%
  arrange(time, .by_group = TRUE) %>%
  mutate(ipi_seconds = ifelse(ipi_group == lag(ipi_group), as.numeric(time) - lag(as.numeric(time)), NA_real_)) %>%
  summarise(
    selection_amplitude = quantile(eod_amplitude, 0.9),
    metric_median_amplitude = median(eod_amplitude),
    metric_max_amplitude = max(eod_amplitude),
    metric_strong_fraction = mean(is_strong),
    metric_median_ipi_seconds = median(ipi_seconds, na.rm = TRUE),
    metric_ipi_iqr_ratio = IQR(ipi_seconds, na.rm = TRUE) / median(ipi_seconds, na.rm = TRUE),
    metric_amp_cv_signed = sd(signed_amplitude) / abs(mean(signed_amplitude)),
    metric_amp_cv_abs = sd(eod_amplitude) / mean(eod_amplitude),
    metric_amp_range_log10 = log10(quantile(eod_amplitude, 0.9) / quantile(eod_amplitude, 0.1)),
    metric_flip_rate = mean(diff(sign(signed_amplitude)) != 0),
    t_first_eod = min(time), t_last_eod = max(time), n_eods_metric = n(),
    .groups = "drop"
  ) %>%
  mutate(metric_duration_seconds = as.numeric(difftime(t_last_eod, t_first_eod, units = "secs")),
         metric_rate_hz = n_eods_metric / metric_duration_seconds)

volley_metrics <- candidate_pulses %>%
  filter(in_volley) %>%
  group_by(session_path, event_id) %>%
  summarise(t_start_first_volley = min(time), t_end_last_volley = max(time),
            n_volleys = n_distinct(volley_group), n_volley_eods = n(), .groups = "drop")

# Review table: metrics side by side for all pulses (all_*) and volley pulses (volley_*, NA without a volley group)
review_metrics <- event_metrics %>%
  filter(pulse_set == "all") %>%
  select(-pulse_set) %>%
  rename_with(~ paste0("all_", sub("^metric_", "", .x)), -c(session_path, event_id)) %>%
  left_join(event_metrics %>%
              filter(pulse_set == "volley") %>%
              select(-pulse_set) %>%
              rename_with(~ paste0("volley_", sub("^metric_", "", .x)), -c(session_path, event_id)),
            by = c("session_path", "event_id"))

candidate_events <- time_candidates %>%
  select(session_path, event_id, session_event_key, eod_recorder, location, trip,
         event_start_time, event_end_time, duration_seconds, n_eods, gap_before_s, gap_after_s) %>%
  inner_join(strong_counts, by = c("session_path", "event_id")) %>%
  left_join(review_metrics, by = c("session_path", "event_id")) %>%
  left_join(volley_metrics, by = c("session_path", "event_id"))

cat(sprintf("Candidate events with >= %d pulses >= %g: %d / %d\n", min_strong_pulses, pulse_amp_cutoff,
            nrow(candidate_events), nrow(time_candidates)))

p_amp <- ggplot(candidate_events, aes(x = all_selection_amplitude)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  labs(title = "P90 amplitude (all pulses)", x = "P90 amplitude")

ggarrange(p_dur, p_n_eods, p_amp, ncol = 3)
ggsave(file.path(output_dir, "prefilter_histograms.png"), width = 15, height = 5)


# ==============================================================================
# INTERLUDE - copy data of candidate events into output folder
# Copies wav, eod_table, waveforms npz, waveforms metadata json and detection png.
# The png gets the <logger>-<timestamp> prefix parsed from the wav file name.
# ==============================================================================
for (session_path_i in unique(candidate_events$session_path)) {
  session_files <- list.files(session_path_i, full.names = FALSE)
  session_event_ids <- unique(candidate_events$event_id[candidate_events$session_path == session_path_i])

  for (event_id_i in session_event_ids) {
    event_suffix <- paste0("_event_", event_id_i)
    png_name <- sprintf("event_%03d_detection.png", as.integer(event_id_i))

    wav_name <- session_files[endsWith(session_files, paste0(event_suffix, ".wav"))]
    if (length(wav_name) == 0) {
      cat(sprintf("  Warning: no .wav file found for event %d in %s\n", event_id_i, session_path_i))
      next
    }
    wav_name <- wav_name[1]
    event_prefix <- sub(paste0(event_suffix, "\\.wav$"), "", wav_name)

    data_names <- c(
      wav_name,
      paste0(event_prefix, event_suffix, "_eod_table.csv"),
      paste0(event_prefix, event_suffix, "_waveforms_concatenated.npz"),
      paste0(event_prefix, event_suffix, "_waveforms_metadata.json")
    )
    for (data_name in data_names) {
      if (!(data_name %in% session_files)) {
        cat(sprintf("  Warning: %s not found in %s\n", data_name, session_path_i))
        next
      }
      file.copy(file.path(session_path_i, data_name), file.path(output_dir, data_name), overwrite = TRUE)
    }

    if (png_name %in% session_files) {
      file.copy(file.path(session_path_i, png_name),
                file.path(output_dir, paste0(event_prefix, "_", png_name)), overwrite = TRUE)
    } else {
      cat(sprintf("  Warning: no detection PNG found for event %d in %s\n", event_id_i, session_path_i))
    }
  }
}

# ==============================================================================
# PHASE D - Export for manual review
# ==============================================================================

# Fill the 'category' column by hand after looking at the detection PNGs:
#   r = re-classify (moving or stationary, decided numerically in Phase E), v = volley,
#   c = co-occurring fish, w = wave-fish (continuous biphasic sinusoidal discharge, 200-1200 Hz),
#   x = noise contaminated / faulty, wv/vw/pv = volleys together with other fish pulses or wave-fish
#   Only r and v are carried into Phase E.
# Save the annotated table as candidate_events_reviewed.csv: re-running this phase overwrites
# candidate_events_for_review.csv with an empty 'category' column.
candidate_events$category <- ""

write.csv(candidate_events, file.path(output_dir, "candidate_events_for_review.csv"), row.names = FALSE)
write.csv(candidate_pulses %>%
            select(session_path, event_id, time, eod_amplitude, signed_amplitude, is_strong, in_volley, volley_group),
          file.path(output_dir, "candidate_pulses_for_review.csv"), row.names = FALSE)
cat(sprintf("Saved %d candidate events for manual review to: %s\n",
            nrow(candidate_events), file.path(output_dir, "candidate_events_for_review.csv")))



# ==============================================================================
# PHASE E - Top n events per category, ranked by the P90 amplitude of the metric pulses
# (volley pulses for volleys, all pulses for the other categories)
# Reads the manually annotated candidate_events_reviewed.csv
# ==============================================================================

n_per_category <- 20
min_volley_eods_selection <- 30 # minimum volley pulses for a volley event to enter the selection
selected_categories <- c("m", "s", "v") # m:moving, s:stationary, v:volley

# Only 'category' is taken from the annotated file (spreadsheet programs mangle the time columns).
# Metrics: volley pulses for volleys, all pulses for the other categories.
reviewed_events <- candidate_events %>%
  select(-any_of("category")) %>%
  inner_join(read.csv(file.path(output_dir, "candidate_events_reviewed.csv"), stringsAsFactors = FALSE) %>%
               select(session_event_key, category),
             by = "session_event_key") %>%
  mutate(pulse_set = ifelse(category == "v", "volley", "all")) %>%
  left_join(event_metrics, by = c("session_path", "event_id", "pulse_set")) %>%
  select(-pulse_set, -starts_with("all_"), -starts_with("volley_")) %>%
  mutate(across(c(t_start_first_volley, t_end_last_volley, n_volleys, n_volley_eods),
                ~ if_else(category == "v", .x, NA)))

cat("Annotated categories:\n")
print(table(reviewed_events$category))

reviewed_events <- reviewed_events %>%
  filter(category %in% c("r", "v"))

# Re-classify r events: high amplitude CV + short median IPI -> moving (m), low amplitude CV + long median IPI -> stationary (s).
# Events that satisfy neither combination are labelled u (undetermined) and excluded from the selection.
# Cutoffs default to the median of the r events; replace with fixed values after checking reclassification_scatter.png.
is_reclassified <- reviewed_events$category == "r"
amp_cv_cutoff <- median(reviewed_events$metric_amp_cv_abs[is_reclassified], na.rm = TRUE)
ipi_cutoff <- median(reviewed_events$metric_median_ipi_seconds[is_reclassified], na.rm = TRUE)

reviewed_events <- reviewed_events %>%
  mutate(category = case_when(
    category != "r" ~ category,
    metric_amp_cv_abs >= amp_cv_cutoff & metric_median_ipi_seconds <= ipi_cutoff ~ "m",
    metric_amp_cv_abs < amp_cv_cutoff & metric_median_ipi_seconds > ipi_cutoff ~ "s",
    TRUE ~ "u"
  ))

cat(sprintf("Re-classification cutoffs: amplitude CV %.3f, median IPI %.4f s\n", amp_cv_cutoff, ipi_cutoff))
cat("Categories after re-classification:\n")
print(table(reviewed_events$category))

write.csv(data.frame(amp_cv_cutoff = amp_cv_cutoff, median_ipi_cutoff_s = ipi_cutoff),
          file.path(output_dir, "reclassification_cutoffs.csv"), row.names = FALSE)

p_reclass <- ggplot(filter(reviewed_events, is_reclassified),
                    aes(x = metric_median_ipi_seconds, y = metric_amp_cv_abs, color = category)) +
  geom_point(alpha = 0.7) +
  geom_vline(xintercept = ipi_cutoff, linetype = "dashed") +
  geom_hline(yintercept = amp_cv_cutoff, linetype = "dashed") +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Re-classification of r events", x = "Median IPI (s)", y = "Amplitude CV (absolute)")
ggsave(file.path(output_dir, "reclassification_scatter.png"), p_reclass, width = 7, height = 5)

# For volley events selection_amplitude and n_eods_metric come from the volley pulses only
reviewed_events <- reviewed_events %>%
  mutate(eligible = !is.na(selection_amplitude) & (category != "v" | n_eods_metric >= min_volley_eods_selection))

volley_without_group <- filter(reviewed_events, category == "v", is.na(selection_amplitude))
if (nrow(volley_without_group) > 0) {
  cat(sprintf("Warning: no volley group (>= %d pulses, IPI <= %g s) in %d volley events, excluded from selection, check annotation:\n",
              min_volley_pulses, volley_max_ipi_s, nrow(volley_without_group)))
  print(volley_without_group$session_event_key)
}

curated_events <- reviewed_events %>%
  filter(category %in% selected_categories, eligible) %>%
  group_by(category) %>%
  slice_max(selection_amplitude, n = n_per_category, with_ties = FALSE) %>%
  ungroup()

cat("Selected events per category:\n")
print(table(curated_events$category))
cat("Metric duration (s) per category:\n")
print(tapply(curated_events$metric_duration_seconds, curated_events$category, median, na.rm = TRUE))

# write.csv(curated_events, file.path(output_dir, "curated_events_final.csv"), row.names = FALSE)

# --- Copy all files of the curated events into one subfolder per category (+ the curated events list) ---
curated_dir <- file.path(output_dir, "curated_events")
dir.create(curated_dir, showWarnings = FALSE)
write.csv(curated_events, file.path(curated_dir, "curated_events_final.csv"), row.names = FALSE)

category_folders <- c(m = "movement", s = "stationary", v = "volley")

for (i in seq_len(nrow(curated_events))) {
  session_path_i <- curated_events$session_path[i]
  event_id_i <- curated_events$event_id[i]
  event_dir <- file.path(curated_dir, category_folders[[curated_events$category[i]]])
  dir.create(event_dir, showWarnings = FALSE)
  session_files <- list.files(session_path_i, full.names = FALSE)
  event_suffix <- paste0("_event_", event_id_i)
  png_name <- sprintf("event_%03d_detection.png", as.integer(event_id_i))

  wav_name <- session_files[endsWith(session_files, paste0(event_suffix, ".wav"))]
  if (length(wav_name) == 0) {
    cat(sprintf("  Warning: no .wav file found for event %d in %s\n", event_id_i, session_path_i))
    next
  }
  wav_name <- wav_name[1]
  event_prefix <- sub(paste0(event_suffix, "\\.wav$"), "", wav_name)

  data_names <- c(
    wav_name,
    paste0(event_prefix, event_suffix, "_eod_table.csv"),
    paste0(event_prefix, event_suffix, "_waveforms_concatenated.npz"),
    paste0(event_prefix, event_suffix, "_waveforms_metadata.json")
  )
  for (data_name in data_names) {
    if (!(data_name %in% session_files)) {
      cat(sprintf("  Warning: %s not found in %s\n", data_name, session_path_i))
      next
    }
    file.copy(file.path(session_path_i, data_name), file.path(event_dir, data_name), overwrite = TRUE)
  }

  if (png_name %in% session_files) {
    file.copy(file.path(session_path_i, png_name),
              file.path(event_dir, paste0(event_prefix, "_", png_name)), overwrite = TRUE)
  } else {
    cat(sprintf("  Warning: no detection PNG found for event %d in %s\n", event_id_i, session_path_i))
  }
}
cat(sprintf("Copied files of %d curated events to: %s\n", nrow(curated_events), curated_dir))

# Amplitude (distance proxy) per category, all annotated vs selected
p_sel <- ggplot(filter(reviewed_events, category %in% selected_categories, eligible),
                aes(x = category, y = selection_amplitude)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(color = session_event_key %in% curated_events$session_event_key), width = 0.15, alpha = 0.6) +
  scale_y_log10() +
  labs(title = "Selection amplitude per category", color = "selected")
ggsave(file.path(curated_dir, "selected_amplitude_by_category.png"), p_sel, width = 7, height = 5)

# Amplitude histograms per category with the top-n selection cutoff (lowest selected median amplitude)
amp_hist_data <- reviewed_events %>%
  filter(category %in% selected_categories, eligible) %>%
  mutate(
    category_name = unname(category_folders[category]),
    selected = session_event_key %in% curated_events$session_event_key
  )

amp_cutoffs <- curated_events %>%
  group_by(category) %>%
  summarise(cutoff = min(selection_amplitude), .groups = "drop") %>%
  mutate(category_name = unname(category_folders[category]))

p_amp_hist <- ggplot(amp_hist_data, aes(x = selection_amplitude, fill = selected)) +
  geom_histogram(bins = 30) +
  geom_vline(data = amp_cutoffs, aes(xintercept = cutoff), color = "red", linetype = "dashed") +
  geom_text(data = amp_cutoffs, aes(x = cutoff, y = Inf, label = sprintf("%.4f", cutoff)),
            inherit.aes = FALSE, color = "red", hjust = -0.1, vjust = 1.5, size = 3) +
  facet_wrap(~ category_name, ncol = 1) +
  scale_x_log10() +
  labs(title = sprintf("Selection amplitude per category (top %d selected)", n_per_category),
       x = "Selection amplitude (P90 of volley pulses / all pulses)", y = "Events", fill = "selected")
ggsave(file.path(output_dir, "selected_amplitude_histograms.png"), p_amp_hist, width = 8, height = 9)

# --- Numeric differences between the visually assigned categories ---
# Run for all annotated candidates (saved in output_dir) and for the curated events (saved in curated_dir)
category_labels <- c(m = "moving", s = "stationary", v = "volley", u = "undetermined")
feature_cols <- c("metric_duration_seconds", "n_eods_metric", "metric_strong_fraction", "metric_rate_hz",
                  "metric_median_ipi_seconds", "metric_ipi_iqr_ratio",
                  "metric_median_amplitude", "selection_amplitude", "metric_max_amplitude",
                  "metric_amp_cv_signed", "metric_amp_cv_abs", "metric_amp_range_log10", "metric_flip_rate")
# Excluded from PCA: zeros in the flip rate break log10; rate and strong fraction are collinear with other features
pca_cols <- setdiff(feature_cols, "metric_flip_rate")

analysis_sets <- list(
  annotated = list(events = reviewed_events, dir = output_dir),
  curated = list(events = curated_events, dir = curated_dir)
)

for (set_name in names(analysis_sets)) {
  set_events <- analysis_sets[[set_name]]$events
  set_dir <- analysis_sets[[set_name]]$dir
  cat(sprintf("\n=== Feature analysis of %s events (n = %d), output in %s ===\n",
              set_name, nrow(set_events), set_dir))

  feature_data <- set_events %>%
    filter(category %in% names(category_labels)) %>%
    mutate(category = factor(category, levels = names(category_labels), labels = category_labels))

  category_medians <- feature_data %>%
    group_by(category) %>%
    summarise(n = n(), across(all_of(feature_cols), ~ median(.x, na.rm = TRUE)), .groups = "drop")
  print(as.data.frame(category_medians), digits = 3)
  write.csv(category_medians, file.path(set_dir, "category_feature_medians.csv"), row.names = FALSE)

  feature_long <- feature_data %>%
    select(category, all_of(feature_cols)) %>%
    pivot_longer(-category, names_to = "feature", values_to = "value") %>%
    mutate(feature = factor(feature, levels = feature_cols))

  p_box <- ggplot(feature_long, aes(x = category, y = value, fill = category)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6) +
    geom_jitter(width = 0.15, alpha = 0.4, size = 0.8) +
    facet_wrap(~ feature, scales = "free_y") +
    scale_y_log10() +
    labs(x = NULL, title = sprintf("Event metrics by visually assigned category (%s events)", set_name)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
  ggsave(file.path(set_dir, "category_feature_boxplots.png"), p_box, width = 16, height = 11)

  p_sc_ipi_cv <- ggplot(feature_data, aes(x = metric_median_ipi_seconds, y = metric_amp_cv_abs, color = category)) +
    geom_point(alpha = 0.7) +
    scale_x_log10() +
    scale_y_log10() +
    labs(x = "Median IPI, metric pulses (s)", y = "Amplitude CV (absolute)")

  p_sc_amp_ipi <- ggplot(feature_data, aes(x = metric_amp_cv_signed, y = metric_amp_cv_abs, color = category)) +
    geom_point(alpha = 0.7) +
    scale_x_log10() +
    scale_y_log10() +
    labs(x = "Amplitude CV (signed)", y = "Amplitude CV (absolute)")

  p_sc_dur <- ggplot(feature_data, aes(x = metric_amp_range_log10, y = metric_ipi_iqr_ratio, color = category)) +
    geom_point(alpha = 0.7) +
    scale_y_log10() +
    labs(x = "Amplitude range log10(P90/P10)", y = "IPI IQR / median")

  # PCA on log10 features (z-scored)
  pca_input <- feature_data %>%
    select(all_of(pca_cols)) %>%
    mutate(across(everything(), log10))
  pca_rows <- apply(pca_input, 1, function(x) all(is.finite(x)))
  pca_fit <- prcomp(pca_input[pca_rows, ], scale. = TRUE)
  cat("PCA variance explained (first 4 PCs):\n")
  print(round(summary(pca_fit)$importance[2, 1:4], 3))
  cat("PCA loadings (PC1, PC2):\n")
  print(round(pca_fit$rotation[, 1:2], 2))

  pca_scores <- data.frame(pca_fit$x[, 1:2], category = feature_data$category[pca_rows])
  p_pca <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = category)) +
    geom_point(alpha = 0.7) +
    labs(title = sprintf("PCA of log10 event metrics (%s events)", set_name))

  p_scatter_pca <- ggarrange(p_sc_ipi_cv, p_sc_amp_ipi, p_sc_dur, p_pca, ncol = 2, nrow = 2, common.legend = TRUE)
  ggsave(file.path(set_dir, "category_scatter_pca.png"), p_scatter_pca, width = 14, height = 11)
}

