# Fig 4 (RQ iv): within-event movement trajectories and phase-locking/co-bursting between fish
# pairs. Restricted to genuine 2-fish events (per user decision, to keep computational load
# tractable - larger groups need a fundamentally different, harder N-way coupling metric and
# are far rarer anyway). Per-pulse per-event data lives on an external HDD, not in the main
# tracking summary pipeline bundle (too voluminous to keep locally alongside the per-fish
# summaries used by scripts 3/3b/3c/3d/4/6).
#
# Directory structure on the HDD: <System>_<LoggerID>\<SessionDate>\<extraction_folder>\track\
# (System "JR" is written as "Juma" on disk). Multiple extraction-parameter folders can exist
# per session (versioned reruns, e.g. "01_...", "02_...", "03_...") - always use the LAST one
# (sorted) to avoid double-counting duplicate/superseded event tables. Each session's track/
# folder has a small tracked_event_summary.csv (one row per event, including n_fish_total) -
# used to select 2-fish events WITHOUT opening every single per-pulse _tracked.csv file (which
# would mean reading tens of thousands of files over a slow external drive - LN_L7 alone has
# 3598). Individual per-pulse tables are only opened for the qualifying events.

library(tidyverse)

source("02 Code/R code/00_plot_helpers.R")  # theme_black(), theme_white(), species_palette(), save_panel_pdf()

HDD_ROOT   <- "H:/"
OUTPUT_DIR <- "03 Analysis Output/figures"

# INPUT_MODE: "curated" reads only the small pre-vetted folder of visually-confirmed 2-fish
# encounters (H:/Example Events/) - fast, for building/checking the analysis and the example
# plot without a full external-drive traversal. "full_scan" does the complete
# System_LoggerID\SessionDate\extraction\track discovery across the whole HDD (slow - see
# TEST_MODE_N_LOGGERS below for a partial version of that).
INPUT_MODE <- "curated"
CURATED_EVENTS_ROOT <- "H:/Example Events"

# TEST_MODE: process only the first N logger folders first (full_scan mode only), to validate
# the discovery/parsing logic cheaply before committing to a full external-drive traversal.
# Set to Inf for the real run.
TEST_MODE_N_LOGGERS <- Inf

# phase-locking is only computed using pulse pairs where the two fish were within this many
# electrode units of each other (per user decision) - coupling is only a meaningful social
# behavior signal when the fish are actually close enough to sense one another
MAX_PROXIMITY_ELECTRODE_UNITS <- 1

# quality filters (per user decision, after inspecting a long/messy example event that
# suggested tracking/segmentation error-inflation over long events): only keep short, clean
# 2-fish encounters - max event duration, minimum time actually spent in proximity, and a
# sanity cap on per-fish pulse rate (implausibly high sustained rates usually mean a tracking
# artifact, e.g. two real fish merged into one track).
MAX_EVENT_DURATION_S     <- 60
MIN_PROXIMITY_DURATION_S <- 4  # user's guidance was "5-10s"; 4s used to include a known-good
                               # ~6s example event where proximity spans 4.94s (nearly the
                               # whole event) - requiring a full 5s there would be an
                               # arbitrary edge-case exclusion of an otherwise clean encounter
MAX_PULSE_RATE_HZ        <- 50

SYSTEM_FOLDER_MAP <- c("Juma" = "JR", "PL" = "PL", "SL" = "SL", "LN" = "LN")

# 1. DISCOVER 2-FISH EVENTS ACROSS ALL SESSIONS ------------------------------------------

if (INPUT_MODE == "curated") {
  # files are already visually confirmed 2-fish encounters - no need for the
  # tracked_event_summary.csv n_fish_total filtering step, just glob directly
  tracked_files <- list.files(CURATED_EVENTS_ROOT, pattern = "_event_.*_tracked\\.csv$",
                              recursive = TRUE, full.names = TRUE)
  event_candidates <- tibble(
    System       = NA_character_,
    Logger_ID    = str_extract(basename(tracked_files), "^L\\d+"),
    Session_date = str_extract(basename(tracked_files), "\\d{8}"),
    event_id     = str_extract(basename(tracked_files), "(?<=_event_)\\d+(?=_tracked\\.csv)") |> as.integer(),
    tracked_file = tracked_files
  )
  cat("Curated mode:", nrow(event_candidates), "pre-vetted 2-fish event file(s) found\n")
} else {

logger_folders <- list.dirs(HDD_ROOT, recursive = FALSE)
logger_folders <- logger_folders[grepl("^(Juma|PL_|SL_|LN_)", basename(logger_folders))]
if (is.finite(TEST_MODE_N_LOGGERS)) logger_folders <- head(logger_folders, TEST_MODE_N_LOGGERS)
cat("Scanning", length(logger_folders), "logger folder(s):\n")
print(basename(logger_folders))

event_candidates <- list()

for (logger_folder in logger_folders) {
  logger_name    <- basename(logger_folder)
  system_prefix  <- str_extract(logger_name, "^[A-Za-z]+")
  system_label   <- SYSTEM_FOLDER_MAP[[system_prefix]]
  logger_id      <- str_extract(logger_name, "L\\d+")

  date_folders <- list.dirs(logger_folder, recursive = FALSE)
  for (date_folder in date_folders) {
    session_date_str <- str_extract(basename(date_folder), "^\\d{8}")  # strips suffixes like "_cable_moved"
    extraction_folders <- list.dirs(date_folder, recursive = FALSE)
    if (length(extraction_folders) == 0) next
    extraction_folder <- tail(sort(extraction_folders), 1)  # last (most refined) extraction run
    track_folder  <- file.path(extraction_folder, "track")
    summary_path  <- file.path(track_folder, "tracked_event_summary.csv")
    if (!file.exists(summary_path)) next

    event_summary <- read.csv(summary_path)
    if (!"n_fish_total" %in% names(event_summary)) next
    two_fish_events <- event_summary |> filter(n_fish_total == 2)
    if (nrow(two_fish_events) == 0) next

    # match each qualifying event_id to its actual _tracked.csv filename by parsing the
    # event_id out of the filename directly - safer than reconstructing the name from
    # file_names, which can be ambiguous for events spanning multiple audio files
    tracked_files <- list.files(track_folder, pattern = "_tracked\\.csv$", full.names = TRUE)
    tracked_event_ids <- str_extract(basename(tracked_files), "(?<=_event_)\\d+(?=_tracked\\.csv)") |> as.integer()

    for (i in seq_len(nrow(two_fish_events))) {
      ev_id <- two_fish_events$event_id[i]
      match_idx <- which(tracked_event_ids == ev_id)
      if (length(match_idx) != 1) next
      event_candidates[[length(event_candidates) + 1]] <- tibble(
        System = system_label, Logger_ID = logger_id, Session_date = session_date_str,
        event_id = ev_id, tracked_file = tracked_files[match_idx]
      )
    }
  }
}

event_candidates <- bind_rows(event_candidates)

}  # end INPUT_MODE == "full_scan"

cat("\nFound", nrow(event_candidates), "candidate 2-fish event(s)\n")

# 2. PER-EVENT PHASE-LOCKING + MOVEMENT METRICS ------------------------------------------
# Phase-locking: for each pulse of fish B, compute its circular PHASE within the enclosing
# inter-pulse interval of fish A (2*pi * (t_B - t_A_before) / (t_A_after - t_A_before)) - the
# standard approach for quantifying pulse-timing coupling in pulse-type weakly electric fish
# (does fish B consistently time its pulses relative to fish A's cycle, rather than randomly).
# Computed in BOTH directions (A relative to B and B relative to A) since either fish could be
# the "driver". Circular mean resultant length R (0 = uniform/no locking, 1 = perfect locking)
# and a Rayleigh test p-value (Zar 1999 approximation, avoids adding a `circular` package
# dependency for one test) assess whether the phase distribution is non-uniform.
rayleigh_test <- function(phases) {
  n <- length(phases)
  if (n < 3) return(list(R = NA_real_, p = NA_real_, n = n))
  R_bar <- Mod(sum(exp(1i * phases))) / n
  Z <- n * R_bar^2
  p <- exp(-Z) * (1 + (2 * Z - Z^2) / (4 * n) - (24 * Z - 132 * Z^2 + 76 * Z^3 - 9 * Z^4) / (288 * n^2))
  list(R = R_bar, p = min(max(p, 0), 1), n = n)
}

phase_of_b_relative_to_a <- function(t_a, loc_a, t_b, loc_b, max_distance = MAX_PROXIMITY_ELECTRODE_UNITS) {
  # phase of each t_b pulse within the enclosing [t_a[i], t_a[i+1]) interval; NA if t_b falls
  # before the first or after the last t_a pulse (can't be bracketed). Also linearly
  # interpolates fish A's location at each t_b (between the bracketing pair) and DROPS any
  # phase point where the two fish were more than max_distance electrode units apart at that
  # moment - phase-locking/coupling is only meaningful as a social behavior when the fish are
  # actually close enough to sense each other, not for far-apart pulses that happen to fall at
  # some phase by chance.
  idx <- findInterval(t_b, t_a)
  valid <- idx >= 1 & idx < length(t_a)
  phase <- rep(NA_real_, length(t_b))
  interp_loc_a <- rep(NA_real_, length(t_b))
  frac <- (t_b[valid] - t_a[idx[valid]]) / (t_a[idx[valid] + 1] - t_a[idx[valid]])
  phase[valid] <- 2 * pi * frac
  interp_loc_a[valid] <- loc_a[idx[valid]] + frac * (loc_a[idx[valid] + 1] - loc_a[idx[valid]])
  close_enough <- valid & !is.na(interp_loc_a) & !is.na(loc_b) &
    (abs(loc_b - interp_loc_a) <= max_distance)
  phase[close_enough]
}

proximity_duration <- function(t_a, loc_a, t_b, loc_b, max_distance = MAX_PROXIMITY_ELECTRODE_UNITS) {
  # total wall-clock time the two fish spent within max_distance of each other - each fish's
  # pulses are checked against the OTHER fish's interpolated location at that same moment;
  # qualifying timestamps within 2s of each other are merged into one continuous episode
  # (separate approaches are summed, not just first-to-last which could span a long gap)
  interp_at <- function(t_query, t_ref, loc_ref) {
    idx <- findInterval(t_query, t_ref)
    valid <- idx >= 1 & idx < length(t_ref)
    frac <- rep(NA_real_, length(t_query))
    interp <- rep(NA_real_, length(t_query))
    frac[valid] <- (t_query[valid] - t_ref[idx[valid]]) / (t_ref[idx[valid] + 1] - t_ref[idx[valid]])
    interp[valid] <- loc_ref[idx[valid]] + frac[valid] * (loc_ref[idx[valid] + 1] - loc_ref[idx[valid]])
    interp
  }
  loc_a_at_b <- interp_at(t_b, t_a, loc_a)
  loc_b_at_a <- interp_at(t_a, t_b, loc_b)
  close_b <- !is.na(loc_a_at_b) & abs(loc_b - loc_a_at_b) <= max_distance
  close_a <- !is.na(loc_b_at_a) & abs(loc_a - loc_b_at_a) <= max_distance
  close_times <- sort(c(t_a[close_a], t_b[close_b]))
  if (length(close_times) < 2) return(0)
  gaps <- diff(close_times)
  episode_id <- cumsum(c(TRUE, gaps > 2))
  sum(tapply(close_times, episode_id, function(x) diff(range(x))))
}

event_results    <- list()
event_pulse_data <- list()  # kept only for events actually processed, for the movement plot

for (i in seq_len(nrow(event_candidates))) {
  row    <- event_candidates[i, ]
  pulses <- tryCatch(read.csv(row$tracked_file), error = function(e) NULL)
  if (is.null(pulses) || !all(c("fish_id", "timestamp_dt", "pulse_location") %in% names(pulses))) next

  # take the two MOST-ACTIVE tracked fish_id values (fish_id == -1 is untracked/noise, dropped)
  # rather than requiring exactly 2 distinct fish_id total - some "2-fish" events also contain
  # a minor 3rd track (a handful of stray/fragment pulses), which shouldn't disqualify the
  # dominant two-fish interaction from being analyzed
  fish_id_counts <- pulses |> filter(fish_id != -1) |> count(fish_id, sort = TRUE)
  if (nrow(fish_id_counts) < 2) next
  fish_ids <- fish_id_counts$fish_id[1:2]

  pulses$t_num <- as.numeric(as.POSIXct(pulses$timestamp_dt, tz = "UTC"))
  fish_a <- pulses |> filter(fish_id == fish_ids[1]) |> arrange(t_num)
  fish_b <- pulses |> filter(fish_id == fish_ids[2]) |> arrange(t_num)
  if (nrow(fish_a) < 3 || nrow(fish_b) < 3) next

  duration_s <- max(pulses$t_num) - min(pulses$t_num)
  if (duration_s > MAX_EVENT_DURATION_S) next  # long events showed tracking/segmentation
                                                # error-inflation on visual inspection
  if (nrow(fish_a) / duration_s > MAX_PULSE_RATE_HZ || nrow(fish_b) / duration_s > MAX_PULSE_RATE_HZ) next

  prox_duration_s <- proximity_duration(fish_a$t_num, fish_a$pulse_location, fish_b$t_num, fish_b$pulse_location)
  if (prox_duration_s < MIN_PROXIMITY_DURATION_S) next

  phase_b_in_a <- phase_of_b_relative_to_a(fish_a$t_num, fish_a$pulse_location, fish_b$t_num, fish_b$pulse_location)
  phase_a_in_b <- phase_of_b_relative_to_a(fish_b$t_num, fish_b$pulse_location, fish_a$t_num, fish_a$pulse_location)
  rt_b_in_a <- rayleigh_test(phase_b_in_a)
  rt_a_in_b <- rayleigh_test(phase_a_in_b)

  event_results[[length(event_results) + 1]] <- tibble(
    System = row$System, Logger_ID = row$Logger_ID, Session_date = row$Session_date,
    event_id = row$event_id,
    species_a = fish_a$species_assigned[1], species_b = fish_b$species_assigned[1],
    n_pulses_a = nrow(fish_a), n_pulses_b = nrow(fish_b),
    duration_s = duration_s, prox_duration_s = prox_duration_s,
    R_b_in_a = rt_b_in_a$R, p_b_in_a = rt_b_in_a$p,
    R_a_in_b = rt_a_in_b$R, p_a_in_b = rt_a_in_b$p,
    range_location_a = diff(range(fish_a$pulse_location)),
    range_location_b = diff(range(fish_b$pulse_location)),
    sd_location_a = sd(fish_a$pulse_location), sd_location_b = sd(fish_b$pulse_location)
  )

  event_key <- paste(row$System, row$Logger_ID, row$Session_date, row$event_id, sep = "_")
  event_pulse_data[[event_key]] <- pulses |> select(fish_id, t_num, pulse_location, species_assigned)
}

event_results <- bind_rows(event_results)
cat("Processed", nrow(event_results), "two-fish event(s) with usable phase-locking data\n")

saveRDS(list(event_results = event_results, event_pulse_data = event_pulse_data),
        file.path(OUTPUT_DIR, "fig4_within_event_data.rds"))
write.csv(event_results, file.path(OUTPUT_DIR, "fig4_two_fish_event_results.csv"), row.names = FALSE)

cat("\n=== Phase-locking (R_b_in_a) summary ===\n")
print(summary(event_results$R_b_in_a))
cat("\n=== Fraction of events with significant phase-locking (p < 0.05, either direction) ===\n")
print(mean(event_results$p_b_in_a < 0.05 | event_results$p_a_in_b < 0.05, na.rm = TRUE))

# 3. EXAMPLE EVENT PLOT (Fig 4 panel) ----------------------------------------------------
# Pick one representative 2-fish event where the fish visibly move toward each other AND show
# significant phase-locking, to illustrate both movement and phase coupling as time series.
# Distance/phase here are NOT proximity-filtered (unlike the statistical test above) - the
# whole point of this panel is to show the approach itself, which necessarily starts out far
# apart before crossing the proximity threshold.
# PREFERRED EXAMPLE (per user's visual review): L3-20231116T182859_event_1190 shows the fish
# "pulsing and pausing together" - a genuine phase-locking illustration, checked directly
# against event_results (not the pulse-count-filtered example_candidates below, since the
# preferred pick should win regardless of that threshold) and used over the automatic
# convergence-based selection if present.
preferred_match <- event_results |>
  mutate(event_key = paste(System, Logger_ID, Session_date, event_id, sep = "_")) |>
  filter(Logger_ID == "L3", Session_date == "20231116", event_id == 1190)

example_candidates <- event_results |>
  filter(!is.na(p_b_in_a), n_pulses_a >= 8, n_pulses_b >= 8) |>
  mutate(event_key = paste(System, Logger_ID, Session_date, event_id, sep = "_"))

# for each candidate, a simple linear trend of interpolated fish-to-fish distance vs time -
# the "moving toward each other" signature is a negative slope
convergence_scores <- map_dbl(example_candidates$event_key, function(key) {
  pulses <- event_pulse_data[[key]]
  fish_id_counts <- pulses |> filter(fish_id != -1) |> count(fish_id, sort = TRUE)
  if (nrow(fish_id_counts) < 2) return(NA_real_)
  fish_ids <- fish_id_counts$fish_id[1:2]
  fish_a <- pulses |> filter(fish_id == fish_ids[1]) |> arrange(t_num)
  fish_b <- pulses |> filter(fish_id == fish_ids[2]) |> arrange(t_num)
  idx <- findInterval(fish_b$t_num, fish_a$t_num)
  valid <- idx >= 1 & idx < nrow(fish_a)
  if (sum(valid) < 5) return(NA_real_)
  frac <- (fish_b$t_num[valid] - fish_a$t_num[idx[valid]]) / (fish_a$t_num[idx[valid] + 1] - fish_a$t_num[idx[valid]])
  interp_loc_a <- fish_a$pulse_location[idx[valid]] + frac * (fish_a$pulse_location[idx[valid] + 1] - fish_a$pulse_location[idx[valid]])
  distance <- abs(fish_b$pulse_location[valid] - interp_loc_a)
  unname(coef(lm(distance ~ fish_b$t_num[valid]))[2])
})
example_candidates$convergence_score <- convergence_scores

# cascading fallback - the strictest selection (visible convergence AND significant locking)
# may find nothing in a small/test sample; relax step by step rather than erroring out
select_example_event <- function(candidates) {
  strict <- candidates |> filter(p_b_in_a < 0.05, !is.na(convergence_score)) |> arrange(convergence_score)
  if (nrow(strict) > 0) return(strict$event_key[1])
  relaxed <- candidates |> filter(!is.na(convergence_score)) |> arrange(convergence_score)
  if (nrow(relaxed) > 0) return(relaxed$event_key[1])
  any_valid <- candidates |> arrange(desc(n_pulses_a + n_pulses_b))
  if (nrow(any_valid) > 0) return(any_valid$event_key[1])
  NA_character_
}

# PREFERRED EXAMPLE (per user's visual review): L3-20231116T182859_event_1190 shows the fish
# "pulsing and pausing together" - a genuine phase-locking illustration, chosen directly over
# the automatic convergence-based selection below (kept as a fallback if this one is filtered
# out for any reason, e.g. re-running against a different curated subset).
if (nrow(preferred_match) > 0) {
  example_event_key <- preferred_match$event_key[1]
} else {
  example_event_key <- select_example_event(example_candidates)
}
cat("\nExample event chosen for Fig 4 panel:", example_event_key, "\n")
if (is.na(example_event_key)) stop("No candidate events available for the example plot - re-run with more logger folders (TEST_MODE_N_LOGGERS).")

example_pulses <- event_pulse_data[[example_event_key]]
fish_id_counts <- example_pulses |> filter(fish_id != -1) |> count(fish_id, sort = TRUE)
fish_ids <- fish_id_counts$fish_id[1:2]
fish_a <- example_pulses |> filter(fish_id == fish_ids[1]) |> arrange(t_num)
fish_b <- example_pulses |> filter(fish_id == fish_ids[2]) |> arrange(t_num)
t0 <- min(example_pulses$t_num)

idx   <- findInterval(fish_b$t_num, fish_a$t_num)
valid <- idx >= 1 & idx < nrow(fish_a)
frac  <- (fish_b$t_num[valid] - fish_a$t_num[idx[valid]]) / (fish_a$t_num[idx[valid] + 1] - fish_a$t_num[idx[valid]])
example_series <- tibble(
  t_s = fish_b$t_num[valid] - t0,
  distance = abs(fish_b$pulse_location[valid] -
                   (fish_a$pulse_location[idx[valid]] + frac * (fish_a$pulse_location[idx[valid] + 1] - fish_a$pulse_location[idx[valid]]))),
  phase = 2 * pi * frac
)
example_trajectories <- bind_rows(
  fish_a |> mutate(t_s = t_num - t0, fish = "Fish A"),
  fish_b |> mutate(t_s = t_num - t0, fish = "Fish B")
)

build_panel_example_event <- function(theme_fn) {
  p_traj <- ggplot(example_trajectories, aes(x = t_s, y = pulse_location, color = fish)) +
    geom_point(size = 1) +
    scale_color_manual(values = c("Fish A" = cbf_palette[2], "Fish B" = cbf_palette[5])) +
    labs(y = "Location (electrode units)") +
    theme_fn(base_size = 14) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank())

  p_dist <- ggplot(example_series, aes(x = t_s, y = distance)) +
    geom_point(size = 1, color = cbf_palette[3]) +
    geom_hline(yintercept = MAX_PROXIMITY_ELECTRODE_UNITS, linetype = "dashed", alpha = 0.5) +
    labs(y = "Distance between fish\n(electrode units)") +
    theme_fn(base_size = 14) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank())

  p_phase <- ggplot(example_series, aes(x = t_s, y = phase)) +
    geom_point(size = 1, color = cbf_palette[3]) +
    scale_y_continuous(limits = c(0, 2 * pi), breaks = c(0, pi, 2 * pi),
                        labels = c("0", "\u03c0", "2\u03c0")) +
    labs(x = "Time since event start (s)", y = "Phase of B\nin A's cycle") +
    theme_fn(base_size = 14)

  ggpubr::ggarrange(p_traj, p_dist, p_phase, ncol = 1, align = "v", heights = c(1, 1, 1.15))
}

fig4_panel_example_event_dark  <- build_panel_example_event(theme_black)
fig4_panel_example_event_white <- build_panel_example_event(theme_white)

ggpubr::ggexport(fig4_panel_example_event_dark,
                  filename = file.path(OUTPUT_DIR, "fig4_panel_example_event_dark.pdf"),
                  width = 12, height = 9)
ggpubr::ggexport(fig4_panel_example_event_white,
                  filename = file.path(OUTPUT_DIR, "fig4_panel_example_event_white.pdf"),
                  width = 12, height = 9)
png_dir <- file.path(OUTPUT_DIR, "png")
if (!dir.exists(png_dir)) dir.create(png_dir, recursive = TRUE)
ggsave(file.path(png_dir, "fig4_panel_example_event_dark.png"), fig4_panel_example_event_dark,
       width = 12, height = 9, dpi = 300)
ggsave(file.path(png_dir, "fig4_panel_example_event_white.png"), fig4_panel_example_event_white,
       width = 12, height = 9, dpi = 300)

saveRDS(list(panel_example_event_dark = fig4_panel_example_event_dark,
             panel_example_event_white = fig4_panel_example_event_white),
        file.path(OUTPUT_DIR, "fig4_panels.rds"))
