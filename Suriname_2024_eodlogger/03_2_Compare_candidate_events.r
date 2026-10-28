# Compare candidate_events_for_review.csv of the old and the new extraction to find the events that need a new visual review.
# Events are matched by session (trip / location / recorder) and overlap in time, not by event_id (ids change between runs).
#   new_event:       no overlapping old candidate                 -> review
#   merged_or_split: overlaps several old events (or vice versa)  -> review
#   changed:         1:1 match, but duration, n_eods or n_strong_eods changed by more than tolerance -> review
#   unchanged:       1:1 match within tolerance                   -> category carried over from the old review
#   invalid_event_times: event_end_time before event_start_time (extraction bug for some events at a file end) -> review, cannot be matched

library(tidyverse)
library(lubridate)

old_file <- "G:/First Trip/Raleighvallen/eel_event_subsetting_win_3000/candidate_events_for_review.csv"            # candidate_events_for_review.csv of the old extraction
old_reviewed_file <- "G:/First Trip/Raleighvallen/eel_event_subsetting_win_3000/candidate_events_reviewed.csv"   # annotated candidate_events_reviewed.csv of the old extraction (category column)
new_file <- "G:/First Trip/Raleighvallen/eel_event_subsetting/candidate_events_for_review.csv"            # candidate_events_for_review.csv of the new extraction
output_file <- "G:/First Trip/Raleighvallen/extraction_comparison.csv"         # csv with the review status per new candidate event
tolerance <- 0.1          # relative change that counts as "changed"

old <- read.csv(old_file, stringsAsFactors = FALSE) %>%
  select(-category) %>%
  left_join(read.csv(old_reviewed_file, stringsAsFactors = FALSE) %>% select(session_event_key, category),
            by = "session_event_key") %>%
  mutate(event_start_time = ymd_hms(event_start_time, quiet = TRUE, truncated = 3),
         event_end_time = ymd_hms(event_end_time, quiet = TRUE, truncated = 3),
         session = paste(trip, location, eod_recorder, sep = "|"))

new <- read.csv(new_file, stringsAsFactors = FALSE) %>%
  mutate(event_start_time = ymd_hms(event_start_time, quiet = TRUE, truncated = 3),
         event_end_time = ymd_hms(event_end_time, quiet = TRUE, truncated = 3),
         session = paste(trip, location, eod_recorder, sep = "|"))

cat(sprintf("Candidates: old %d, new %d\n", nrow(old), nrow(new)))

overlaps <- new %>%
  select(session, new_key = session_event_key, new_start = event_start_time, new_end = event_end_time) %>%
  inner_join(old %>% select(session, old_key = session_event_key, old_start = event_start_time, old_end = event_end_time),
             by = "session", relationship = "many-to-many") %>%
  filter(new_start <= old_end, new_end >= old_start) %>%
  select(new_key, old_key)

new_matches <- overlaps %>%
  group_by(new_key) %>%
  summarise(n_old = n(), old_key = first(old_key), .groups = "drop")
old_matches <- overlaps %>%
  group_by(old_key) %>%
  summarise(n_new = n(), .groups = "drop")

new_status <- new %>%
  left_join(new_matches, by = c("session_event_key" = "new_key")) %>%
  left_join(old_matches, by = "old_key") %>%
  left_join(old %>% select(old_key = session_event_key, old_event_id = event_id, old_n_eods = n_eods,
                           old_n_strong = n_strong_eods, old_duration = duration_seconds, old_category = category),
            by = "old_key") %>%
  mutate(
    n_old = coalesce(n_old, 0L),
    n_new = coalesce(n_new, 0L),
    duration_change = duration_seconds / old_duration - 1,
    n_eods_change = n_eods / old_n_eods - 1,
    n_strong_change = n_strong_eods / old_n_strong - 1,
    status = case_when(
      event_end_time < event_start_time ~ "invalid_event_times",
      n_old == 0 ~ "new_event",
      n_old > 1 | n_new > 1 ~ "merged_or_split",
      abs(duration_change) > tolerance | abs(n_eods_change) > tolerance | abs(n_strong_change) > tolerance ~ "changed",
      TRUE ~ "unchanged"
    ),
    category = ifelse(status == "unchanged", coalesce(old_category, ""), "")
  )

cat("Status of the new candidate events:\n")
print(table(new_status$status))
cat(sprintf("Events to review: %d / %d\n", sum(new_status$status != "unchanged"), nrow(new_status)))

dropped_old <- old %>%
  anti_join(overlaps, by = c("session_event_key" = "old_key"))
cat(sprintf("Old candidates without a new counterpart: %d\n", nrow(dropped_old)))
print(table(dropped_old$category))

new_status %>%
  select(session_event_key, trip, location, eod_recorder, event_id, event_start_time, event_end_time,
         status, category, old_event_id, n_old, n_new, duration_change, n_eods_change, n_strong_change) %>%
  arrange(status, event_start_time) %>%
  write.csv(output_file, row.names = FALSE)

