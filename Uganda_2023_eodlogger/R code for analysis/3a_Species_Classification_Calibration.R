# Calibrate species classification thresholds (lda_max_dist_factor / lda_min_probability)
# using the baseline reclassification (factor=1.0, min_probability=0.0, from
# 04_1c_Batch_Reclassify_Species.py). At that baseline, lda_dist_threshold already equals
# the raw (unscaled) max-control-scatter distance for each fish's assigned species, so any
# candidate factor can be tested here purely by rescaling it - no Python re-run needed:
#   uncertain_new = (lda_proba_assigned < min_prob) | (lda_dist_centroid > lda_dist_threshold * factor)

library(tidyverse)
library(jsonlite)

BASELINE_DIR <- "01 Data/EOD Loggers/Tracking Results Reclassified/baseline/"
OUTPUT_DIR   <- "03 Analysis Output/figures"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Load baseline per-fish classification results ---------------------------
fish_list <- list.files(BASELINE_DIR, pattern = "tracked_fish_summary\\.csv",
                        recursive = TRUE, full.names = TRUE)

fish_cal <- NULL
for (ft_file in fish_list) {
  folder_parts <- strsplit(basename(dirname(ft_file)), "_")[[1]]
  ft_dat <- read.csv2(ft_file, sep = ",", dec = ".", stringsAsFactors = FALSE)
  if (!"lda_dist_centroid" %in% names(ft_dat)) next  # 1-NN fallback sessions (<2 control species) - no threshold to sweep
  ft_dat$Logger_ID     <- folder_parts[1]
  ft_dat$Session_date  <- as.Date(folder_parts[2], format = "%Y%m%d")
  ft_dat$Site          <- folder_parts[3]
  ft_dat$System        <- basename(dirname(dirname(ft_file)))
  fish_cal <- bind_rows(fish_cal, ft_dat)
}
fish_cal$species_code <- str_remove(fish_cal$species_assigned, "\\?")

cat("Loaded", nrow(fish_cal), "fish from", length(fish_list), "session file(s)\n")
print(table(fish_cal$species_code))

# ---- Reference LOO accuracy (factor/min_prob-independent, same control library) -------
report_files <- list.files(BASELINE_DIR, pattern = "classifier_report\\.json",
                           recursive = TRUE, full.names = TRUE)
loo_ref <- fromJSON(report_files[1])$loo_per_species |>
  enframe(name = "species_code") |>
  unnest_wider(value)
print(loo_ref)

# ---- Threshold sweep ----------------------------------------------------------
factor_grid <- c(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
prob_grid   <- c(0, 0.5, 0.6, 0.7, 0.8, 0.9)
sweep_grid  <- expand_grid(factor = factor_grid, min_prob = prob_grid)

sweep_by_species <- sweep_grid |>
  pmap_dfr(function(factor, min_prob) {
    fish_cal |>
      mutate(uncertain_new = (lda_proba_assigned < min_prob) |
                              (lda_dist_centroid > lda_dist_threshold * factor)) |>
      group_by(species_code) |>
      summarise(n_total = n(), n_confident = sum(!uncertain_new), .groups = "drop") |>
      mutate(pct_confident = 100 * n_confident / n_total, factor = factor, min_prob = min_prob)
  })

sweep_overall <- sweep_grid |>
  pmap_dfr(function(factor, min_prob) {
    fish_cal |>
      mutate(uncertain_new = (lda_proba_assigned < min_prob) |
                              (lda_dist_centroid > lda_dist_threshold * factor)) |>
      summarise(n_total = n(), n_confident = sum(!uncertain_new)) |>
      mutate(pct_confident = 100 * n_confident / n_total,
             species_code = "ALL", factor = factor, min_prob = min_prob)
  })

sweep_all <- bind_rows(sweep_by_species, sweep_overall)
write.csv(sweep_all |> arrange(species_code, factor, min_prob),
          file.path(OUTPUT_DIR, "species_classification_threshold_sweep.csv"), row.names = FALSE)

# ---- Heatmap: % confident by factor x min_prob, faceted by species -----------
p_heatmap <- ggplot(sweep_all, aes(x = factor(factor), y = factor(min_prob), fill = pct_confident)) +
  geom_tile() +
  geom_text(aes(label = round(pct_confident)), size = 3) +
  scale_fill_viridis_c(name = "% confident", limits = c(0, 100)) +
  facet_wrap(~ species_code) +
  labs(x = "lda_max_dist_factor", y = "lda_min_probability",
       title = "Confident fraction by species across threshold grid") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "species_classification_threshold_sweep.png"),
       p_heatmap, width = 9, height = 7, dpi = 150)
