# Shared plotting conventions (species palette, colorblind-safe palette, black theme) sourced by
# script 3 and all downstream RQ scripts (3b, 3c, 3d, 4, 5, 6, 7) - single source of truth.

library(ggpubr)  # for save_panel_pdf()'s ggexport() call

SPECIES_PALETTE <- c(
  "GL" = "#CC79A7",
  "PN" = "#56B4E9",
  "PD" = "#009E73",
  "MV" = "#E69F00",
  "Unknown" = "white"
)
# Same species colors, but "Unknown" recolored for legibility on a white background
# (used by the white/print theme's panels; see species_palette() below).
SPECIES_PALETTE_WHITE <- replace(SPECIES_PALETTE, "Unknown", "grey40")

SPECIES_CODES <- names(SPECIES_PALETTE)  # includes "Unknown" as a genuine 5th level

# Returns the palette matching a given theme function (theme_black -> white "Unknown",
# theme_white -> grey40 "Unknown"), so the same panel-building code works for both.
species_palette <- function(theme_fn) {
  if (identical(theme_fn, theme_white)) SPECIES_PALETTE_WHITE else SPECIES_PALETTE
}

# Colorblind-friendly palette (Wong 2011), reordered for contrast on black
# Order: yellow, sky blue, orange, bluish green, vermillion, blue, reddish purple
cbf_palette <- c(
  "#F0E442", "#56B4E9", "#E69F00", "#009E73",
  "#D55E00", "#0072B2", "#CC79A7"
)

theme_black <- function(base_size = 12, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # backgrounds
      plot.background  = element_rect(fill = "black", color = NA),
      panel.background = element_rect(fill = "black", color = NA),
      legend.background = element_rect(fill = "black", color = NA),
      legend.key        = element_rect(fill = "black", color = NA),
      strip.background  = element_rect(fill = "black", color = "white", linewidth = 0.5),

      # axes
      axis.line  = element_line(color = "white", linewidth = 0.5),
      axis.ticks = element_line(color = "white", linewidth = 0.4),

      # text
      plot.title    = element_text(color = "white", face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "white", hjust = 0),
      plot.caption  = element_text(color = "grey70", hjust = 1),
      axis.title    = element_text(color = "white"),
      axis.text     = element_text(color = "white"),
      legend.title  = element_text(color = "white"),
      legend.text   = element_text(color = "white"),
      strip.text    = element_text(color = "white", face = "bold"),

      # grid (off by default as in theme_classic, but white if enabled)
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# Clean white/print theme for the manuscript figures (final layout done in Illustrator) -
# same colorblind-safe SPECIES_PALETTE/cbf_palette work on both this and theme_black().
theme_white <- function(base_size = 12, base_family = "") {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      plot.background   = element_rect(fill = "white", color = NA),
      panel.background  = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key        = element_rect(fill = "white", color = NA),
      strip.background  = element_rect(fill = "white", color = "black", linewidth = 0.3),

      axis.line  = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.3),

      plot.title    = element_text(color = "black", face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "black", hjust = 0),
      plot.caption  = element_text(color = "grey30", hjust = 1),
      axis.title    = element_text(color = "black"),
      axis.text     = element_text(color = "black"),
      legend.title  = element_text(color = "black"),
      legend.text   = element_text(color = "black"),
      strip.text    = element_text(color = "black", face = "bold"),

      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# Save a panel as a vector PDF, ready for Illustrator - strips the title (captions are added at
# layout time) and keeps the plot itself uncluttered. Uses ggpubr::ggexport() (already a project
# dependency via script 3) for consistency with the rest of the codebase. Doubled the default
# width/height (was 6x4.5) since labels were overlapping at the smaller size. Also exports a PNG
# copy (same size, 300 dpi) into a "png" subfolder next to the PDF, for quick use in slides.
save_panel_pdf <- function(plot, filename, width = 12, height = 7, dpi = 300) {
  plot <- plot + labs(title = NULL, subtitle = NULL, caption = NULL)
  ggpubr::ggexport(plot, filename = filename, width = width, height = height)

  png_dir <- file.path(dirname(filename), "png")
  if (!dir.exists(png_dir)) dir.create(png_dir, recursive = TRUE)
  png_filename <- file.path(png_dir, sub("\\.pdf$", ".png", basename(filename)))
  ggsave(png_filename, plot = plot, width = width, height = height, dpi = dpi)
}

# Same as save_panel_pdf() but does NOT strip title/subtitle/caption - for the rare panel where
# an annotation (e.g. a baked-in "n = " sample-size label) is actual data, not a removable
# figure caption meant to be added later in Illustrator.
save_panel_pdf_with_caption <- function(plot, filename, width = 12, height = 9, dpi = 300) {
  ggpubr::ggexport(plot, filename = filename, width = width, height = height)

  png_dir <- file.path(dirname(filename), "png")
  if (!dir.exists(png_dir)) dir.create(png_dir, recursive = TRUE)
  png_filename <- file.path(png_dir, sub("\\.pdf$", ".png", basename(filename)))
  ggsave(png_filename, plot = plot, width = width, height = height, dpi = dpi)
}
