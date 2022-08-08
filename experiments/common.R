library(ggplot2)
library(dplyr)
library(purrr)
library(readr)
library(gdata)
library(stringr)
library(tidyr)
library(scales)
library(showtext)

font_add("Computer Modern", "cmunrm.ttf")
# Automatically use showtext to render text for future devices.
showtext_auto()

# Load Computer Modern font
font_add("Computer Modern", "cmunrm.ttf")

# Prepare logarithmic grid lines
log_axis_breaks <- 10^(-10:10)
log_axis_minor_breaks <- rep(1:9, 21)*(10^rep(-10:10, each=9))

# Define some shortcuts
geom_line_with_points_and_errorbars <- function(data = NULL, mapping = NULL, point_size = 0.25, errorbar_width = 0.25) {
  list(
    geom_point(data = data, mapping = mapping, size = point_size),
    geom_path(data = data, mapping = mapping),
    geom_errorbar(data = data, mapping = mapping, width = errorbar_width)
  )
}

scale_color_dark2 <- function() {
  scale_color_brewer(type = "qual", palette = "Dark2")
}

scale_y_log_with_ticks_and_lines <- function(scale_accuracy = NULL, limits = NULL) {
  list(
    scale_y_log10(
      breaks = log_axis_breaks,
      minor_breaks = log_axis_minor_breaks,
      labels = comma(log_axis_breaks, accuracy = scale_accuracy),
      limits = limits 
    ),
    annotation_logticks(base = 10, sides = "l")
  )
}

theme_husky <- function(...) {
  return(
    theme_bw() +
    theme(
      # Remove unneeded grid elements.
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
   
      # Remove the background behind the legend.
      legend.background = element_blank(),
      legend.key = element_blank(),
     
      # Reduce the font size to 8 pt and adjust the line spacing in the legend accordingly. 
      legend.key.size = unit(3, "mm"),
      text = element_text(
        family = "Computer Modern",
        #size = 8
      ),
      
      # Pass on further parameters.
      ...
    )
  )
}

# Convert ns times to ms times
ns2ms <- function(x) {
  return(x / 10^6)
}
