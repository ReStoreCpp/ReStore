library(ggplot2)
library(dplyr)
library(purrr)
library(readr)
library(gdata)
library(stringr)
library(tidyr)
library(scales)
library(showtext)
library(latex2exp)
library(tools)
library(purrr)

# TODO: Develop this collection of helper functions into a reusable library

husky_shapes <- c(19, 15, 17, 18, 3, 4, 7, 8, 10, 11, 12, 13, 14)

# A neutral element for the `+` operator of the `ggplot` object.
# Useful for example as the last element, so that you can comment out the previous elements
# without having to worry about the trailing `+`.
# Can also be used as an alternitive in an in-line `if` statement.
gg_eps <- function() { list() }

# Remove an element of a named vector by name
remove_elem <- function(vector, name) {
  vector[!names(vector) %in% name]
}

style_to_filetype <- function(style) {
  if (style == "slides") {
    "png"
  } else {
    "pdf"
  }
}

# Load Computer Modern font
font_add("Computer Modern", "cmunrm.ttf")

# Automatically use showtext to render text for future devices.
showtext_auto()

# Prepare logarithmic grid lines
log_axis_breaks <- 10^(-10:10)
log_axis_minor_breaks <- rep(1:9, 21)*(10^rep(-10:10, each=9))

# Define some shortcuts
geom_line_with_points_and_errorbars <- function(data = NULL, mapping = NULL, point_size = 0.25, errorbar_width = 0.25, line_width = 1) {
  list(
    geom_point(data = data, mapping = mapping, size = point_size),
    geom_path(data = data, mapping = mapping, size = line_width),
    geom_errorbar(data = data, mapping = mapping, width = errorbar_width)
  )
}

scale_color_dark2 <- function(...) {
  scale_color_brewer(type = "qual", palette = "Dark2", ...)
}

# Generate the named vector for the `values` vector of a scale_shale_* call
# from a named vector of e.g. colors.    
husky_shape_values <- function(color_values) {
    setNames(
        husky_shapes[1:length(color_values)],
        names(color_values)
    )
}

scale_color_shape_discrete <- function(
    color_scale = scale_color_dark2,
    shape_scale = scale_shape_discrete,
    name = "default",
    ...) {
    list(
        color_scale(guide = "legend", name = name, ...),
        shape_scale(guide = "legend", name = name, ...)
    )
}

scale_color_shape_manual <- function(
    color_values,
    shape_values = waiver(),
    name = "default",
    ...) {
    if (missing(shape_values)) {
        shape_values <- husky_shape_values(color_values)
    }
    scale_color_shape_discrete(
        color_scale = partial(
            scale_color_manual,
            values = color_values,
            ...
        ),
        shape_scale = partial(
            scale_shape_manual,
            values = shape_values,
            ...
        ),
        name = name
    )
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

theme_husky <- function(style = "print", ..., text_size = waiver()) {
  text_size <- if (missing(text_size)) {
    if (style == "print") {
      8
    } else if (style == "slides") {
      22
    } else {
      stop("Unknown style")
    }
  } else {
    text_size
  } 
  print(text_size)
 
  husky <- 
    theme_bw() +
    theme(
      # Remove unneeded grid elements.
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
   
      # Remove the background behind the legend.
      legend.background = element_blank(),
      legend.key = element_blank(),

      # Specify font
      text = element_text(
        family = "Computer Modern",
        size = text_size
      ),


      # Pass on further parameters.
      ...
    )

  if (style == "print")
    husky <- husky + theme (
    # Adjust the line spacing in the legend to the smaller text size. 
    legend.key.size = unit(3, "mm"),
  )

  return(husky)
}

# Create a labellor for log-axis
label_log <- function(base, tex = TRUE) {
    function(n) {
        # We do not get exact values
        exp <- log(round(n), base = base)
        if (tex) {
            lapply(sprintf('$%d^{%d}$', base, exp), TeX)
        } else {
            sprintf('%d^%d', base, exp)
        }
    }
}
label_log2 <- function(tex = TRUE) {
    label_log(base = 2, tex)
}

# Convert ns times to ms times
ns2ms <- function(x) {
  return(x / 10^6)
}

# Convert a vector image to a jpeg file for inclusion in IPE slides.
convert_img <- function(filename, filetype) {
    output_filename <- sprintf("%s.%s", tools::file_path_sans_ext(filename), filetype)
    system(sprintf("convert -density 300 -trim %s %s", filename, output_filename))
}

to_jpeg <- partial(convert_img, filetype = "jpg")
to_png <- partial(convert_img, filetype = "png")

# ggsave the last plot
ggsave_factory <- function(directory, unit = "mm") {
    function(name, width, height, style = NULL) {
        filetype <- style_to_filetype(style)
        filepath <- sprintf("%s/%s-%s.%s", directory, name, style, filetype)

        # TODO Describe rationale in comment
        if (filetype %in% c("svg", "pdf", "eps", "tex")) {
            filetype = "svg"
        }

        ggsave(filepath, width = width, height = height, units = unit)

        if (filetype %in% c("svg", "pdf", "eps", "tex")) {
            filetype = "svg"
            convert_img("ft-raxml-empirical-slides.svg", filetype)
        }
    }
}
