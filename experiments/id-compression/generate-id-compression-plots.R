library(ggplot2)
library(dplyr)
library(readr)
library(gdata)
library(purrr)

# Set the current working directory
setwd("~/projects/ReStore/experiments/id-compression")

# Load the simulations' results
data <- read_csv("results.csv",
    col_types = cols(
      code = col_character(),
      benchmark = col_character(),
      bytesPerBlock = col_double(),
      replicationLevel = col_integer(),
      bytesPerRank = col_double(),
      iterations = col_integer(),
      real_time = col_double(),
      cpu_time = col_double(),
      time_unit = col_character()
  )) %>%
  mutate(
    real_time = real_time * 10^6,
    cpu_time = cpu_time * 10^6,
    time_unit = "ns",
    bytesPerBlock = humanReadable(bytesPerBlock, standard = "IEC", digits = 0),
    bytesPerBlock = factor(
      bytesPerBlock,
      levels = c("8 B  ", "1 KiB", "1 MiB")
  )) %>%
  group_by(benchmark, code, bytesPerBlock, replicationLevel, bytesPerRank) %>%
  summarize(
    real_time_mean = mean(real_time),
    real_time_sd = sd(real_time),
    .groups = "keep"
  ) %>%
  mutate(bytesPerRankHR = humanReadable(bytesPerRank, standard = "IEC", digits = 0))


facet_labeller = function(description, value) {
  return(paste(description, ": ", value, sep = ""))
}

facet_labeller_factory <- function(description) {
  return(partial(facet_labeller, description))
}

plot_grid <- function(data, benchmarkName, x, xlab, facet_x, facet_x_description, facet_y, facet_y_description) {
  x <- enquo(x)
  facet_x <- enquo(facet_x)
  facet_y <- enquo(facet_y)
  
  # Create the labeller for the facets
  remove_first_char = function(string) {
    return(substr(string, 1, nchar(string))[2])
  }
  
  facet_labellers = c(
    facet_labeller_factory(facet_x_description),
    facet_labeller_factory(facet_y_description)
  )
  names(facet_labellers) <- c(remove_first_char(facet_x), remove_first_char(facet_y))
  
  # Plot
  data %>%
    filter(benchmark == benchmarkName) %>%
  ggplot(
    aes(
      x = !!x,
      color = code,
      y = real_time_mean / bytesPerRank,
      ymin = (real_time_mean - real_time_sd) / bytesPerRank,
      ymax = (real_time_mean + real_time_sd) / bytesPerRank,
      group = code
    )) + 
    geom_point() +
    geom_path() +
    geom_errorbar(width = 0.25) +
    facet_grid(
      rows = vars(!!facet_y),
      cols = vars(!!facet_x),
      labeller = do.call("labeller", facet_labellers)
    ) + 
    scale_x_discrete() +
    scale_y_log10() +
    scale_color_brewer(type = "qual", palette = "Dark2") +
    theme_bw() +
    theme(
      # Remove unneeded grid elements
      panel.grid.minor.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_blank(),
      
      # Move legend position and remove its title
      #legend.position = c(0.15, 0.85),
      #legend.title = element_blank()
    ) +
    xlab(xlab) +
    ylab("time per byte per rank [ns]")
}

# Runtime depending on the number of bytes per rank
plot_grid(
  data = data,
  benchmarkName = "submitBlocks",
  x = bytesPerRankHR,
  xlab = "bytes per rank",
  facet_x = replicationLevel,
  facet_x_description = "replication level",
  facet_y = bytesPerBlock,
  facet_y_description = "bytes per block"
)

# Runtime depending on the number of bytes per block
plot_grid(
  data = data,
  benchmarkName = "submitBlocks",
  x = bytesPerBlock,
  xlab = "bytes per block",
  facet_x = replicationLevel,
  facet_x_description = "replication level",
  facet_y = bytesPerRankHR,
  facet_y_description = "bytes per rank"
)

# Runtime depending on the replication level
plot_grid(
  data = data,
  benchmarkName = "submitBlocks",
  x = as.factor(replicationLevel),
  xlab = "replication level",
  facet_x = bytesPerBlock,
  facet_x_description = "bytes per block",
  facet_y = bytesPerRankHR,
  facet_y_description = "bytes per rank"
)
