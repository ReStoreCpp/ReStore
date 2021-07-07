library(ggplot2)
library(dplyr)
library(readr)
library(gdata)
library(purrr)
library(stringr)

# Set the current working directory
setwd("~/projects/ReStore/experiments/id-compression")

# Load the simulations' results
data <- read_csv("results.csv",
    col_types = cols(
      code = col_character(),
      numberOfRanks = col_integer(),
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
      levels = c("  8 B  ", " 16 B  ", " 32 B  ", " 64 B  ", "128 B  ",
                 "256 B  ", "512 B  ", "  1 KiB", "  1 MiB")
  )) %>%
  group_by(numberOfRanks, benchmark, code, bytesPerBlock, replicationLevel, bytesPerRank) %>%
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

plot_grid <- function(data, benchmarkName, x, xlab, facet_x, facet_x_description,
                      facet_y, facet_y_description, color, color_label,
                      facet_wrap = "both") {
  x <- enquo(x)
  facet_x <- enquo(facet_x)
  facet_y <- enquo(facet_y)
  color <- enquo(color)
  
  # Create the labeller for the facets
  remove_first_char = function(string) {
    return(substr(string, 1, nchar(string))[2])
  }
  
  facet_labellers = c(
    facet_labeller_factory(facet_x_description),
    facet_labeller_factory(facet_y_description)
  )
  names(facet_labellers) <- c(remove_first_char(facet_x), remove_first_char(facet_y))
  
  data <- data %>%
    mutate(
      ymin = max(1, (real_time_mean - real_time_sd) / bytesPerRank),
      ymax = (real_time_mean + real_time_sd) / bytesPerRank
    )
  
  # Plot
  plot <- data %>%
    filter(benchmark == benchmarkName) %>%
    ggplot(
      aes(
        x = !!x,
        color = !!color,
        y = real_time_mean / bytesPerRank,
        ymin = ymin,
        ymax = ymax,
        group = !!color
    )) + 
    geom_point() +
    geom_path() +
    geom_errorbar(width = 0.25) +
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
    ) +
    xlab(xlab) +
    ylab("time per byte per rank [ns]") +
    labs(color = color_label)

  if (facet_wrap == "both") {
    plot <- plot +  
      facet_grid(
        rows = vars(!!facet_y),
        cols = vars(!!facet_x),
        labeller = do.call("labeller", facet_labellers)
      )
  } else if (facet_wrap == "single") {
    plot <- plot + 
      facet_wrap(
        vars(!!facet_x),
        labeller = do.call("labeller", facet_labellers)
      )
  }
  
  print(plot)
}

# Subset the data
tuning_evaluation_data <- filter(data,
  code %in%
    c("without-id-compression", "with-id-compression", "tuned-id-compression")
  && replicationLevel %in% c(2, 3, 4)
  && bytesPerBlock %in% c("  8 B  ", "  1 KiB", "  1 MiB")
)

eval_tuned_data <- filter(data,
  code == "tuned-id-compression"
)

scaling_evaluation_data <- filter(data,
  str_detect(code, "^var-pe-[[:digit:]]+$")
  && replicationLevel == 3
)

# Runtime depending on the number of bytes per rank
plot_grid(
  data = tuning_evaluation_data,
  benchmarkName = "submitBlocks",
  x = bytesPerRankHR,
  xlab = "bytes per rank",
  facet_x = replicationLevel,
  facet_x_description = "replication level",
  facet_y = bytesPerBlock,
  facet_y_description = "bytes per block",
  color = code,
  color_label = "code"
)

# Runtime depending on the number of bytes per block
plot_grid(
  data = tuning_evaluation_data,
  benchmarkName = "submitBlocks",
  x = bytesPerBlock,
  xlab = "bytes per block",
  facet_x = replicationLevel,
  facet_x_description = "replication level",
  facet_y = bytesPerRankHR,
  facet_y_description = "bytes per rank",
  color = code,
  color_label = "code"
)

# Runtime depending on the replication level
plot_grid(
  data = tuning_evaluation_data,
  benchmarkName = "submitBlocks",
  x = as.factor(replicationLevel),
  xlab = "replication level",
  facet_x = bytesPerBlock,
  facet_x_description = "bytes per block",
  facet_y = bytesPerRankHR,
  facet_y_description = "bytes per rank",
  color = code,
  color_label = "code"
)

# More data for the tuned algorithm
plot_grid(
  data = eval_tuned_data,
  benchmarkName = "submitBlocks",
  x = bytesPerBlock,
  xlab = "bytes per block",
  facet_x = bytesPerRankHR,
  facet_x_description = "bytes per rank",
  facet_y = NA,
  facet_y_description = NA,
  color = as.factor(replicationLevel),
  color_label = "replication level",
  facet_wrap = "single"
)

# Scaling with the number of PEs
plot_grid(
  data = scaling_evaluation_data,
  benchmarkName = "submitBlocks",
  x = as.factor(numberOfRanks * 40),
  xlab = "number of ranks",
  facet_x = bytesPerBlock,
  facet_x_description = "bytes per block",
  facet_y = NA,
  facet_y_description = NA,
  color = bytesPerRankHR,
  color_label = "bytes per rank",
  facet_wrap = "single"
)


#### pushBlocks
# Runtime depending on the number of bytes per rank
plot_grid(
  data = tuning_evaluation_data %>%
    filter(
      code == "tuned-id-compression",
      replicationLevel == 3
    ),
  benchmarkName = "pushBlocks",
  x = bytesPerRankHR,
  xlab = "bytes per rank",
  facet_x = bytesPerBlock,
  facet_x_description = "bytes per block",
  facet_y = NA,
  facet_y_description = NA,
  color = as.factor(bytesPerBlock),
  color_label = "bytes per block",
  facet_wrap = "none"
)

# Runtime depending on the number of bytes per block
plot_grid(
  data = tuning_evaluation_data %>% filter(code == "tuned-id-compression"),
  benchmarkName = "pushBlocks",
  x = bytesPerBlock,
  xlab = "bytes per block",
  facet_x = bytesPerRankHR,
  facet_x_description = "bytes per rank",
  facet_y = NA,
  facet_y_description = NA,
  color = as.factor(replicationLevel),
  color_label = "replication level",
  facet_wrap = "single"
)

# Scaling with the number of PEs
# TODO Fix name of numberOfRanks
plot_grid(
  data = scaling_evaluation_data %>%
    filter(bytesPerBlock == "  1 KiB"),
  benchmarkName = "pushBlocks",
  x = as.factor(numberOfRanks * 40),
  xlab = "number of ranks",
  facet_x = bytesPerBlock,
  facet_x_description = "bytes per block",
  facet_y = NA,
  facet_y_description = NA,
  color = bytesPerRankHR,
  color_label = "bytes per rank",
  facet_wrap = "none"
)
