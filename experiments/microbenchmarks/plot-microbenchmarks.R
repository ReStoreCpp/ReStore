source("../common.R")

NUM_RANKS_PER_NODE <- 48

# Set the current working directory
input_dir <- "~/projects/ReStore/experiments/microbenchmarks"
output_dir <- input_dir
setwd(input_dir)

# Select the data source depending on the plot you want to generate.
#csv_file <- "determine-optimal-blocksPerPermutationRange.csv"
csv_file <- "microbenchmarks.csv"

asPowerOfTwo <- function(n) {
  treshold <- 1024
  exponent <- log2(n)
  stopifnot(2**exponent == n)
  
  return(ifelse (n <= treshold, 
          paste(n),
          paste("2^", exponent, sep = ''))
  )
}
# TODO generalize this
powerOfTwoLevels <- c("1", "2", "4", "8", "16", "32", "64", "128", "256", "512",
                      "1024", "2^11", "2^12", "2^13", "2^14", "2^15", "2^16",
                      "2^17", "2^18", "2^19") 
bytesPerRankLevels <- c(" 16 KiB", " 64 KiB", "256 KiB", "  1 MiB", "  4 MiB",
                        " 16 MiB", " 64 MiB")
#bytesPerPermutationRangeLevels <- c(" 64 B  ", "128 B  ", "256 B  ", "512 B  ", "  1 KiB", "  2 KiB", "  4 KiB",
#                                    "  8 KiB", " 16 KiB", " 32 KiB", " 64 KiB", "128 KiB", "256 KiB", "512 KiB",
#                                    "  1 MiB", "  2 MiB", "  4 MiB", "  8 MiB", " 16 MiB")
bytesPerPermutationRangeLevels <- c("64 B", "128 B", "256 B", "512 B", "1 KiB", "2 KiB", "4 KiB",
                                    "8 KiB", "16 KiB", "32 KiB", "64 KiB", "128 KiB", "256 KiB", "512 KiB",
                                    "1 MiB", "2 MiB", "4 MiB", "8 MiB", "16 MiB")

# Load the simulations' results
data <- read_csv(csv_file,
    col_types = cols(
      idRandomization = col_character(),
      numberOfNodes = col_integer(),
      benchmark = col_character(),
      bytesPerBlock = col_double(),
      replicationLevel = col_integer(),
      bytesPerRank = col_double(),
      promilleOfRanksThatFail = col_integer(),
      blocksPerPermutationRange = col_integer(),
      iterations = col_integer(),
      real_time = col_double(),
      cpu_time = col_double(),
      time_unit = col_character()
  )) %>%
  mutate(
    numberOfRanks = numberOfNodes * NUM_RANKS_PER_NODE,
    numberOfRankFailures = ceiling(numberOfRanks * promilleOfRanksThatFail / 1000.),
    real_time = real_time, # * 10^6,
    cpu_time = cpu_time, # * 10^6,
    time_unit = "ns",
    bytesPerBlockHR = humanReadable(bytesPerBlock, standard = "IEC", digits = 0),
    bytesPerBlockHR = factor(bytesPerBlockHR, levels = c("64 B")),
    bytesPerRankHR = humanReadable(bytesPerRank, standard = "IEC", digits = 0),
    #bytesPerRankHR = factor(bytesPerRankHR,  levels = bytesPerPermutationRangeLevels),
    blocksPerPermutationRangeHR = factor(asPowerOfTwo(blocksPerPermutationRange), powerOfTwoLevels),
    bytesPerPermutationRange = blocksPerPermutationRange * bytesPerBlock,
    bytesPerPermutationRangeHR = trimws(humanReadable(bytesPerPermutationRange, standard = "IEC", digits = 0)),
    bytesPerPermutationRangeHR = factor(bytesPerPermutationRangeHR, levels = bytesPerPermutationRangeLevels),
    idRandomization = factor(recode(idRandomization, `FALSE` = "Off", `TRUE` = "On")),
    benchmarkHR = factor(recode(benchmark,
      submitBlocks = "submit to restore",
      pullBlocksRedistribute = "load from restore (all data)",
      pullBlocksSmallRange = "load from restore (1 % of data)",
      DiskRedistribute = "load from disk (all data)",
      DiskSmallRange = "load from disk (1 % of data)",
      pullBlocksSingleRank = "load from restore (data of a single rank)",
      DiskSingleRank = "load from disk (data of a single rank)"
    ), levels = c("submit to restore", "load from restore (all data)", "load from restore (1 % of data)",
                  "load from disk (all data)", "load from disk (1 % of data)",
                  "load from restore (data of a single rank)", "load from disk (data of a single rank)"))
  ) %>%
  group_by(
    numberOfRanks, benchmark, benchmarkHR, bytesPerBlock, replicationLevel, bytesPerRank, blocksPerPermutationRange,
    blocksPerPermutationRangeHR, bytesPerRankHR, promilleOfRanksThatFail, idRandomization, bytesPerPermutationRangeHR,
    bytesPerPermutationRange
  ) %>%
  summarize(
    real_time_mean = mean(real_time),
    real_time_sd = sd(real_time),
    #ymin = real_time_mean - real_time_sd,
    #ymax = real_time_mean + real_time_sd,
    #ymin = min(real_time),
    #ymax = max(real_time),
    ymin = quantile(real_time, probs = c(0.1)),
    ymax = quantile(real_time, probs = c(0.9)),
    .groups = "keep"
  ) %>%
  filter(replicationLevel != 1)

# Define some labellers for facets
# facet_labeller_width_description = function(description, value) {
#   return(paste(description, ": ", value, sep = ""))
# }
# 
# facet_labeller_factory <- function(description) {
#   return(partial(facet_labeller, description))
# }

# Which measurements are available?
data %>% pull(bytesPerRankHR) %>% unique()
data %>% pull(replicationLevel) %>% unique()
data %>% pull(blocksPerPermutationRange) %>% unique()
data %>% pull(promilleOfRanksThatFail) %>% unique()
data %>% pull(benchmark) %>% unique()
data %>% pull(idRandomization) %>% unique()

### What is the ideal value for blocksPerPermutationRange? ###
# load determine-optimal-blocksPerPermutationRange.csv
data %>%
    filter(
      bytesPerRankHR == "16 MiB",
      replicationLevel == 4,
      idRandomization == "On",
  ) %>%
  mutate(
    benchmarkHR = recode(benchmarkHR,
      "submit to restore" = "submit",
      "load from restore (1 % of data)" = "load 1 % of data"
  )) %>%
ggplot(
    aes(
      x = bytesPerPermutationRangeHR,
      color = as.factor(numberOfRanks),
      y = real_time_mean,
      ymin = ymin,
      ymax = ymax,
      group = numberOfRanks,
  )) +
  geom_line_with_points_and_errorbars() +
  geom_vline(
    aes(xintercept = which(levels(bytesPerPermutationRangeHR) == "256 KiB")),
    linetype = "dashed"
  ) +
  scale_color_dark2() +
  scale_y_log_with_ticks_and_lines() +
  labs(
    x = "bytes per permutation range",
    y = "time [ms]",
    color = "#PEs"
  ) +
  facet_wrap(
    scales = "free",
    ncol = 2,
    vars(benchmarkHR),
  ) +
  theme_husky(
    legend.position = "bottom",
    legend.box.margin = margin(1, 1, 1, 1),
    legend.box.spacing = margin(0, 0, 0, 0),
    # legend.position = c(1, 1),
    # legend.justification = c("right", "top"),
    # legend.box.just = "right",
    legend.text.align = 1,
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  )
ggsave(
  paste(output_dir, "determine-optimal-bytes-per-permutation-range.pdf", sep = '/'),
  width = 120, height = 60, units = "mm"
)

### ID randomization On vs Off? ###
# load microbenchmarks.csv
data %>%
    filter(
      bytesPerRankHR == " 16 MiB",
      replicationLevel == 4,
      bytesPerBlock == 64,
      blocksPerPermutationRange == 4096,
      promilleOfRanksThatFail == 10,
      benchmarkHR %in% c("submit to restore", "load from restore (all data)", "load from restore (1 % of data)")
  ) %>%
  mutate(
    benchmarkHR = recode(benchmarkHR,
      "submit to restore" = "submit",
      "load from restore (all data)" = "load all data",
      "load from restore (1 % of data)" = "load 1 % of data"
  )) %>%
ggplot(
    aes(
      x = as.factor(numberOfRanks),
      color = benchmarkHR,
      #color = as.factor(replicationLevel),
      y = real_time_mean,
      ymin = ymin,
      ymax = ymax,
      group = benchmarkHR,
  )) +
  geom_line_with_points_and_errorbars() +
  scale_color_dark2() +
  scale_y_log_with_ticks_and_lines() +
  labs(
    x = "#PEs",
    y = "time [ms]",
    color = "replication level"
  ) +
  facet_wrap(
    vars(idRandomization),
    labeller = labeller(idRandomization = c("Off" = "consecutive IDS", "On" = "permuted IDs"))
  ) +
  theme_husky(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.box.margin = margin(1, 1, 1, 1),
    legend.box.spacing = margin(0, 0, 0, 0),
    axis.text.x = element_text(angle = 45, hjust = 1),
  )
ggsave(
  paste(output_dir, "id-randomization-on-or-off.pdf", sep = '/'),
  width = 120, height = 70, units = "mm"
)

### Compare load-from-restore vs load-from-disk ###
# load microbenchmarks.csv
data %>%
    mutate(
      amountOfData = factor(recode(benchmark,
        pullBlocksRedistribute = "all data",
        pullBlocksSmallRange = "1 % of data",
        DiskRedistribute = "all data",
        DiskSmallRange = "1 % of data",
        pullBlocksSingleRank = "data of a single rank",
        DiskSingleRank = "data of a single rank"
      ), levels = c("data of a single rank", "1 % of data", "all data" )),
      benchmark = recode(benchmark,
        submitBlocks = "submit to restore",
        pullBlocksRedistribute = "load from restore",
        pullBlocksSmallRange = "load from restore",
        DiskRedistribute = "load from disk",
        DiskSmallRange = "load from disk",
        pullBlocksSingleRank = "load from restore",
        DiskSingleRank = "load from disk"
    )) %>%
    filter(
      bytesPerRankHR == " 16 MiB",
      replicationLevel == 4,
      idRandomization == "On" && (amountOfData == "1 % of data" || benchmark == "load from disk") || 
      idRandomization == "Off" && amountOfData == "all data",
      blocksPerPermutationRange == 4096,
      promilleOfRanksThatFail == 10,
      benchmark %in% c("load from disk", "load from restore"),
      amountOfData %in% c("1 % of data", "all data")
    ) %>%
ggplot(
    aes(
      x = as.factor(numberOfRanks),
      color = benchmark,
      y = real_time_mean,
      ymin = ymin,
      ymax = ymax,
      group = benchmark,
  )) +
  geom_line_with_points_and_errorbars() +
  scale_color_dark2() +
  scale_y_log_with_ticks_and_lines(scale_accuracy = 0.1) +
  labs(
    x = "#PEs",
    y = "time [ms]",
    color = "algorithm"
  ) +
  facet_wrap(
    scales = "fixed",
    ncol = 3,
    vars(amountOfData),
  ) +
  theme_husky(
    legend.position = c(1, 0),
    legend.justification = c("right", "bottom"),
    #legend.box.just = "right",
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
  )
ggsave(
  paste(output_dir, "load-from-restore-vs-load-from-disk.pdf", sep = '/'),
  width = 120, height = 60, units = "mm"
)