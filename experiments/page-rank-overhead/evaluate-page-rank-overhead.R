library(readr)
library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(tidyr)

# Read the input data
setwd("~/projects/ReStore/experiments/page-rank-overhead/")

ft_off_data <- read_csv(
  "measurements-ft-off.csv",
  col_types = cols(
    numRanks = col_integer(),
    numReplications = col_integer(),
    numIterations = col_integer(),
    numVertices = col_double(),
    numEdges = col_double(),
    numRanksFailed = col_integer(),
    failureProbability = col_double(),
    graphReadingTime = col_double(),
    seed = col_double(),
    Init = col_double(),
    Pagerank = col_double(),
    total = col_double()
)) %>%
  mutate(
    useFaultTolerance = "off"
  )

ft_on_data <- read_csv(
  "measurements-ft-on.csv",
  col_types = cols(
    numRanks = col_integer(),
    numReplications = col_integer(),
    numIterations = col_integer(),
    numVertices = col_double(),
    numEdges = col_double(),
    numRanksFailed = col_integer(),
    failureProbability = col_double(),
    graphReadingTime = col_double(),
    seed = col_double(),
    Init = col_double(),
    Pagerank = col_double(),
    total = col_double()
)) %>%
  mutate(
    useFaultTolerance = "on"
  )

# Joint the data from the experiments with and without failure simulation.
# The experiments without simulated failures will have fewer columns, these are
# filled with NA (the timings for code sections only executed upon a failure).
data <- bind_rows(ft_off_data, ft_on_data)

# Compute the number of nodes we started with and ended with (after failures).
# This is different than for k-means!
data <- data %>%
  mutate(
    numRanksStart = numRanks,
    numRanksEnd = numRanks - numRanksFailed
  ) %>%
  select(-numRanks)

# Replace missing (NA) timings with 0 [s]
data <- data %>% replace_na( list( `Recovery` = 0, `Recomputation` = 0 ))

# `total` already does not contain loading the input graph.

# Rename some columns to be more in-line with the k-means measurements.
data <- data %>%
  rename(
    numSimulatedRankFailures = numRanksFailed,
    replicationLevel = numReplications,
    `submit-data` = Init,
    `pagerank-iterations` = Pagerank,
    `restore-data` = Recovery,
    `recompute-lost-data` = Recomputation
  )

# Sanity checks and ensuring that we're not comparing apples and oranges.
stopifnot(
  data %>%
    filter(useFaultTolerance == 0) %>%
    pull(numSimulatedRankFailures)
  == 0
)

stopifnot(data$numVertices > 0)
stopifnot(data$numEdges > 0)
stopifnot(data %>% select(numVertices) %>% unique() %>% length() == 1)
stopifnot(data %>% select(numEdges) %>% unique() %>% length() == 1)

stopifnot(data$numIterations > 0)
stopifnot(data %>% select(numIterations) %>% unique() %>% length() == 1)

stopifnot(data$replicationLevel > 0)
stopifnot(data %>% select(replicationLevel) %>% unique() %>% length() == 1)

stopifnot(data$numIterations > 0)
stopifnot(data %>% select(numIterations) %>% unique() %>% length() == 1)

stopifnot(data$useFaultTolerance == "off" | data$useFaultTolerance == "on")

# Pivot longer
data <- data %>%
  pivot_longer(
    c(`submit-data`, `pagerank-iterations`, `restore-data`,
      `recompute-lost-data`, `total`),
    names_to = "timer",
    values_to = "time_s"
  )

# How many failures are there?
data %>% filter(useFaultTolerance == "on") %>%
  ggplot(aes(x = numRanksStart, y = numSimulatedRankFailures)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 0.05)

# Group the replicates and compute summary statistics
aggregated_data <- data %>%
  group_by(
    replicationLevel, numIterations, numVertices, numEdges, numRanksStart,
    timer, useFaultTolerance
  ) %>%
  summarise(
    time_mean_s = mean(time_s),
    time_sd_s = sd(time_s),
    failures_min = min(numSimulatedRankFailures),
    failures_max = max(numSimulatedRankFailures),
    failures_mean = mean(numSimulatedRankFailures),
    failures_sd = sd(numSimulatedRankFailures),
    repeats = n(),
    .groups = "keep"
  )

# TODO How many runs did not finish?

# Plot the data
x_breaks <- aggregated_data %>% pull(numRanksStart) %>% unique
aggregated_data %>%
  #filter(
  #  !(timer %in% c("center-rollback", "checkpoint-creation", "total",
  #                 "commit-to-restoration", "fix-communicator", "load-data",
  #                 "reassign-points-after-failure", "update-centers-after-failure"))
  #) %>%
  ggplot(aes(x = numRanksStart, y = time_mean_s, color = useFaultTolerance)) +
  geom_line() +
  geom_point() +
  geom_errorbar(
    aes(
      ymin = time_mean_s - time_sd_s,
      ymax = time_mean_s + time_sd_s
    ),
    width = 0.2
  ) +
  facet_wrap(~factor(
    timer,
    levels = c("submit-data", "pagerank-iterations", "restore-data",
               "recompute-lost-data", "total")
  )) +
  theme_bw() +
  theme(
    # Remove unneeded grid elements
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
    
    # Angle the x-axis labels
    axis.text.x = element_text(angle = 45, hjust = 1),
    
    # Move legend position and remove its title
    legend.position = c(0.15, 0.85)
  ) +
  scale_x_log10(breaks = x_breaks) +
  labs(
    x = "#ranks",
    y = "time [s]",
    color = "rank failures"
  )

slowdown_data <- data %>%
  filter(timer == "total") %>%
  select(
    -replicationLevel, -failureSimulatorSeed, -failureProbability,
    -numSimulatedRankFailures, -numRanksEnd, -timer
  )
slowdown_data <-
  inner_join(
    slowdown_data %>% filter(useFaultTolerance == "off"),
    slowdown_data %>% filter(useFaultTolerance == "on"),
    by = c(
      "simulationId", "numDataPointsPerRank",  "numCenters", "numIterations",
      "numDimensions", "clusterCenterSeed", "numRanksStart",  "repeatId"
    )) %>%
  select(-useFaultTolerance.x, -useFaultTolerance.y) %>%
  rename(
    noft_time_s = time_s.x,
    ft_time_s = time_s.y,
    numRanks = numRanksStart,
  )

slowdown_data %>%
  ggplot(aes(x = numRanks, y = ft_time_s / noft_time_s)) +
  geom_point() +
  theme_bw() +
  theme(
    # Remove unneeded grid elements
    panel.grid.minor.x = element_blank(),
    #panel.grid.major.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    #panel.grid.major.y = element_blank(),
    
    # Angle the x-axis labels
    #axis.text.x = element_text(angle = 45, hjust = 1),
    
    # Move legend position and remove its title
    legend.position = c(0.15, 0.85),
    legend.title = element_blank()
  ) +
  scale_x_log10(breaks = x_breaks) +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  xlab("#ranks") +
  ylab("slowdown caused by fault-tolerance and faults")
