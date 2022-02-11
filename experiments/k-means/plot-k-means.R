library(readr)
library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(tidyr)

# Automatically use showtext to render text for future devices.
showtext_auto()

# Load Computer Modern font
font_add("Computer Modern", "cmunrm.ttf")

# Read the input data
input_dir = "~/projects/ReStore/results/k-means-overhead/"
output_dir <- input_dir
setwd(input_dir)

ft_off_data <- 
  list.files(pattern = "ft-off.*.intelmpi.*.csv") %>% 
  map_df(~fread(.)) %>%
  # Older versions of the benchmark have a bug where the replication level is
  # not initialized if fault tolerance is turned off.
  mutate(replicationLevel = 0)

ft_on_data <- 
  list.files(pattern = "ft-on.*.intelmpi.*.csv") %>% 
  map_df(~fread(.))

# Joint the data from the experiments with and without failure simulation.
# The experiments without simulated failures will have fewer columns, these are
# filled with NA (the timings for code sections only executed upon a failure).
data <- bind_rows(ft_off_data, ft_on_data)

# Compute the number of nodes we started with and ended with (after failures).
# This is different than for page-rank!
data <- data %>%
  mutate(
    numRanksStart = numRanks + numSimulatedRankFailures,
    numRanksEnd = numRanks
  ) %>%
  select(-numRanks)

# Replace missing (NA) timings with 0 [s]
data <- data %>% replace_na(
  list(
    `center-rollback` = 0, `center-rollback` = 0, `fix-communicator` = 0,
    `rebalance-after-failure` = 0, `restore-data` = 0, `submit-data` = 0,
    `reassign-points-after-failure` = 0, `update-centers-after-failure` = 0,
    `commit-to-restoration` = 0, `checkpoint-creation` = 0
))

# Discretize useFaultTolerance
data <- data %>% mutate(
  useFaultTolerance = if_else(useFaultTolerance == 0, "off", "on")
)

# Remove data loading from total
data <- data %>% mutate(
  total = total - `load-data`
)

# Aggregate timings
data <- data %>% mutate(
    # Negligible timings
    `other-ft-mechanisms` = `center-rollback` + `checkpoint-creation` + `commit-to-restoration` + `fix-communicator` +
                            `reassign-points-after-failure` + `update-centers-after-failure`,
    # Re-balance after a failure
    #`rebalance-after-failure` = `get-ranks-died-since-last-call` + `get-my-original-rank` +
    #                            `get-new-blocks-after-failure-for-pull-blocks`,
    # Restore overhead
    `restore-overhead` = `submit-data` + `restore-data`
)

# Sanity checks and ensuring that we're not comparing apples and oranges.
stopifnot(data %>% filter(useFaultTolerance == 0) %>% pull(numSimulatedRankFailures) == 0 )

stopifnot(data$numDataPointsPerRank > 0)
stopifnot(data %>% select(numDataPointsPerRank) %>% unique() %>% length() == 1)

stopifnot(data$numCenters > 0)
stopifnot(data %>% select(numCenters) %>% unique() %>% length() == 1)

stopifnot(data$numDimensions > 0)
stopifnot(data %>% select(numDimensions) %>% unique() %>% length() == 1)

stopifnot(data$numIterations > 0)
stopifnot(data %>% select(numIterations) %>% unique() %>% length() == 1)

stopifnot(data$useFaultTolerance == "off" | data$useFaultTolerance == "on")

stopifnot(data %>% select(replicationLevel) %>% unique() %>% length() == 1)

# Pivot longer
data <- data %>%
  pivot_longer(
    c(`load-data`, `pick-centers`, `perform-iterations`, `total`,`submit-data`,                     
      `checkpoint-creation`, `center-rollback`, `fix-communicator`,
      `rebalance-after-failure`,
      #`get-ranks-died-since-last-call`, `get-my-original-rank`, `get-new-blocks-after-failure-for-pull-blocks`,
      `other-ft-mechanisms`, 
      `restore-data`, `commit-to-restoration`, `reassign-points-after-failure`,
      `update-centers-after-failure`, `restore-overhead`
    ),
    names_to = "timer",
    values_to = "time_s"
  )

# How many failures are there?
data %>% filter(useFaultTolerance == "on") %>%
ggplot(aes(x = numRanksStart, y = numSimulatedRankFailures)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 0.01)

# Group the replicates and compute summary statistics
aggregated_data <- data %>%
  group_by(
    simulationId, numDataPointsPerRank, numCenters, numIterations,
    numDimensions, useFaultTolerance, replicationLevel, clusterCenterSeed,
    failureProbability, numRanksStart, timer
  ) %>%
  summarise(
    time_mean_s = mean(time_s),
    time_sd_s = sd(time_s),
    time_q10_s = quantile(time_s, probs = c(0.1)),
    time_q90_s = quantile(time_s, probs = c(0.9)),
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
  filter(
    #timer %in% c("submit-data", "restore-data", "perform-iterations", "total")
    timer %in% c("restore-overhead", "perform-iterations", "total")
    #!(timer %in% c("center-rollback", "checkpoint-creation", "pick-centers", 
    #  "commit-to-restoration", "fix-communicator", "load-data",
    #  "reassign-points-after-failure", "update-centers-after-failure"))
  ) %>%
  mutate(
    timer = factor(recode(timer,
      `perform-iterations` = "k-means loop",
      `submit-data` = "submit data to restore",
      `restore-data` = "restore lost data",
      `rebalance-after-failure` = "rebalance after failure",
      `other-ft-mechanisms` = "other ft mechanisms",
      `restore-overhead` = "restore overhead",
      `total` = "overall runtime"),
      levels = c("k-means loop", "restore overhead", "submit data to restore", "rebalance after failure",
                 "restore lost data", "other ft mechanisms", "overall runtime")
  )) %>% 
ggplot(
  aes(
    x = numRanksStart,
    y = time_mean_s,
    color = useFaultTolerance,
    # ymin = time_mean_s - time_sd_s,
    # ymax = time_mean_s + time_sd_s
    ymin = time_q10_s,
    ymax = time_q90_s
  )) +
  geom_line_with_points_and_errorbars() +
  scale_x_log10(breaks = x_breaks) +
  facet_wrap(
    vars(timer),
    nrow = 1,
    scales = "fixed"
  ) +
  theme_husky(
    # Angle the x-axis labels
    axis.text.x = element_text(angle = 45, hjust = 1),
    
    # Move legend position and remove its title
    legend.position = c(1, 0),
    legend.justification = c("right", "bottom"),
  ) +
  labs(
    x = "#PEs",
    y = "time [s]",
    color = "rank failures"
  )
ggsave(
  paste(output_dir, "k-means.pdf", sep = '/'),
  width = 120, height = 50, units = "mm"
)

# time per failure
aggregated_data %>%
  filter(timer == "restore-data") %>%
  mutate(
    meanTimePerFailure_s = time_mean_s / failures_mean,
    sdTimePerFailure_s = time_sd_s /failures_mean
    ) %>%
ggplot(aes(x = numRanksStart, y = meanTimePerFailure_s, color = useFaultTolerance)) +
  geom_line() +
  geom_point() +
  geom_errorbar(
    aes(
      ymin = meanTimePerFailure_s - sdTimePerFailure_s,
      ymax = meanTimePerFailure_s + sdTimePerFailure_s
    ),
    width = 0.2
  ) +
  facet_wrap(~factor(
    timer,
    levels = c(
      "pick-centers", "perform-iterations", "submit-data", "restore-data",
      "rebalance-after-failure",  "other-ft-mechanisms", "total", "get-ranks-died-since-last-call",
      "get-my-original-rank", "get-new-blocks-after-failure-for-pull-blocks"
    ))) +
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

# slowdown introduced by fault-tolerance and failures
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
  ) %>%
  mutate(
    slowdown = ft_time_s / noft_time_s
  )

slowdown_data %>%
ggplot(aes(x = numRanks, y = slowdown)) +
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

# Overhead of ReStore
data %>%
  filter(
    numDataPointsPerRank == 65536,
    numCenters == 20,
    numIterations == 500,
    useFaultTolerance == "on",
    replicationLevel == 4,
    blocksPerPermutationRange == 4096,
    timer %in% c("total", "restore-overhead")
  ) %>%
  select(repeatId, timer, time_s, numRanksStart) %>%  
  pivot_wider(
    id_cols = c("repeatId", "numRanksStart"),
    names_from = "timer",
    values_from = "time_s"
  ) %>%
  mutate(
    slowdon_by_restore = `restore-overhead` / total
  ) %>%
  summary()
  
  