data_dir <- "~/projects/ReStore/experiments/failures-until-data-loss"
output_dir <- data_dir
setwd(data_dir)

source("../common.R")

# Load the simulated data
data <- read_csv(
  "failure-probability.csv",
  col_types = cols(
    numberOfPEs = col_integer(),
    replicationLevel = col_integer(),
    roundsUntilIrrecoverableDataLoss = col_integer()
  )) %>%
  group_by(numberOfPEs, replicationLevel) %>%
  summarize(
    roundsUntilIrrecoverableDataLoss_mean = mean(roundsUntilIrrecoverableDataLoss),
    roundsUntilIrrecoverableDataLoss_median = median(roundsUntilIrrecoverableDataLoss),
    roundsUntilIrrecoverableDataLoss_sd = sd(roundsUntilIrrecoverableDataLoss),
    roundsUntilIrrecoverableDataLoss_q10 = quantile(roundsUntilIrrecoverableDataLoss, probs = 0.9),
    roundsUntilIrrecoverableDataLoss_q90 = quantile(roundsUntilIrrecoverableDataLoss, probs = 0.1),
    .groups = "keep"
  )

# x-breaks
x_breaks <- data %>% pull(numberOfPEs) %>% unique

ggplot(
  data %>%
    mutate(replicationLevel = paste("r := ", replicationLevel, sep = "")),
  aes(
    x = numberOfPEs,
    y = roundsUntilIrrecoverableDataLoss_mean / numberOfPEs * 100,
    ymin = roundsUntilIrrecoverableDataLoss_q10 / numberOfPEs * 100,
    ymax = roundsUntilIrrecoverableDataLoss_q90 / numberOfPEs * 100,
    color = as.factor(replicationLevel),
    group = replicationLevel,
  )) +
  geom_line_with_points_and_errorbars() +
  scale_color_dark2() +
  scale_y_log_with_ticks_and_lines(scale_accuracy = 0.01) +
  scale_x_log10(breaks = x_breaks, label = comma_format(accuracy = 1)) +
  theme_husky(
    legend.position = c(0.025, 0),
    legend.justification = c("left", "bottom"),
    axis.text.x = element_text(angle = 45, hjust = 1),
  ) +
  labs(
    x = "#PEs",
    y = "% of PEs that failed\nuntil irrecoverable data loss",
    color = "redundant copies"
  )
ggsave(
  paste(output_dir, "failures-until-irrecoverable-data-loss.pdf", sep = '/'),
  width = 120, height = 55, units = "mm"
)
