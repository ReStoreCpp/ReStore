library(ggplot2)
library(dplyr)
library(readr)
library(cowplot)

# Set the current working directory
setwd("~/projects/ReStore/experiments/simulate-failures-until-data-loss")

# Load the simulations' results
data <- read_csv("simulate_failures_until_data_loss.csv",
  col_types = cols(
    simulation_id = col_character(),
    mode = col_character(),
    seed = col_integer(),
    num_ranks = col_integer(),
    num_blocks = col_integer(),
    replication_level = col_integer(),
    failures_until_data_loss = col_integer()
))

# Plot: Keep the number of blocks constant, color code the replication level k and vary the number of ranks
data %>%
  filter(simulation_id == "indep-ranks-rankwise") %>%
  ggplot(aes(x = as.factor(num_ranks), y = failures_until_data_loss/num_ranks, color = as.factor(replication_level))) +
    geom_point(position = position_dodge(0.3)) +
    #scale_x_log10() +
    scale_y_log10(labels = scales::percent) +
    scale_color_brewer(palette = "Dark2") +
    theme_cowplot(12) +
    xlab("number of ranks") +
    ylab("failed ranks until data loss") +
    labs(color = "replication level") +
    theme(legend.position = c(0.02, 0.1))

# Keep k and the number of ranks constant, vary the number of blocks
# Ad expected, the number of blocks does not change anything
data %>%
  filter(simulation_id == "indep-blocks") %>%
  ggplot(aes(x = num_blocks, y = failures_until_data_loss/num_ranks, color = as.factor(num_ranks))) +
  geom_point(position = position_dodge(0.3)) +
  scale_x_log10() +
  scale_y_continuous(labels = scales::percent) +
  scale_color_brewer(palette = "Dark2") +
  theme_cowplot(12)

# Simulate node and rack failures