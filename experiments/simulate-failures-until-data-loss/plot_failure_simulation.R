library(ggplot2)
library(dplyr)
library(readr)
library(cowplot)

# Set the current working directory
setwd("~/projects/ReStore/experiments/simulate-failures-until-data-loss")

# Load the simulations' results
data <- read_csv("simulate-failures-until-data-loss.csv",
  col_types = cols(
    simulationId = col_character(),
    mode = col_character(),
    seed = col_integer(),
    numRanks = col_integer(),
    numBlocks = col_integer(),
    replicationLevel = col_integer(),
    failuresUntilDataLoss = col_integer()
))

# Plot: Keep the number of blocks constant, color code the replication level k and vary the number of ranks
plotFailuresUntilDataLossOverNumRanks <- function(simulationMode) {
  data %>%
    filter(simulationId == simulationMode) %>%
    ggplot(aes(x = as.factor(numRanks), y = failuresUntilDataLoss/numRanks, color = as.factor(replicationLevel))) +
      geom_point(position = position_dodge(0.3)) +
      #scale_x_log10() +
      scale_y_log10(labels = scales::percent) +
      scale_color_brewer(palette = "Dark2") +
      theme_cowplot(12) +
      xlab("number of ranks") +
      ylab("failed ranks until data loss") +
      labs(color = "replication level") +
      theme(legend.position = c(0.02, 0.1))
}

data %>%
  filter(simulationId == "indep-ranks-rankwise") %>%
  group_by(
    numRanks, numBlocks, replicationLevel 
  ) %>%
  summarise(
    mean = mean(failuresUntilDataLoss),
    sd = sd(failuresUntilDataLoss)
  ) %>%
  arrange(replicationLevel)

plotFailuresUntilDataLossOverNumRanks("indep-ranks-rankwise")
plotFailuresUntilDataLossOverNumRanks("indep-ranks-nodewise")
plotFailuresUntilDataLossOverNumRanks("indep-ranks-rackwise-40rpr")
plotFailuresUntilDataLossOverNumRanks("indep-ranks-rackwise-80rpr")
plotFailuresUntilDataLossOverNumRanks("indep-ranks-rackwise-160rpr")


joined_rankwise_nodewise <- left_join(
  data %>%
    filter(simulationId == "indep-ranks-rankwise") %>%
    select(-simulationId, -mode) %>%
    mutate(numRanks = numRanks * 20),
  data %>% filter(simulationId == "indep-ranks-nodewise") %>% select(-simulationId, -mode),
  by = c("seed", "replicationLevel", "numRanks")
)
sum(joined_rankwise_nodewise$failuresUntilDataLoss.x * 20
    != joined_rankwise_nodewise$failuresUntilDataLoss.y)

joined_rankwise_rackwise_40rpr <- left_join(
  data %>%
    filter(simulationId == "indep-ranks-rankwise") %>%
    select(-simulationId, -mode) %>%
    mutate(numRanks = numRanks * 40),
  data %>% filter(simulationId == "indep-ranks-rackwise-40rpr") %>% select(-simulationId, -mode),
  by = c("seed", "replicationLevel", "numRanks")
)
sum(joined_rankwise_rackwise_40rpr$failuresUntilDataLoss.x * 40
    != joined_rankwise_rackwise_40rpr$failuresUntilDataLoss.y)

joined_rankwise_rackwise_80rpr <- left_join(
  data %>%
    filter(simulationId == "indep-ranks-rankwise") %>%
    select(-simulationId, -mode) %>%
    mutate(numRanks = numRanks * 80),
  data %>% filter(simulationId == "indep-ranks-rackwise-80rpr") %>% select(-simulationId, -mode),
  by = c("seed", "replicationLevel", "numRanks")
)
sum(joined_rankwise_rackwise_80rpr$failuresUntilDataLoss.x * 80
    != joined_rankwise_rackwise_80rpr$failuresUntilDataLoss.y)

joined_rankwise_rackwise_160rpr <- left_join(
  data %>%
    filter(simulationId == "indep-ranks-rankwise") %>%
    select(-simulationId, -mode) %>%
    mutate(numRanks = numRanks * 160),
  data %>% filter(simulationId == "indep-ranks-rackwise-160rpr") %>% select(-simulationId, -mode),
  by = c("seed", "replicationLevel", "numRanks")
)
sum(
  joined_rankwise_rackwise_160rpr %>% filter(numRanks != 160000000) %>% pull(failuresUntilDataLoss.x) * 160
    != joined_rankwise_rackwise_160rpr%>% filter(numRanks != 160000000) %>% pull(failuresUntilDataLoss.y)
)

# Keep k and the number of ranks constant, vary the number of blocks
# As expected, the number of blocks does not change anything
data %>%
  filter(simulationId == "indep-blocks") %>%
  ggplot(aes(x = numBlocks, y = failuresUntilDataLoss/numRanks, color = as.factor(numRanks))) +
  geom_point(position = position_dodge(0.3)) +
  scale_x_log10() +
  scale_y_continuous(labels = scales::percent) +
  scale_color_brewer(palette = "Dark2") +
  theme_cowplot(12)

# Simulate node and rack failures
data %>%
  filter(simulationId == "indep-")