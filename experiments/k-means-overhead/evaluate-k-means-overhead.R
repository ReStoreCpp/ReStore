library(readr)
library(dplyr)
library(ggplot2)

setwd("./experiments/k-means-overhead/")
data <- read_csv("k-means-overhead.csv",
    col_types = cols(
        simulationId = col_character(),
        numDataPointsPerRank = col_double(),
        numCenters = col_double(),
        numIterations = col_double(),
        numDimensions = col_integer(),
        useFaultTolerance = col_logical(),
        replicationLevel = col_integer(),
        seed = col_double(),
        `load-data` = col_double(),
        `submit-data` = col_double(),
        `pick-centers` = col_double(),
        `perform-iterations` = col_double(),
        `checkpoint-creation` = col_double(),
        `num-simulated-rank-failures` = col_double()
    )
)

# Sanity checks and ensuring that we're not comparing apples and oranges.
assert(data$numDataPointsPerRank > 0)
assert(data %>% select(numDataPointsPerRank) %>% unique() %>% length() == 1)

assert(data$numCenters > 0)
assert(data %>% select(numCenters) %>% unique() %>% length() == 1)

assert(data$numDimensions > 0)
assert(data %>% select(numDimensions) %>% unique() %>% length() == 1)

assert(data$numIterations > 0)
assert(data %>% select(numIterations) %>% unique() %>% length() == 1)

assert(data$useFaultTolerance == "TRUE" | data$useFaultTolerance == "FALSE")

assert(data %>% select(replicationLevel) %>% unique() %>% length() == 1)

# TODO: Group the replicates

# TODO: Pivot longer

# TODO: Plot the data
ggplot(aes(x = numRanks, ))