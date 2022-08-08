input_dir <- "~/projects/ReStore/experiments/ft-raxml-simulated-datasets" 
output_dir <- input_dir
csv_file <- paste(input_dir, "data.csv", sep = '/') 

source("../common.R")

NUM_RANKS_PER_NODE <- 48

### Data loading ###
data <- read_csv(
  csv_file,
  col_types = cols(
    numberOfNodes = col_integer(),
    seed = col_integer(),
    repetition = col_integer(),
    rank = col_integer(),
    processor = col_character(),
    numLoadMSAFromRBACached = col_integer(),
    nsSumLoadMSAFromRBACached = col_double(),
    numLoadMSAFromRBAUncached = col_integer(),
    nsSumLoadMSAFromRBAUncached = col_double(),
    numLoadMSAFromReStore = col_integer(),
    nsSumLoadMSAFromReStore = col_double(),
    numLoadMetadataFromRBA = col_integer(),
    nsSumLoadMetadataFromRBA = col_double(),
    numSaveToReStore = col_integer(),
    nsSumSaveToReStore = col_double(),
    numwork = col_integer(),
    nsSumwork = col_double()
  )) %>%
  select(-numwork, -nsSumwork) %>%
  mutate(
    numberOfRanks = numberOfNodes * NUM_RANKS_PER_NODE,
    `pm_loadMSAFromRBACached` = nsSumLoadMSAFromRBACached / numLoadMSAFromRBACached, 
    `pm_loadMSAFromRBAUncached` = nsSumLoadMSAFromRBAUncached / numLoadMSAFromRBAUncached, 
    `pm_loadMSAFromReStore` = nsSumLoadMSAFromReStore / numLoadMSAFromReStore, 
    `pm_saveMSAToRestore` = nsSumSaveToReStore / numSaveToReStore, 
    `pm_loadMetadataFromRBA` = nsSumLoadMetadataFromRBA / numLoadMetadataFromRBA
  ) %>%
  select(
    -numLoadMSAFromRBACached, 
    -numLoadMSAFromRBAUncached, 
    -numLoadMSAFromReStore, 
    -numSaveToReStore, 
    -numLoadMetadataFromRBA,
    -starts_with("nsSum")
  ) %>%
  pivot_longer(
    cols = starts_with("pm_"),
    names_prefix = "pm_",
    names_to = "timer",
    values_to = "time_ns"
  ) %>%
  group_by(numberOfRanks, numberOfNodes, repetition, seed, timer) %>%
  summarize(
    time_mean_ms = ns2ms(mean(time_ns)),
    time_sd_ms = ns2ms(sd(time_ns)),
    .groups = "keep"
  )

### Plot data ###
data %>% filter(timer != "loadMetadataFromRBA") %>%
ggplot(
  aes(
    y = time_mean_ms,
    ymin = time_mean_ms - time_sd_ms,
    ymax = time_mean_ms + time_sd_ms,
    x = as.factor(numberOfRanks),
    color = timer,
    group = timer
)) +
  geom_point(
    position = position_dodge(width = 0.3),
    size = 1
  ) +
  theme_husky(
    #legend.position = c(0, 0),
    legend.position = "bottom",
    legend.margin = margin(),
    legend.box.margin = margin(),
    legend.box.spacing = unit(1, "mm"),
    #legend.justification = c("left", "bottom"),
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    #axis.title.y = element_blank()
  ) +
  ylab("time [ms]") +
  xlab("number of PEs") +
  annotation_logticks(base = 10, sides = "l") +
  scale_y_log10(breaks = log_axis_breaks, minor_breaks = log_axis_minor_breaks) +
  scale_color_manual(
      labels = c(
        "saveMSAToRestore" = "submit to ReStore",
        "loadMSAFromReStore" = "load from ReStore (all data)",
        "loadMSAFromRBACached" = "load cached binary file",
        "loadMSAFromRBAUncached" = "load uncached binary file"
      ),
      values = c("#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"),
  ) +
  guides(
    color = guide_legend(
      nrow = 2,
      byrow = TRUE,
      keyheight  = unit(0, "mm"),
  ))
# ggsave(paste(output_dir, "ft-raxml-simulated.pdf", sep = '/'), width = 58.5, height = 55, units = "mm")
ggsave(paste(output_dir, "ft-raxml-simulated.pdf", sep = '/'), width = 85, height = 55, units = "mm")

