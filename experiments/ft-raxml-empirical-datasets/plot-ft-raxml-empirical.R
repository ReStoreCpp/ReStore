output_dir <- "~/projects/ReStore/experiments/ft-raxml-empirical-datasets"
data_dir <- paste(output_dir, "data", sep = '/') 
setwd(output_dir)

source("../common.R")

# Dataset to verbose label
dataset2Label <- function(s) {
  datatype <- str_extract(s, "^(dna|aa)")
  dataset <- str_extract(s, "(?<=_)[:alnum:]+(?=_)")
  ranksPerNode <- str_extract(s, "(?<=_)[:digit:]+(?=@)")
  nNodes <- str_extract(s, "(?<=@)[:digit:]+$")
  description <- paste(toupper(datatype), " dataset ", dataset, "\n", nNodes, " nodes with ", ranksPerNode, " ranks each", sep = "")
  return(description)
}

### Data loading ###
# load profiling data from single csv file and add `dataset` column
read_csv_file <- function(filename) {
  read_csv(
    filename,
    col_types = cols(
        .default = col_double(),
        processor = col_character()
    )) %>%
    mutate(
        dataset = str_extract(filename, "(dna|aa)_.+(A|D)[:digit:]"),
        rep = as.integer(str_extract(filename, "(?<=/rep)[:digit:]"))
    )
}

# Load all profiling data from multiple runs and aggregate it into `data`
per_rank_stats <-
  list.files(
    path = data_dir,
    pattern = "*.profiler.csv", 
    full.names = TRUE
  ) %>% 
  map_df(~read_csv_file(.)) %>%
  as_tibble() %>%
  mutate(dataset = factor(dataset))


# Aggregate the per rank statistics into per measurement statistics
## First, convert the measurements into a long format
per_rank_stats_long_time <- per_rank_stats %>%
  select(- starts_with("num")) %>%
  gather(
    key = timer,
    value = nsSum,
    starts_with("nsSum")
  ) %>%
  mutate (
    timer = str_replace(timer, "nsSum", "")
  )

per_rank_stats_long_num <- per_rank_stats %>%
  select(- starts_with("nsSum")) %>%
  gather(
    key = timer,
    value = num,
    starts_with("num")
  ) %>%
  mutate(
    timer = str_replace(timer, "num", "")
  )

per_rank_stats_long <- inner_join(
  by = c("rank", "processor", "dataset", "timer", "rep"),
  per_rank_stats_long_time,
  per_rank_stats_long_num
) %>%
  rename(
    nsRankwiseTotalTime = nsSum,
    nCallsOnThisRank = num
  ) %>%
  mutate(
    nsRankwiseTimePerCall = nsRankwiseTotalTime / nCallsOnThisRank
  ) %>%
  filter(
      timer %in% c("LoadMSAFromRBACached","LoadMSAFromRBAUncached",
                   "LoadMSAFromReStore", "SaveToReStore")
  )

# Did every one of the events happen exactly once per rank?
per_rank_stats_long %>% pull(nCallsOnThisRank) %>% unique()

## Next, aggregate the per rank measurements over the ranks into per dataset summary statistics.
across_ranks_stats_long <- per_rank_stats_long %>%
  group_by(dataset, timer, rep) %>%
  summarise(
    .groups = "keep",
    nsAcrossRanksTimePerCallAvg = mean(nsRankwiseTimePerCall),
    nsAcrossRanksTimePerCallMax = max(nsRankwiseTimePerCall),
    nsAcrossRanksTimePerCallSD = sd(nsRankwiseTimePerCall),
  ) %>%
  mutate(
    msAcrossRanksTimePerCallAvg = ns2ms(nsAcrossRanksTimePerCallAvg),
    msAcrossRanksTimePerCallMax = ns2ms(nsAcrossRanksTimePerCallMax),
    msAcrossRanksTimePerCallSD = ns2ms(nsAcrossRanksTimePerCallSD),
  )

# Plot comparing the time a rank requires to load a part of the MSA {the first time, if someone else already loaded it}
across_ranks_stats_long <- across_ranks_stats_long %>%
  select(- starts_with("ns")) %>%
  rename(
      msAvg = msAcrossRanksTimePerCallAvg,
      msSD = msAcrossRanksTimePerCallSD,
      msMax = msAcrossRanksTimePerCallMax
  )
  
across_ranks_stats_long <- across_ranks_stats_long %>%
    mutate(
        dataset = factor(dataset, levels = c("dna_rokasD4", "dna_rokasD1", "aa_rokasA1", "aa_rokasA8", "dna_PeteD8", "aa_rokasA4", "dna_rokasD7"))
    )

ggplot() +
  geom_point(
    data = across_ranks_stats_long,
    aes(y = msMax, x = dataset, color = timer),
    position = position_dodge(width = 0.3),
    size = 1
  ) +
  theme_husky(
    legend.position = "none",
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.45),
    axis.title.x = element_blank(),
    # legend.position = c(0, 1),
    # legend.justification = c("left", "top"),
    # legend.box.just = "left",
    # legend.title = element_blank(),
  ) +
  ylab("time [ms]") +
  scale_y_log10(breaks = log_axis_breaks, minor_breaks = log_axis_minor_breaks) +
  scale_x_discrete(
    labels = c(
      #"aa_rokasA1" = "AA NagyA1 @ 96 ranks\n172k sites · 60 taxa\n10 MiB\n 0.10 MiB/rank",
      #"aa_rokasA4" = "AA ChenA4 @ 4000 ranks\n1806k sites · 58 taxa\n100 MiB\n0.03 MiB/rank",
      #"aa_rokasA8" = "AA YangA8 @ 96 ranks\n505k sites · 95 taxa\n46 MiB\n0.48 MiB/rank",
      #"dna_PeteD8" = "DNA PeteD8 @ 240 ranks\n3011k sites · 174 taxa\n500 MiB\n2.08 MiB/rank",
      #"dna_rokasD1" = "DNA SongD1 @ 480 ranks\n1339k sites · 37 taxa\n48 MiB\n0.10 MiB/rank",
      #"dna_rokasD4" = "DNA XiD4 @ 192 ranks\n240k sites · 46 taxa\n11 MiB\n0.06 MiB/rank",
      #"dna_rokasD7" = "DNA TarvD7 @ 4000 ranks\n21411k sites · 36 taxa\n736 MiB\n0.18 MiB/rank"
      "aa_rokasA1" = "NagyA1 · 96 PEs\n0.10 MiB/PE",
      "aa_rokasA4" = "ChenA4 · 4000 PEs\n0.03 MiB/PE",
      "aa_rokasA8" = "YangA8 · 96 PEs\n0.48 MiB/PE",
      "dna_PeteD8" = "PeteD8 · 240 PEs\n2.08 MiB/PE",
      "dna_rokasD1" = "SongD1 · 480 PEs\n0.10 MiB/PE",
      "dna_rokasD4" = "XiD4 · 192 PEs\n0.06 MiB/PE",
      "dna_rokasD7" = "TarvD7 · 4000 PEs\n0.18 MiB/PE"
    ),
    #limits = (across_ranks_stats_long %>% filter(timer == "LoadAssignmentDataFirstLoad") %>% arrange(msAvg))$dataset
  ) +
  scale_color_manual(
      labels = c(
        "SaveToReStore" = "submit to ReStore",
        "LoadMSAFromReStore" = "load from ReStore",
        "LoadMSAFromRBACached" = "load from binary file (cached, partial)",
        "LoadMSAFromRBAUncached" = "load from binary file (uncached, partial)"
      ),
      values = c("#1f77b4", "#2ca02c", "#d62728", "#ff7f0e")
  ) + 
  annotation_logticks(sides = "l")
ggsave(paste(output_dir, "ft-raxml-empirical.pdf", sep = '/'), width = 60, height = 52, units = "mm")
