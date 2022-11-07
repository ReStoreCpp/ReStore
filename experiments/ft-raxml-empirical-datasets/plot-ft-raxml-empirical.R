output_dir <- "~/projects/ReStore/experiments/ft-raxml-empirical-datasets"
data_dir <- paste(output_dir, "data", sep = "/")
setwd(output_dir)

source("../common.R")

save_plot <- ggsave_factory(output_dir)

# STYLE <- "print"
STYLE <- "slides"

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

dataset_labels <- c()
if (STYLE == "print") {
    dataset_labels <- c(
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
    )
} else {
    dataset_labels <- c(
      "aa_rokasA1" = "96 PEs\n0.10 MiB/PE",
      "aa_rokasA4" = "4000 PEs\n0.03 MiB/PE",
      "aa_rokasA8" = "96 PEs\n0.48 MiB/PE",
      "dna_PeteD8" = "240 PEs\n2.08 MiB/PE",
      "dna_rokasD1" = "480 PEs\n0.10 MiB/PE",
      "dna_rokasD4" = "192 PEs\n0.06 MiB/PE",
      "dna_rokasD7" = "4000 PEs\n0.18 MiB/PE"
    )
}

benchmark_colors <-c(
    "SaveToReStore" = "#ff7f0e",
    "LoadMSAFromReStore" = "#d62728",
    "LoadMSAFromRBACached" = "#1f77b4",
    "LoadMSAFromRBAUncached" = "#2ca02c"
)
if (STYLE == "slides") {
    benchmark_colors <- benchmark_colors %>% remove_elem("LoadMSAFromRBACached")
}

if (STYLE == "slides") {
    across_ranks_stats_long <- across_ranks_stats_long %>%
        filter(timer != "LoadMSAFromRBACached")
}

benchmark_labels <- c(
    "SaveToReStore" = "submit to ReStore",
    "LoadMSAFromReStore" = "load from ReStore",
    "LoadMSAFromRBACached" = "load from binary file (cached, partial)",
    "LoadMSAFromRBAUncached" = "load from binary file (uncached, partial)"
)
if (STYLE == "slides") {
    benchmark_labels["LoadMSAFromRBACached"] = "load from disk"
}

ggplot() +
  geom_point(
    data = across_ranks_stats_long,
    aes(
        y = msMax,
        x = dataset,
        color = timer,
        shape = timer
    ),
    position = position_dodge(width = 0.3),
    size = 3
  ) +
  theme_husky(
    style = STYLE,
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_blank(),
  ) +
  ylab("time [ms]") +
  scale_y_log10(
    breaks = log_axis_breaks,
    minor_breaks = log_axis_minor_breaks
  ) +
  scale_x_discrete(
    labels = dataset_labels,
  ) +
  scale_color_shape_manual(
    color_values = benchmark_colors,
    labels = benchmark_labels
  ) +
  annotation_logticks(sides = "l")

if (STYLE == "print") {
    save_plot(
        "ft-raxml-empirical", style = STYLE,
        width = 85, height = 56.5
    )
} else if (STYLE == "slides") {
    save_plot(
        "ft-raxml-empirical", style = STYLE,
        width = 1.3 * 140, height = 1.3 * 147
    )
}
