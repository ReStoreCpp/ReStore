data_dir <- "~/projects/ReStore/experiments/failures-until-data-loss"
output_dir <- data_dir
setwd(data_dir)

source("../common.R")

STYLE <- "slides"
#STYLE <- "print"
# STYLE <- "interactive"

# Load the empirical data.
empirical_data <- read_csv(
    "failure-probability.csv",
    col_types = cols(
        numberOfPEs = col_integer(),
        replicationLevel = col_integer(),
        roundsUntilIrrecoverableDataLoss = col_integer()
    )
) %>%
    group_by(numberOfPEs, replicationLevel) %>%
    summarize(
        roundsUntilIrrecoverableDataLoss_mean = mean(roundsUntilIrrecoverableDataLoss),
        roundsUntilIrrecoverableDataLoss_median = median(roundsUntilIrrecoverableDataLoss),
        roundsUntilIrrecoverableDataLoss_sd = sd(roundsUntilIrrecoverableDataLoss),
        roundsUntilIrrecoverableDataLoss_q10 = quantile(roundsUntilIrrecoverableDataLoss, probs = 0.9),
        roundsUntilIrrecoverableDataLoss_q90 = quantile(roundsUntilIrrecoverableDataLoss, probs = 0.1),
        .groups = "keep"
    )

# Load the data yielded by theory.
theoretical_data <- read_csv(
    "theoretical-failure-probability.csv",
    col_types = cols(
        numberOfPEs = col_integer(),
        replicationLevel = col_integer(),
        roundsUntilIrrecoverableDataLoss = col_double()
    )
)

# Compute approximate data
approximate_rounds_until_idl <- function(numberOfPEs, replicationLevel) {
    (replicationLevel**(1 / replicationLevel)) / (numberOfPEs**(1 / replicationLevel)) * numberOfPEs
}
approximate_data <- empirical_data %>%
    select(numberOfPEs, replicationLevel) %>%
    filter(replicationLevel == 4) %>%
    mutate(
        approximateRoundsUntilIrrecoverableDataLoss = approximate_rounds_until_idl(numberOfPEs, replicationLevel)
    )

# Display number of PEs as 2^n
# add_log2_numberOfPEs_cols <- function(data) {
#     data <- data %>% mutate(
#         log2_numberOfPEs = log2(numberOfPEs),
#         log2_numberOfPEs_HR = lapply(
#             sprintf('2^{%d}', log2_numberOfPEs),
#             TeX
#     ))
# }
# empirical_data <- add_log2_numberOfPEs_cols(empirical_data)
# theoretical_data <- add_log2_numberOfPEs_cols(theoretical_data)
# approximate_data <- add_log2_numberOfPEs_cols(approximate_data)

# theoretical_data <- theoretical_data %>%
#     mutate(
#         log2_numberOfPEs = log2(numberOfPEs),
#         log2_numberOfPEs_HR = lapply(
#             sprintf('2^{%d}', log2_numberOfPEs),
#             TeX
#     ))

# x-breaks
x_breaks <- empirical_data %>%
    pull(numberOfPEs) %>%
    unique()

# Theory (vs empirical)
curves <- c("simulated", "exact", "approximate")
curve_colors <- setNames(c("#7570b3", "#a6761d", "#e7298a"), curves)
curve_labels <- setNames(c(
    TeX(r'($r=4$ simulated)'),
    TeX(r'($r=4$ exact)'),
    TeX(r'($r=4$ approximate $O(p^{-{1/r}})$)')
), curves)

ggplot() +
    geom_line_with_points_and_errorbars(
        data = empirical_data %>%
            filter(replicationLevel == 4) %>%
            mutate(replicationLevel = "simulated"),
        mapping = aes(
            x = numberOfPEs,
            y = roundsUntilIrrecoverableDataLoss_mean / numberOfPEs * 100,
            ymin = roundsUntilIrrecoverableDataLoss_q10 / numberOfPEs * 100,
            ymax = roundsUntilIrrecoverableDataLoss_q90 / numberOfPEs * 100,
            color = as.factor(replicationLevel),
            group = replicationLevel
        ),
        line_width = 1
    ) +
    geom_line(
        data = theoretical_data %>% mutate(replicationLevel = "exact"),
        aes(
            x = numberOfPEs,
            y = roundsUntilIrrecoverableDataLoss / numberOfPEs * 100,
            color = as.factor(replicationLevel),
            group = replicationLevel
        ),
        size = 1
    ) +
    geom_line(
        data = approximate_data %>% mutate(replicationLevel = "approximate"),
        aes(
            x = numberOfPEs,
            y = approximateRoundsUntilIrrecoverableDataLoss / numberOfPEs * 100,
            color = as.factor(replicationLevel),
            group = replicationLevel
        ),
        size = 1
    ) +
    scale_color_manual(values = curve_colors, labels = curve_labels) +
    scale_y_log_with_ticks_and_lines(scale_accuracy = 0.01, limits = c(0.006, 100)) +
    scale_x_log10(breaks = x_breaks, labels = label_log2(tex = FALSE)) +
    theme_husky(
        style = STYLE,
        legend.position = c(0.025, 0.0),
        legend.justification = c("left", "bottom"),
        legend.text.align = 0, # left
        axis.text.x = element_text(angle = 45, hjust = 1),
        #axis.title.y = element_blank(),
        legend.title = if (STYLE == "print") element_text() else element_blank(),
    ) +
    labs(
        x = "#PEs",
        y = "% of PEs that failed\nuntil irrecoverable data loss",
        color = "redundant copies"
    )

if (STYLE == "slides") {
    ggsave(
        paste(output_dir, "failures-until-irrecoverable-data-loss-theory-slides.svg", sep = "/"),
        width = 5 * 60, height = 3 * 53, units = "mm"
    )
    to_jpeg("failures-until-irrecoverable-data-loss-theory-slides.svg")
} else if (STYLE == "print") {
    # ggsave(
    #     paste(output_dir, "failures-until-irrecoverable-data-loss-theory-print.pdf", sep = "/"),
    #     width = 60, height = 53, units = "mm"
    # )
    ggsave(
        paste(output_dir, "failures-until-irrecoverable-data-loss-theory-print.pdf", sep = "/"),
        width = 85, height = 80, units = "mm"
    )
}

# Empirical
ggplot() +
    geom_line_with_points_and_errorbars(
        data = empirical_data,
        mapping = aes(
            x = numberOfPEs,
            y = roundsUntilIrrecoverableDataLoss_mean / numberOfPEs * 100,
            ymin = roundsUntilIrrecoverableDataLoss_q10 / numberOfPEs * 100,
            ymax = roundsUntilIrrecoverableDataLoss_q90 / numberOfPEs * 100,
            color = as.factor(replicationLevel),
            shape = as.factor(replicationLevel),
            group = replicationLevel
        ),
        line_width = 1,
        point_size = 3
    ) +
    scale_shape_and_color(
        name = "redundant copies",
        labels = function(replicationLevel) {
            TeX(sprintf('$r = %s$', replicationLevel))
    }) +
    scale_y_log_with_ticks_and_lines(scale_accuracy = 0.01) +
    #scale_x_log10(breaks = x_breaks, label = comma_format(accuracy = 1)) +
    scale_x_log10(breaks = x_breaks, label = label_log2(tex = FALSE)) +
    theme_husky(
        style = STYLE,
        legend.position = c(0.025, 0),
        legend.justification = c("left", "bottom"),
        axis.text.x = element_text(angle = 45, hjust = 1),
    ) +
    labs(
        x = "#PEs",
        y = "% of PEs that failed\nuntil irrecoverable data loss",
        color = "redundant copies"
    )

if (STYLE == "slides") {
    ggsave(
        paste(output_dir, "failures-until-irrecoverable-data-loss-slides.svg", sep = "/"),
        width = 5 * 60, height = 3 * 53, units = "mm"
    )
    to_jpeg("failures-until-irrecoverable-data-loss-slides.svg")
} else if (STYLE == "print") {
    # ggsave(
    #     paste(output_dir, "failures-until-irrecoverable-data-loss-print.pdf", sep = "/"),
    #     width = 60, height = 53, units = "mm"
    # )
    ggsave(
        paste(output_dir, "failures-until-irrecoverable-data-loss-print.pdf", sep = "/"),
        width = 85, height = 80, units = "mm"
    )
}
