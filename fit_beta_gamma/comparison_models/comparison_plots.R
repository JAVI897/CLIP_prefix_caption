setwd('C:/Users/Usuario/Desktop/CLIP_prefix_caption/fit_beta_gamma/comparison_models')
library(ggstatsplot)


df <- read.csv('comparison_df.csv')
group_by <- aggregate(df$REFCLIP_SCORE, by = list(df$Group), FUN = mean)
group_by <- group_by[order(-group_by$x),]
group_by <- group_by[1:4,]$Group.1

df_best <- df[which(df$Group %in% group_by),]

ggbetweenstats(
  data = df_best,
  x = Group,
  y = REFCLIP_SCORE,
  type = "parametric",
  xlab = " ",
  ylab = "RefClipScore",
  pairwise.comparisons = TRUE,
  pairwise.display = 'all',
  p.adjust.method = "fdr",
  outlier.tagging = FALSE,
)

specific <- c('Beta_0.0_N_10', 'Beta_0.05_N_10', 'Beta_0.1_N_10', 'Beta_0.15_N_10', 'Beta_0.2_N_10', 
              'Beta_0.25_N_10', 'Beta_0.3_N_10', 'Beta_0.35_N_10', 'Beta_0.4_N_10',
              'Beta_0.45_N_10', 'Beta_0.5_N_10', 'Beta_0.55_N_10', 'Beta_0.6_N_10',
              'Beta_0.65_N_10', 'Beta_0.7_N_10', 'Beta_0.75_N_10',  'Beta_0.8_N_10',
              'Beta_0.85_N_10', 'Beta_0.9_N_10', 'Beta_0.95_N_10', 'Beta_1_N_10')
df_specific <- df[which(df$Group %in% specific),]

ggbetweenstats(
  data = df_specific,
  x = Group,
  y = REFCLIP_SCORE,
  type = "parametric",
  xlab = " ",
  ylab = "RefClipScore",
  pairwise.comparisons = TRUE,
  p.adjust.method = "fdr",
  outlier.tagging = FALSE,
)