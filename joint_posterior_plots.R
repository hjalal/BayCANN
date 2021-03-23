#### Visualization of pairwise joint distribitions and correlations ####
library(reshape2)
library(GGally)
# Read the unscaled posterior samples
Xq_unscaled <- read.csv(file = "output/calibrated_posteriors.csv")[, -1]
df_post <- data.frame(Xq_unscaled)
colnames(df_post) <- x_names

df_post_long <- reshape2::melt(df_post,
                               variable.name = "Parameter")
df_post_long$Parameter <- factor(df_post_long$Parameter, 
                                 levels = levels(df_post_long$Parameter),
                                 ordered = TRUE, 
                                 labels = c(expression(l),
                                            expression(gamma),
                                            expression(lambda[2]),
                                            expression(lambda[3]),
                                            expression(lambda[4]),
                                            expression(lambda[5]),
                                            expression(lambda[6]),
                                            expression(p[adeno]),
                                            expression(p[small])))

gg_calib_post_pair_corr <- GGally::ggpairs(df_post,
                                           upper = list(continuous = wrap("cor",
                                                                          color = "black",
                                                                          size = 5)),
                                           diag = list(continuous = wrap("barDiag",
                                                                         alpha = 0.8)),
                                           lower = list(continuous = wrap("points", 
                                                                          alpha = 0.3,
                                                                          size = 0.5)),
                                           columnLabels = c("l",
                                                            "gamma",
                                                            "lambda[2]",
                                                            "lambda[3]",
                                                            "lambda[4]",
                                                            "lambda[5]",
                                                            "lambda[6]",
                                                            "p[adeno]",
                                                            "p[small]"),
                                           labeller = "label_parsed") +
  theme_bw(base_size = 18) +
  theme(axis.title.x = element_blank(),
        axis.text.x  = element_text(size=6),
        axis.title.y = element_blank(),
        axis.text.y  = element_blank(),
        axis.ticks.y = element_blank(),
        strip.background = element_rect(fill = "white",
                                        color = "white"),
        strip.text = element_text(hjust = 0))
gg_calib_post_pair_corr

ggsave(filename = "output/posterior_distribution_pairwise_corr.pdf",
       gg_calib_post_pair_corr,
       width = 12, height = 8)
ggsave(filename = "output/posterior_distribution_pairwise_corr.jpeg",
       gg_calib_post_pair_corr,
       width = 12, height = 8)
ggsave(filename = "output/posterior_distribution_pairwise_corr.png",
       width = 12, height = 8)

