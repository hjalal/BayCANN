# load libraries ========
library(keras)
library(rstan)
library(reshape2)
library(tidyverse)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# load baycann functions =======
source("baycann_functions.R")

# ==================
# Input parameters
n_iter <- 10000
n_hidden_nodes <- 100
n_hidden_layers <- 2 
n_epochs <- 10000
verbose <- 0
n_batch_size <- 2000
n_chains <- 4
set.seed(1234)
# load the training and test datasets for the simulations =========
load("data/05_DoE-unif-crc-nhm-det_test_n_train.rData")

prepared_data <- prepare_data(xtrain = samp_i_unif_train,
                              ytrain = out_i_det_unif_train,
                              xtest  = samp_i_unif_test,
                              ytest  = out_i_det_unif_test)

list2env(prepared_data, envir = .GlobalEnv)


# load the targets and their se ==========
load("data/03_targets-crc-nhm_100-runs.RData")
targets_combinedmean100 <- rbind(df.true.adeno.100runs, df.true.crc.100runs)
targets_combinedse100 <- rbind(df.true.adeno.se.100runs, df.true.crc.se.100runs)
denom_seq <- c(rep(100,20), rep(100000,16)) # adenomas are per 100, and CRC per 100K
true_targets_mean <- rowMeans(targets_combinedmean100[,4:103]) / denom_seq #pooled average of 100 sims
true_targets_se <- sqrt(rowMeans(targets_combinedse100[,4:103]^2)) / denom_seq #pooled standard error of 100 sims


# get the true data
x_true_data <- read.csv(file="data/01_true-params.csv", header = T, sep = ",")
x_true_unscaled <- x_true_data$x

# =====================================
# Scale the targets and their SE
y_targets <- 2 * (true_targets_mean - ymins) / (ymaxs - ymins) - 1
y_targets_se <- 2 * (true_targets_se) / (ymaxs - ymins)

y_targets <- t(as.matrix(y_targets))
y_targets_se <- t(as.matrix(y_targets_se))


# ============== TensorFlow Keras ANN Section ========================

model <- keras_model_sequential() 
mdl_string <- paste("model %>% layer_dense(units = n_hidden_nodes, activation = 'tanh', input_shape = n_inputs) %>%", 
                   paste(rep(x = "layer_dense(units = n_hidden_nodes, activation = 'tanh') %>%", 
                   n_hidden_layers), collapse = " "), 
                   "layer_dense(units = n_outputs)")
eval(parse(text = mdl_string))
summary(model)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)
keras.time <- proc.time()
history <- model %>% fit(
  xtrain_scaled, ytrain_scaled,
  epochs = n_epochs, 
  batch_size = n_batch_size, 
  validation_data = list(xtest_scaled, ytest_scaled), 
  verbose = verbose
)
proc.time() - keras.time #keras ann fitting time

png(filename='output/ann_convergence.png')
plot(history)
dev.off()

weights <- get_weights(model) #get ANN weights
pred <- model %>% predict(xtest_scaled)
ytest_scaled_pred <- data.frame(pred)
colnames(ytest_scaled_pred) <- y_names
head(ytest_scaled_pred)

ann_valid <- rbind(data.frame(sim = 1:n_test, ytest_scaled, type = "model"), 
                   data.frame(sim = 1:n_test, ytest_scaled_pred, type = "pred"))
ann_valid_transpose <- ann_valid %>% 
  pivot_longer(cols = -c(sim, type)) %>% 
  pivot_wider(id_cols = c(sim, name), names_from = type, values_from = value)
ggplot(data = ann_valid_transpose, aes(x = model, y = pred)) + 
  geom_point(alpha = 0.5, color = "tomato") + 
  facet_wrap(~name, ncol = 6) + 
  xlab("Model outputs (scaled)") + 
  ylab("ANN predictions (scaled)") + 
  coord_equal() + 
  theme_bw()

ggsave(filename = "figs/fig4_ann_validation_vs_observed.pdf", width = 8.5, height = 11)
ggsave(filename = "figs/fig4_ann_validation_vs_observed.png", width = 8.5, height = 11)
ggsave(filename = "figs/fig4_ann_validation_vs_observed.jpg", width = 8.5, height = 11)

# ======== STAN SECTION ====================
# pass the weights and biases to Stan for Bayesian calibration
n_layers <- length(weights)
weight_first <- weights[[1]]
beta_first <- 1 %*% weights[[2]] 
weight_last <- weights[[n_layers-1]]
beta_last <- 1 %*% weights[[n_layers]]
weight_middle <- array(0, c(n_hidden_layers, n_hidden_nodes, n_hidden_nodes))
beta_middle <- array(0, c(n_hidden_layers, 1, n_hidden_nodes))
for (l in 1:n_hidden_layers){
  weight_middle[l,,] <- weights[[l*2+1]]
  beta_middle[l,,] <- weights[[l*2+2]]
}
stan.dat=list(
  num_hidden_nodes = n_hidden_nodes, 
  num_hidden_layers= n_hidden_layers, 
  num_inputs=n_inputs,
  num_outputs=n_outputs,
  num_targets=1,
  y_targets = y_targets,
  y_targets_se = y_targets_se,
  beta_first = beta_first,
  beta_middle = beta_middle,
  beta_last = beta_last, 
  weight_first = weight_first, 
  weight_middle = weight_middle, 
  weight_last = weight_last)

stan.time <- proc.time()
m <- stan(file = "post_multi_perceptron.stan", 
          data = stan.dat, 
          iter = n_iter, 
          chains = n_chains, 
          pars = c("Xq"), 
          seed = 12345) #for reproducibility. R's set.seed() will not work for stan
proc.time() - stan.time # stan sampling time
summary(m)

params <- rstan::extract(m)
lp <- params$lp__
Xq <- params$Xq
Xq_df = as.data.frame(Xq)

# Scale the posteriors
Xq_unscaled <- unscale_data(Xq_df, vec.mins = xmins, vec.maxs = xmaxs)
Xq_lp <- cbind(Xq_unscaled, lp) 
# Save the unscaled posterior samples
write.csv(Xq_lp, 
          file = "output/calibrated_posteriors.csv", 
          row.names = FALSE)


#### Visualization of priors and posteriors ####
# Plot histogram for individual posteriors and compare to prior
# Read the unscaled posterior samples
Xq_lp <- read.csv(file = "output/calibrated_posteriors.csv")
n_col <- ncol(Xq_lp)
lp <- Xq_lp[, n_col]
Xq_unscaled <- Xq_lp[, -n_col]
priordf <- data.frame(samp_i_unif_train)
postdf <- data.frame(Xq_unscaled)
colnames(postdf) <- x_names
map_baycann <- Xq_unscaled[which.max(lp), ]

# plotting BayCANN's posterior distributions ========
priordf$type <- 'prior'
postdf$type <- paste('chain ', sort(rep(1:4, n_iter/2)))
priorpost <- rbind(priordf, postdf)
melt_df <- melt(priorpost)
line_df <- data.frame(variable = x_names, intercept = cbind(x_true_unscaled))
map_baycann_df <- data.frame(variable = x_names, intercept = t(map_baycann))
colnames(line_df) <- c("variable","intercept")
colnames(map_baycann_df) <- c("variable","intercept")
ggplot(melt_df, aes(value, fill=type, colour = type)) + 
  geom_density(alpha = 0.1) + 
  facet_wrap(~variable, scales="free") + 
  geom_vline(data=line_df, aes(xintercept=intercept)) + 
  geom_vline(data=map_baycann_df, aes(xintercept=intercept), color = "red")
ggsave("output/convergence.png")

# Plot histogram for combined posterior and compare to the truth and the prior
melt_df_comb = melt_df
melt_df_comb[melt_df_comb == "chain  1" | melt_df_comb == "chain  2" |
               melt_df_comb == "chain  3" | melt_df_comb == "chain  4" ] <- "post"
ggplot(melt_df_comb, aes(value, fill=type, colour = type)) + 
  geom_density(alpha = 0.1) + 
  facet_wrap(~variable, scales="free") + 
  geom_vline(data=line_df, aes(xintercept=intercept)) + 
  geom_vline(data=map_baycann_df, aes(xintercept=intercept), color = "red")

ggsave("output/prior_post_truth.png")

#### Visualization of pairwise joint distributions and correlations ####
library(GGally)
# Read the unscaled posterior samples
#Xq_unscaled <- read.csv(file = "output/calibrated_posteriors.csv")[, -n_col]
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

ggsave(filename = "figs/fig7_posterior_distribution_pairwise_corr.pdf",
       gg_calib_post_pair_corr,
       width = 12, height = 8)
ggsave(filename = "figs/fig7_posterior_distribution_pairwise_corr.jpeg",
       gg_calib_post_pair_corr,
       width = 12, height = 8)
ggsave(filename = "figs/fig7_posterior_distribution_pairwise_corr.png",
       width = 12, height = 8)

#### ANN vs. IMIS ####
### Load IMIS posterior
load(file="data/04_crc-nhm-det_posterior-IMIS-unif.RData") 
df_post_imis <- post_imis_unif

### Load ANN posterior
df_post_ann <- read.csv(file = "output/calibrated_posteriors.csv")[, -n_col]
colnames(df_post_ann) <- x_names

n_samp <- 1000
df_samp_prior <- melt(cbind(Distribution = "Prior", 
                            as.data.frame(samp_i_unif_train[1:1000, ])), 
                      variable.name = "Parameter")
df_samp_post_imis  <- melt(cbind(Distribution = "Posterior IMIS", 
                                 as.data.frame(df_post_imis[1:1000, ])), 
                           variable.name = "Parameter")
# (v.params0 - colMeans(df_post_imis))^2
df_samp_post_ann   <- melt(cbind(Distribution = "Posterior BayCANN", 
                                 as.data.frame(df_post_ann[1:1000, ])), 
                           variable.name = "Parameter")
df_samp_prior_post <- rbind(df_samp_prior, 
                            df_samp_post_ann, 
                            df_samp_post_imis)
df_samp_prior_post$Distribution <- ordered(df_samp_prior_post$Distribution, 
                                           levels = c("Prior", 
                                                      "Posterior IMIS", 
                                                      "Posterior BayCANN"))
v_names_params_greek <- c(expression(l),
                          expression(g),  # expression(gamma)
                          expression(lambda[2]),
                          expression(lambda[3]),
                          expression(lambda[4]),
                          expression(lambda[5]),
                          expression(lambda[6]),
                          expression(p[adeno]),
                          expression(p[small]))
df_samp_prior_post$Parameter <- factor(df_samp_prior_post$Parameter,
                                       levels = levels(df_samp_prior_post$Parameter),
                                       ordered = TRUE,
                                       labels = v_names_params_greek)

df_maps_n_true_params <- data.frame(Type = ordered(rep(c("True parameter", 
                                                     "IMIS MAP", 
                                                     "BayCANN MAP"), each = 9), 
                                             levels = c("True parameter", 
                                                        "IMIS MAP", 
                                                        "BayCANN MAP")), 
                                    Parameter = as.character(v_names_params_greek),
                                    value = c(x_true_data$x, 
                                              v_calib_post_map, 
                                              t(map_baycann)))
df_maps_n_true_params
### Plot priors and ANN and IMIS posteriors
gg_ann_vs_imis <- ggplot(df_samp_prior_post, 
           aes(x = value, y = ..density.., fill = Distribution)) +
    facet_wrap(~Parameter, scales = "free", 
               ncol = 3,
               labeller = label_parsed) +
    # geom_vline(data = data.frame(Parameter = as.character(v_names_params_greek),
    #                              value = x_true_data$x, row.names = v_names_params_greek), 
    #            aes(xintercept = value)) +
    # geom_vline(data = data.frame(Parameter = as.character(v_names_params_greek),
    #              value = c(t(map_baycann)), row.names = v_names_params_greek), 
    #   aes(xintercept = value), color = "tomato") +
    geom_vline(data = df_maps_n_true_params,
               aes(xintercept = value, linetype = Type, color = Type)) +
    scale_x_continuous(breaks = dampack::number_ticks(5)) +
    scale_color_manual("", values = c("black", "navy blue", "tomato")) +
    geom_density(alpha=0.5) +
    theme_bw(base_size = 16) +
    guides(fill = guide_legend(title = "", order = 1),
           linetype = guide_legend(title = "", order = 2),
           color = guide_legend(title = "", order = 2)) +
    theme(legend.position = "bottom",
          legend.box = "vertical",
          legend.margin=margin(),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          strip.background = element_rect(fill = "white",
                                          color = "white"),
          strip.text = element_text(hjust = 0))
gg_ann_vs_imis
ggsave(gg_ann_vs_imis, 
       filename = "figs/fig5_ANN-vs-IMIS-posterior.pdf", 
       width = 10, height = 7)
ggsave(gg_ann_vs_imis, 
       filename = "figs/fig5_ANN-vs-IMIS-posterior.png", 
       width = 10, height = 7)
ggsave(gg_ann_vs_imis, 
       filename = "figs/fig5_ANN-vs-IMIS-posterior.jpeg", 
       width = 10, height = 7)
