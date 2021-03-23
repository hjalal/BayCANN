# load libraries ========
library(keras)
library(rstan)
library(reshape2)
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
mdl_string = paste("model %>% layer_dense(units = n_hidden_nodes, activation = 'tanh', input_shape = n_inputs) %>%", 
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
  xscaled, yscaled, 
  epochs = n_epochs, batch_size = n_batch_size, 
  validation_split = n_test/(n_train + n_test),
  verbose = verbose
)
proc.time() - keras.time #keras ann fitting time

png(filename='output/ann_convergence.png')
plot(history)
dev.off()

weights <- get_weights(model) #get ANN weights
pred <- model %>% predict(xtest_scaled)
png(filename='output/ann_validation_vs_observed.png')
par("mar", mfrow = c(6,6), mar=c(1,1,1,1))
for (o in 1:n_outputs){
  plot(ytest_scaled[,o], pred[,o]) 
}
dev.off()


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
m <- stan_model("post_multi_perceptron.stan")
stan.time <- proc.time()
s <- sampling(m, data = stan.dat, iter = n_iter, chains = n_chains, 
              pars = c("Xq"))
proc.time() - stan.time # stan sampling time
fitmat = as.matrix(s)
Xq <- fitmat[,grep("Xq", colnames(fitmat))]
summary(m)

# Scale the posteriors
Xq_unscaled <- unscale_data(Xq, vec.mins = xmins, vec.maxs = xmaxs)

# Save the unscaled posterior samples
write.csv(Xq_unscaled, file = "output/calibrated_posteriors.csv")


#### Visualization of priors and posteriors ####
# Plot histogram for individual posteriors and compare to prior
# Read the unscaled posterior samples
Xq_unscaled <- read.csv(file = "output/calibrated_posteriors.csv")[, -1]
priordf <- data.frame(samp_i_unif_train)
postdf <- data.frame(Xq_unscaled)
colnames(postdf) <- x_names

priordf$type <- 'prior'
postdf$type <- paste('chain ', sort(rep(1:4, n_iter/2)))
priorpost <- rbind(priordf, postdf)
melt_df <- melt(priorpost)
line_df <- data.frame(variable = x_names, intercept = cbind(x_true_unscaled))
colnames(line_df) <- c("variable","intercept")
ggplot(melt_df, aes(value, fill=type, colour = type)) + 
  geom_density(alpha = 0.1) + 
  facet_wrap(~variable, scales="free") + 
  geom_vline(data=line_df, aes(xintercept=x_true_unscaled))
ggsave("output/convergence.png")

# Plot histogram for combined posterior and compare to the truth and the prior
melt_df_comb = melt_df
melt_df_comb[melt_df_comb == "chain  1" | melt_df_comb == "chain  2" |
               melt_df_comb == "chain  3" | melt_df_comb == "chain  4" ] <- "post"
ggplot(melt_df_comb, aes(value, fill=type, colour = type)) + 
  geom_density(alpha = 0.1) + 
  facet_wrap(~variable, scales="free") + 
  geom_vline(data=line_df, aes(xintercept=x_true_unscaled))
ggsave("output/prior_post_truth.png")

