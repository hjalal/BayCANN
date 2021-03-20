library(keras)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Set working directory
setwd()

# ==================
# Input parameters
n_obs <- 10000
n_iter <- 10000
n_hidden_nodes <- 100
n_hidden_layers <- 2 
n_epochs <- 10000
verbose <- 0
n_batch_size <- 2000
validation_split <- 0.2
n_chains <- 4

# =================================================
# functions
scale_data <- function(unscaled_data){
  vec.maxs <- apply(unscaled_data, 2, max) 
  vec.mins <- apply(unscaled_data, 2, min)
  vec.ones <- matrix(1, nrow = nrow(unscaled_data), 1)
  mat.maxs <- vec.ones %*% vec.maxs
  mat.mins <- vec.ones %*% vec.mins
  scaled_data <- 2 * (unscaled_data - mat.mins) / (mat.maxs - mat.mins) - 1
  results <- list(scaled_data = scaled_data, vec.mins = vec.mins, vec.maxs = vec.maxs)
  return(results)
}
unscale_data <- function(scaled_data, vec.mins, vec.maxs){
  vec.ones <- matrix(1, nrow = nrow(scaled_data), 1)
  mat.mins <- vec.ones %*% vec.mins
  mat.maxs <- vec.ones %*% vec.maxs
  unscaled_data <- (scaled_data + 1) * (mat.maxs - mat.mins) / 2 + mat.mins
}

load("data/03_targets-crc-nhm_100-runs.RData")
targets_combinedmean100 <- rbind(df.true.adeno.100runs, df.true.crc.100runs)
targets_combinedse100 <- rbind(df.true.adeno.se.100runs, df.true.crc.se.100runs)
denom_seq <- c(rep(100,20), rep(100000,16)) # adenomas are per 100, and CRC per 100K
true_targets_mean <- rowMeans(targets_combinedmean100[,4:103]) / denom_seq #pooled average of 100 sims
true_targets_se <- sqrt(rowMeans(targets_combinedse100[,4:103]^2)) / denom_seq #pooled standard error of 100 sims

load("data/05_DoE-unif-crc-nhm-det.rData")
df_nn <- data.frame(cbind(samp_i_unif, out_i_det_unif))
y_idx <- 10:45 
x_idx <- 1:9
Xunscaled <- df_nn[,x_idx]
Yunscaled <- df_nn[,y_idx]

# scale the PSA inputs and outputs
xresults <- scale_data(Xunscaled) 
yresults <- scale_data(Yunscaled)
xscaled <- xresults$scaled_data 
yscaled <- yresults$scaled_data 
xmins <- xresults$vec.mins
xmaxs <- xresults$vec.maxs
ymins <- yresults$vec.mins
ymaxs <- yresults$vec.maxs
y_names <-colnames(out_i_det_unif)
x_names <-colnames(samp_i_unif)

# get the true data
x_true_data <- read.csv(file="data/01_true-params.csv", header = T, sep = ",")
x_true_unscaled <- x_true_data$x

# =====================================
# Scale the targets and their SE
y_targets <- 2 * (true_targets_mean - ymins) / (ymaxs - ymins) - 1
y_targets_se <- 2 * (true_targets_se) / (ymaxs - ymins)
y_targets <- t(as.matrix(y_targets))
y_targets_se <- t(as.matrix(y_targets_se))

# ======================================
# Verify the truth is within the bounds of the x's!
par(mfrow = c(3,3), mar=c(1,1,1,1))
for (i in 1:9){
  hist(Xunscaled[,i], main = x_names[i])
  abline(v=x_true_unscaled[i],col="red")
}


# ============== TensorFlow Keras ANN Section ========================
N <- floor(n_obs * .8)
Nt <- n_obs-N
train_ind <- sample(n_obs,N)
test_ind <- setdiff(1:n_obs, train_ind) 
X <- xscaled[train_ind,]
Y <- yscaled[train_ind,]
Xt <- xscaled[test_ind,]
yt <- yscaled[test_ind,]
num_outputs <- ncol(Y)
indcol <- 1:36
x_train <- data.matrix(X)
y_train <- data.matrix(Y[,indcol])
x_test <- data.matrix(Xt)
y_test <- data.matrix(yt[,indcol])
n_outputs <- dim(y_test)[2]
n_inputs <- dim(x_test)[2]
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = n_hidden_nodes, activation = 'tanh', input_shape = n_inputs) %>% 
  layer_dense(units = n_hidden_nodes, activation = 'tanh') %>%
  layer_dense(units = n_hidden_nodes, activation = 'tanh') %>%
  layer_dense(units = n_outputs)
summary(model)
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)
keras.time <- proc.time()
history <- model %>% fit(
  x_train, y_train, 
  epochs = n_epochs, batch_size = N, 
  validation_split = validation_split,
  verbose = verbose
)
proc.time() - keras.time #keras ann fitting time

png(filename='output/ann_convergence.png')
plot(history)
dev.off()

model %>% evaluate(x_test, y_test)
weights <- get_weights(model) #get ANN weights
pred <- model %>% predict(x_test)
png(filename='output/ann_validation_vs_observed.png')
par("mar", mfrow = c(6,6), mar=c(1,1,1,1))
for (o in 1:n_outputs){
  plot(y_test[,o], pred[,o]) 
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
m <- stan_model("code/post_multi_perceptron.stan")
stan.time <- proc.time()
s <- sampling(m, data = stan.dat, iter = n_iter, chains = n_chains, 
              pars = c("Xq"))
proc.time() - stan.time # stan sampling time
fitmat = as.matrix(s)
Xq <- fitmat[,grep("Xq", colnames(fitmat))]

# get ypred using stan output and the Keras ANN
Xqmeans <- colMeans(Xq)
pred_keras <- model %>% predict(Xq)
pred_keras_means <- colMeans(pred_keras)
plot(y_targets, pred_keras_means)

# Scale the posteriors
Xq_unscaled <- unscale_data(Xq, vec.mins = xmins, vec.maxs = xmaxs)

# SAve the unscaled posterior samples
write.csv(Xq_unscaled, file = "output/calibrated_posteriors.csv")


