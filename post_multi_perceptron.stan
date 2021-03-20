functions {
 matrix calculate_alpha(matrix X, matrix beta_first, matrix[] beta_middle, matrix beta_last, matrix weight_first, matrix[] weight_middle, matrix weight_last, int N, int num_hidden_layers, int num_hidden_nodes){
    matrix[rows(X), cols(beta_first)] layer_values_first;
    matrix[rows(X), cols(beta_first)] layer_values_middle[num_hidden_layers];
    matrix[rows(X), cols(beta_last)] alpha;
    layer_values_first = tanh(beta_first + X * weight_first);   
    layer_values_middle[1] = tanh(beta_middle[1] + layer_values_first * weight_middle[1]);
    for(i in 2:(num_hidden_layers)){
      layer_values_middle[i] = tanh(beta_middle[i] + layer_values_middle[i-1] * weight_middle[i]);
    }
    alpha = beta_last + layer_values_middle[num_hidden_layers] * weight_last;
    return alpha;
  }
}

data {
  int<lower=0> num_targets;
  int<lower=0> num_inputs;
  int<lower=0> num_outputs;
  int<lower=0> num_hidden_nodes;
  int<lower=1> num_hidden_layers;
  matrix[num_targets,num_outputs] y_targets;
  matrix[num_targets,num_outputs] y_targets_se;
  matrix[num_inputs, num_hidden_nodes] weight_first;
  matrix[num_hidden_nodes, num_hidden_nodes] weight_middle[num_hidden_layers];
  matrix[num_hidden_nodes, num_outputs] weight_last;
  matrix[1, num_hidden_nodes] beta_first;
  matrix[1, num_hidden_nodes] beta_middle[num_hidden_layers];
  matrix[1, num_outputs] beta_last;
}

parameters {
  matrix<lower=-1, upper=1>[num_targets,num_inputs] Xq;
}
model{
  matrix[1, num_outputs] alpha_post;
    alpha_post = calculate_alpha(Xq, beta_first, beta_middle, beta_last, weight_first, weight_middle, weight_last, num_targets, num_hidden_layers, num_hidden_nodes);
    to_vector(y_targets) ~ normal(to_vector(alpha_post),to_vector(y_targets_se)); //get SE directly from data
    to_vector(Xq) ~ uniform(-1,1); // assume uniform prior
}




