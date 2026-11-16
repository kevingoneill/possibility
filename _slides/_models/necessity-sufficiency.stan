functions {
  // given the necessity strength n, the sufficiency strength s,
  // and the prior probability p, calculate the causal strength
  // of a variable
  real K_NS(real n, real s, real p) {
    return (1-p)*n + p*s;
  }
}

data {
  int<lower=1> N;                          // number of data points
  int<lower=1> C;                          // number of conditions (2: control/manipulation)
  array[N] int<lower=1, upper=C> condition;      // condition per observation (1=control, 2=manipulation)
  vector<lower=0, upper=1>[N] cause_PC;    // causal judgments of the productive cause
  vector<lower=0, upper=1>[N] cause_DP;    // causal judgments of the double preventer
  int<lower=0, upper=1> prior_only;        // sample from just the prior (1) or the posterior (0)?
}

parameters {
  real<lower=0, upper=1> p_PC;
  real<lower=0, upper=1> p_PP;
  vector<lower=0, upper=1>[C] p_DP;
  
  // residual standard deviation
  vector<lower=0>[C] sigma_cause_PC;
  vector<lower=0>[C] sigma_cause_DP;
}

transformed parameters {
  vector<lower=0, upper=1>[C] N_PC;   // necessity
  vector<lower=0, upper=1>[C] N_DP;
  vector<lower=0, upper=1>[C] S_PC;   // sufficiency
  vector<lower=0, upper=1>[C] S_DP;  
  vector<lower=0, upper=1>[C] K_PC;   // causal strength
  vector<lower=0, upper=1>[C] K_DP;

  for (c in 1:C) {
    N_PC[c] = 1;
    N_DP[c] = 1;
    
    S_PC[c] = (1-p_PP)*(1-p_DP[c]) + (1-p_PP)*p_DP[c] + p_PP*p_DP[c];
    S_DP[c] = p_PC;
    
    K_PC[c] = K_NS(N_PC[c], S_PC[c], p_PC);
    K_DP[c] = K_NS(N_DP[c], S_DP[c], p_DP[c]);
  }
}

model {
  // uniform prior over prior probabilities
  p_PC ~ uniform(0, 1);
  p_PP ~ uniform(0, 1);
  p_DP ~ uniform(0, 1);

  // prior over standard deviations
  sigma_cause_PC ~ normal(0, 1);
  sigma_cause_DP ~ normal(0, 1);
  
  if (!prior_only) {
    for (i in 1:N) {
      cause_PC[i] ~ normal(K_PC[condition[i]], sigma_cause_PC[condition[i]]);
      cause_DP[i] ~ normal(K_DP[condition[i]], sigma_cause_DP[condition[i]]);
    }
  }
}

generated quantities {
  array[N] real cause_PC_hat = normal_rng(K_PC[condition], sigma_cause_PC[condition]);
  array[N] real cause_DP_hat = normal_rng(K_DP[condition], sigma_cause_DP[condition]);
}
