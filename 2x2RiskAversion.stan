
data {
  int<lower=0> ngames;
  
  // action counts
  vector[2] UpDown[ngames];
  vector[2] LeftRight[ngames];
  
  // payoffs
  matrix[2,2] Urow[ngames]; 
  matrix[2,2] Ucol[ngames];
  
  
  vector[2] prior_lambda; // log-normal prior
  vector[2] prior_r; // log-normal prior
  
  real<lower=0> w; // tuning parameter (a scalar)
  
  // =1 if you want to use the data, oltherwise you're tracing out the locus
  int UseData; 
  
}


parameters {
  real<lower=0> lambda; // logit choice precision
  real<lower=0> r; // risk aversion parameter in u(x) = x^r
  
  simplex[2] sigmaRow[ngames]; // Row player's mixed strategy
  simplex[2] sigmaCol[ngames]; // Column player's mixed strategy
}

transformed parameters {
  
  real objFun = 0;
  
  for (gg in 1:ngames) {
    objFun += -sum(pow(sigmaRow[gg] -softmax(lambda*pow(Urow[gg],r)*sigmaCol[gg]),2.0));
    objFun += -sum(pow(sigmaCol[gg] -softmax(lambda*pow(Ucol[gg],r)*sigmaRow[gg]),2.0));
  }
  
}

model {
  
  // QRE penalty function
  target += w*objFun;
  
  // priors
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  r      ~ lognormal(prior_r[1],prior_r[2]);
  
  if (UseData==1) {
    // likelihood
    for (gg in 1:ngames) {
      target += UpDown[gg]'*log(sigmaRow[gg]);
      target += LeftRight[gg]'*log(sigmaCol[gg]);
    }
  }
  
  
}

