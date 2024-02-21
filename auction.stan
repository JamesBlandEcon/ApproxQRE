
/*
First-price auction with uniform independent values
*/

functions {
  
  
  vector u(vector x,real r) {
    
    return 1.0-exp(-r*x);
    
  }
}

data {
  
  int N; // number of bidders in game
  
  int ns; // number of bids/signals. These will be normalized to lie on the unit interval
  
  real<lower=0> w; // weighting parameter (a scalar)
  
  vector[2] prior_lambda; // longormal prior
  vector[2] prior_r; // lognormal prior
  
  /*
  Set both elements of this to negative numbers to run the estimation.
  Setting both to positive numbers overrides the parameters r and lambda
  so that we are drawing from the logit QRE with the parameters (lambda,r)
  in this vector
  */
  vector[2] lambda_r_override; 
  
  
  //A count of signal-bid pairs. Each column is a signal, each row is a bid
  matrix[ns,ns] action_count;
}

transformed data {
  
  
  // a vector of bids/signals
  vector[ns] BS;
  
  for (ss in 1:ns) {
    BS[ss] = (ss-1.0)/(ns-1.0+0.0);
  }
  
  
}


parameters {
  
  
  real<lower=0> lambda;
  real<lower=0> r;
  
  simplex[ns] bidDist;
  
  
}

transformed parameters {
  
  real objFun;
  vector[ns] bidcdf = cumulative_sum(bidDist);
  vector[ns] aggregateLogitResponse = rep_vector(0.0,ns);
  real likelihood;
  
  {
    real lambda_override;
    real r_override;
    
    if (fmin(lambda_r_override[1],lambda_r_override[2])>0) {
      lambda_override = lambda_r_override[1];
      r_override = lambda_r_override[2];
    } else {
      lambda_override = lambda;
      r_override = r;
    }
    
    vector[ns] PrWin = pow(bidcdf,N-1);
    matrix[ns,ns] logEqStrategy;
    
    for (ss in 1:ns) {
      
      vector[ns] U  = u(BS[ss]-BS,r_override).*PrWin;
      aggregateLogitResponse += softmax(lambda_override*U)/(0.0+ns);
      
      logEqStrategy[,ss] = log_softmax(lambda_override*U);
      
    }
    
    likelihood = sum(action_count.*logEqStrategy);
    
    
  }
  
  
  objFun = -sum(pow(bidDist-aggregateLogitResponse,2.0));
}


model {
  
  r ~ lognormal(prior_r[1],prior_r[2]);
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  
  target += w*objFun;
  target += likelihood;
  
  
  
}

generated quantities {
  
  matrix[ns,ns] biddist;
  
  {
    real lambda_override;
    real r_override;
    
    if (fmin(lambda_r_override[1],lambda_r_override[2])>0) {
      lambda_override = lambda_r_override[1];
      r_override = lambda_r_override[2];
    } else {
      lambda_override = lambda;
      r_override = r;
    }
    
    vector[ns] PrWin = pow(bidcdf,N-1);
    
    for (ss in 1:ns) {
      
      vector[ns] U  = u(BS[ss]-BS,r_override).*PrWin;
      biddist[,ss] = softmax(lambda_override*U);
      
    }
    
    
    
  }
  
}




