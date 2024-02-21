
data {
  int<lower=0> ngames;
  
  // action counts
  vector[2] UpDown[ngames];
  vector[2] LeftRight[ngames];
  
  // payoffs
  matrix[2,2] Urow[ngames]; 
  matrix[2,2] Ucol[ngames];
  
  
  real<lower=0> w; // tuning parameter (a scalar)
  
  // =1 if you want to use the data, oltherwise you're tracing out the locus
  int UseData; 
  
}


parameters {
  
  simplex[2] sigmaRow[ngames]; // Row player's mixed strategy
  simplex[2] sigmaCol[ngames]; // Column player's mixed strategy
}

transformed parameters {
  
  real objFun = 0;
  
  real lambda;
  
  {
    
    real numerator = 0;
    real denominator = 0;
    
    for (gg in 1:ngames) {
      
      numerator += logit(sigmaRow[gg][1])*(Urow[gg][1,]-Urow[gg][2,])*sigmaCol[gg];
      numerator += logit(sigmaCol[gg][1])*(Ucol[gg][1,]-Ucol[gg][2,])*sigmaRow[gg];
      
      denominator += pow((Urow[gg][1,]-Urow[gg][2,])*sigmaCol[gg],2.0);
      denominator += pow((Ucol[gg][1,]-Ucol[gg][2,])*sigmaRow[gg],2.0);
    }
    
    lambda = fmax(numerator/denominator,0.0);
    
    for (gg in 1:ngames) {
      objFun += -sum(pow(sigmaRow[gg] -softmax(lambda*Urow[gg]*sigmaCol[gg]),2.0));
      objFun += -sum(pow(sigmaCol[gg] -softmax(lambda*Ucol[gg]*sigmaRow[gg]),2.0));
    }
    
    
  }
  
  
  
  
}

model {
  
  // QRE penalty function
  target += w*objFun;

  
  if (UseData==1) {
    // likelihood
    for (gg in 1:ngames) {
      target += UpDown[gg]'*log(sigmaRow[gg]);
      target += LeftRight[gg]'*log(sigmaCol[gg]);
    }
  }
  
  
}

