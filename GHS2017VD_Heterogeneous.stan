
/*
Heterogeneous-agents logit QRE for the Volunteer's Dilemma
*/

data {
  
  // participant-level data
  int<lower=0> N; // number of participants
  int<lower=0> VolunteerCount[N]; // count of the number of times V was played
  int<lower=0> DecisionCount[N]; // total number of decisions the participant made
  int<lower=0> GroupSizeID[N]; // id variable identifying which group size the participant was in
  
  // treatment-level data (group sizes)
  int nGroups; 
  vector[nGroups] GroupSize;
  
  // payoff parameters (constant across treatments)
  real V; // benefit if one participantg volunteers
  real c; // cost of volunteering
  real L; // payoff if nobody volunteers
  
  // Weighting parameter (a scalar)
  real<lower=0> w;
  
  // lower and upper bound (in terms of standard deviations) for heterogeneity
  real<lower=0> sdMax;
  
  // priors
  vector[2] prior_lambda; // log-normal
  vector[2] prior_MU; // normal
  real<lower=0> prior_TAU; // half-Cauchy
  
  // some uniforms for Monte Carlo integration
  int nunif;
  vector<lower=0,upper=1>[nunif] unif;
  
  int UseData;
  
  // =1 if you want to estimate the warm glow model, 
  // otherwise you get the duplicate aversion model
  int WarmGlow; 
  
  
  
}

transformed data {
  
  vector[nunif] ztrunc = inv_Phi(
      Phi(-sdMax)+unif*(Phi(sdMax)-Phi(-sdMax))
    );
  
}

parameters {
  
  real MU; // mean of warm glow utility
  real<lower=0> TAU; // standard deviation of warm glow utility
  real<lower=0> lambda; // logit choice precision
  
  // probability of volunteering
  vector<lower=0,upper=1>[nGroups] sigmaV;
  // warm glow from volunteering, normalized 
  vector<lower=-sdMax,upper=sdMax>[N] ztheta;
  
}

transformed parameters {
  
  real objFun = 0;
  
  vector[N] theta = MU+TAU*ztheta;
  vector[N] sigmaVi; // participant i's volunteer probability
  
  {
    
    vector[nGroups] meanPBR;
    
    vector[nunif] thetaSim = MU+TAU*ztrunc;
    
    if (WarmGlow==1) {
    
    for (gg in 1:nGroups) {
      
      vector[nunif] UtilityDiff = V-c+thetaSim
                    -V*(1.0-pow(1.0-sigmaV[gg],GroupSize[gg]-1.0))
                    -L*pow(1.0-sigmaV[gg],GroupSize[gg]-1.0)
                    ;
      meanPBR[gg] = mean(inv_logit(lambda*UtilityDiff));
      
    }
    
    sigmaVi = inv_logit(lambda*(
                    V-c+theta
                    -V*(1.0-pow(1.0-sigmaV[GroupSizeID],GroupSize[GroupSizeID]-1.0))
                    -L*pow(1.0-sigmaV[GroupSizeID],GroupSize[GroupSizeID]-1.0)
      )
    );
    
    } else {
      
    for (gg in 1:nGroups) {
      
      vector[nunif] UtilityDiff = V-c-thetaSim*(1.0-pow(1.0-sigmaV[gg],GroupSize[gg]-1))
                    -V*(1.0-pow(1.0-sigmaV[gg],GroupSize[gg]-1.0))
                    -L*pow(1.0-sigmaV[gg],GroupSize[gg]-1.0)
                    ;
      meanPBR[gg] = mean(inv_logit(lambda*UtilityDiff));
      
    }
    
    sigmaVi = inv_logit(lambda*(
                    V-c-theta.*(1.0-pow(1.0-sigmaV[GroupSizeID],GroupSize[GroupSizeID]-1))
                    -V*(1.0-pow(1.0-sigmaV[GroupSizeID],GroupSize[GroupSizeID]-1.0))
                    -L*pow(1.0-sigmaV[GroupSizeID],GroupSize[GroupSizeID]-1.0)
      )
    );
      
    }

    objFun += -sum(pow(sigmaV-meanPBR,2.0));
  } 
}

model {
  
  // hierarchical structure
  ztheta ~ std_normal();
  
  // penalty function
  target += w*objFun;
  
  // prior
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  MU ~ normal(prior_MU[1],prior_MU[2]);
  TAU ~ cauchy(0.0,prior_TAU);
  
  
  // likelihood contribution
  if (UseData==1) {
    target += binomial_lpmf(VolunteerCount | DecisionCount,sigmaVi);
  }
}

