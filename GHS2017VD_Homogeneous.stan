
/*
Homogeneous-agents logit QRE for the Volunteer's Dilemma
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
  
  int UseData;
  
}

parameters {
  
  // probability of volunteering
  vector<lower=0,upper=1>[nGroups] sigmaV;
  
}

transformed parameters {
  
  real lambda;
  real objFun = 0;
  
  {
  
  // convert probabilities into log probabilities
    vector[nGroups] rhoV = log(sigmaV);
    vector[nGroups] rhoN = log(1.0-sigmaV);
    
    vector[nGroups] logprobDiff = rhoV-rhoN;
    vector[nGroups] utilityDiff = 
      V-c-V*(1.0-exp((GroupSize-1.0).*rhoN))-L*exp((GroupSize-1.0).*rhoN);
      
    lambda = fdim(sum(logprobDiff.*utilityDiff)/sum(pow(utilityDiff,2)),0.0);
    
    //objFun = -quad_form(W,logprobDiff-lambda*utilityDiff);
    
    objFun += -sum(
        pow(sigmaV-inv_logit(lambda*utilityDiff),2.0)
        +pow(1-sigmaV-inv_logit(-lambda*utilityDiff),2.0)
    );
      
    
    
    
  } 
}

model {
  
  target += w*objFun;
  
  
  if (UseData==1) {
    target += binomial_lpmf(VolunteerCount | DecisionCount,sigmaV[GroupSizeID]);
  }
}

