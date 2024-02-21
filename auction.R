library(tidyverse)
library(rstan)
library(kableExtra)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

ncores<-1

rerun<-TRUE

# Number of bids and signals for the discretization
ns<-101

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

model <- "auction.stan" |>
  stan_model()

###############################################################################
# Simulate the logit QRE for lambda = 10, r = 0.5
###############################################################################

file<-"outputs/auction_QRE.rds"

if (!file.exists(file) | rerun) {

  d<-list(
    N=3,
    ns = ns,
    
    w = 100000,
    
    prior_lambda<-c(log(1),2),
    prior_r<-c(log(1),1),
    
    lambda_r_override=c(10,0.5),
    
    action_count = matrix(0,ns,ns)
    
  )
  
  #stop()
  
  samples<-model |>
    sampling(data=d,seed=42)
  
  summary(samples)$summary |>
    saveRDS("outputs/auction_QRE_SimulationSummary.rds")
  
  # Extract the draw with the best value of objFun
  
  objFun<-extract(samples)$objFun
  objFunII<-which.max(objFun)
  
  biddist<-extract(samples)$biddist[objFunII,,]
  
  biddist |>
    saveRDS(file)
  
} else {
  biddist<-file |> readRDS()
}

################################################################################
# Simulate data from this and then estimate the model
################################################################################

file<-"outputs/auction_estimates.rds" 

if (!file.exists(file) | rerun) {
  set.seed(123)
  action_count = matrix(0,ns,ns)
  
  playsPerSignal<-20
  
  for (ss in 1:ns) {
    bids<-sample(1:ns,playsPerSignal,replace=TRUE,prob = biddist[,ss])
    for (bb in 1:ns) {
      action_count[bb,ss]<-sum(bids==bb)
    }
  }
  
  d<-list(
    N=3,
    ns = ns,
    
    w = 100000,
    
    prior_lambda<-c(log(1),2),
    prior_r<-c(log(1),1),
    
    lambda_r_override=c(-1,-1),
    
    action_count = action_count
    
  )
  
  Fit<-model |>
    sampling(data=d,seed=42)
  
  summary(Fit)$summary |> 
    saveRDS(file)
}