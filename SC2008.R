library(tidyverse)
library(rstan)
library(kableExtra)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)



ncores<-1

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))




SC2008<-"data/SC2008Summarized.rds" |> readRDS()

UpDown<-list()
LeftRight<-list()

for (gg in 1:12) {
  
  d<-SC2008 |>
    filter(game==gg)
  
  UpDown[[gg]]<-c(d$Up,d$count-d$Up)
  LeftRight[[gg]]<-c(d$Left,d$count-d$Left)
  
  
}

# Game payoffs

UROW<-list(
  rbind(c(10,0),c(9,8)),
  rbind(c(9,0),c(6,8)),
  rbind(c(8,0),c(7,10)),
  rbind(c(7,0),c(5,9)),
  rbind(c(7,0),c(4,8)),
  rbind(c(7,1),c(3,8)),
  rbind(c(10,4),c(9,14)),
  rbind(c(9,3),c(6,11)),
  rbind(c(8,3),c(7,13)),
  rbind(c(7,2),c(5,11)),
  rbind(c(7,2),c(4,10)),
  rbind(c(7,3),c(3,10))
)
UCOL<-list(
  rbind(c(8,18),c(9,8)) |> t(),
  rbind(c(4,13),c(7,5)) |> t(),
  rbind(c(7,14),c(7,4)) |> t(),
  rbind(c(4,11),c(6,2)) |> t(),
  rbind(c(2,9),c(5,1)) |> t(),
  rbind(c(1,7),c(5,0)) |> t(),
  
  rbind(c(12,22),c(9,8)) |> t(),
  rbind(c(7,16),c(7,5)) |> t(),
  rbind(c(9,17),c(7,4)) |> t(),
  rbind(c(6,13),c(6,2)) |> t(),
  rbind(c(4,11),c(5,1)) |> t(),
  rbind(c(3,9),c(5,0)) |> t()
)

# Get the data into a Stan-friendly format
d<-list(
  ngames = 12,
  
  UpDown = UpDown,
  LeftRight = LeftRight,
  Urow = UROW,
  Ucol = UCOL,
  
  prior_lambda = c(-1.52,1.41),
  prior_r = c(log(0.5),0.5),
  
  w=10000,
  
  UseData=1
)


################################################################################
# Trace out the locus of risk-neutral logit QRE 
################################################################################
# This worked much better by focusing on one game at a time

file<-"outputs/SC2008_locus.rds"

if (!file.exists(file)) {

  model<-"2x2RiskNeutralEliminateLambda.stan" |> 
    stan_model()
  
  
  
  locus<-tibble()
  
  for (gg in 1:length(UROW)) {
    
    print(paste("Computing locus for game",gg))
    
    dg<-list(
      ngames = 1,
      UpDown = list(d$UpDown[[gg]]),
      LeftRight = list(d$LeftRight[[gg]]),
      Urow = list(d$Urow[[gg]]),
      Ucol = list(d$Ucol[[gg]]),
      
      w=10000,
      
      UseData=0
      
    )
    
    samples<-model |>
      sampling(data=dg,seed=42,
               cores = ncores,
               iter=10000)
    
    locus<-rbind(
      locus,
      tibble(
        game = gg,
        lambda = extract(samples)$lambda,
        Up = extract(samples)$sigmaRow[,1,1],
        Left = extract(samples)$sigmaCol[,1,1],
      )
    )
    
    
    
  }
  
  locus |> saveRDS(file)
}

################################################################################
# Estimate the model with risk neutral players
################################################################################

model<-"2x2RiskNeutral.stan" |> 
  stan_model()

file<-"outputs/SC2008_EstimatesRiskNeutral.rds"

if (!file.exists(file)) {
  Fit<-model |>
    sampling(data=d,seed=42,
             cores = ncores)
  
  Fit |>
    saveRDS(file)
}




################################################################################
# Estimate the model with risk aversion
################################################################################

model<-"2x2RiskAversion.stan" |> 
  stan_model()

file<-"outputs/SC2008_EstimatesRiskAversion.rds"

if (!file.exists(file)) {
  Fit<-model |>
    sampling(data=d,seed=42,
             cores = ncores)
  
  Fit |>
    saveRDS(file)
}
