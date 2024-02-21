library(tidyverse)
library(rstan)
library(kableExtra)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)



ncores<-1

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

GHS2017<-"data/GHS2017VD.csv" |>
  read.csv() |>
  dplyr::select(ID,Decision..1.vol..,Other.Id.,SessionID) |>
  # This dataset does not explicitly store the group size
  # I can infer it from the "Other.Id." variable, but
  # this is not consistent with the experiment description
  # in the "Procedures" section of the paper. Specifically,
  # it looked like there were two GroupSize=7 session, when
  # in reality there should have been one with 9, and one
  # with 12. This fixes things
  mutate(GroupSize = str_count(Other.Id., "ID")+1,
         GroupSize = ifelse(SessionID==10,9,GroupSize),
         GroupSize = ifelse(SessionID==11,12,GroupSize)
  ) |>
  rename(Volunteer = Decision..1.vol..) |>
  mutate(
    id = paste0(ID,SessionID) |> as.factor() |> as.numeric()
  ) |>
  group_by(id,GroupSize) |>
  summarize(
    VolunteerCount = sum(Volunteer),
    DecisionCount = n()
  ) |>
  arrange(GroupSize) |>
  ungroup() |>
  mutate(
    GroupSizeID = (GroupSize/1000) |> paste("-") |> as.factor() |> as.numeric()
  )

GHS2017 |> 
  saveRDS("data/GHS2017_cleaned.rds")

###############################################################################
# Approximating the logit QRE correspondence
###############################################################################

file<-"outputs/GHS2017_locus.rds"

if (!file.exists(file)) {
  
  model<-"GHS2017VD_Homogeneous.stan" |>
    stan_model()
  
  d<-list(
    N = dim(GHS2017)[1],
    VolunteerCount = GHS2017$VolunteerCount,
    DecisionCount = GHS2017$DecisionCount,
    GroupSizeID = GHS2017$GroupSizeID,
    
    nGroups = GHS2017$GroupSize |> unique() |> length(),
    GroupSize = GHS2017$GroupSize |> unique(),
    
    V = 1,
    c = 0.2,
    L = 0.2,
    
    w = 10000,
    
    UseData = 0
  )
  
  Fit<-model |>
    sampling(
      data=d,seed=42,
      iter=10000,
      cores=ncores
    )
  
  Fit |>
    saveRDS(file)
  
}


###############################################################################
# Homogeneous agents logit QRE
###############################################################################

file<-"outputs/GHS2017_EstimatesHomogeneous.rds"

if (!file.exists(file)) {
  
  model<-"GHS2017VD_Homogeneous.stan" |>
    stan_model()
  
  d<-list(
    N = dim(GHS2017)[1],
    VolunteerCount = GHS2017$VolunteerCount,
    DecisionCount = GHS2017$DecisionCount,
    GroupSizeID = GHS2017$GroupSizeID,
    
    nGroups = GHS2017$GroupSize |> unique() |> length(),
    GroupSize = GHS2017$GroupSize |> unique(),
    
    V = 1,
    c = 0.2,
    L = 0.2,
    
    w = 10000,
    
    UseData = 1
  )
  
  # This one runs so fast that it isn't worth running the chains in parallel
  # (on my laptop, at least)
  Fit<-model |>
    sampling(
      data=d,seed=42,
      cores=ncores
    )
  
  Fit |>
    saveRDS(file)
  
}

################################################################################
# For the heterogeneous models, I explored the implications of bounding the 
# distribution at different numbers of standard deviations from the mean
# GHS2017 use one sd
# It didn't change things much, so the loop below does it for the +/1 1sd
# option.
################################################################################

for (sd in 1:1) {
  
  print(paste("estimating for +/-",sd,"sd"))

  ##############################################################################
  # Warm glow volunteering
  ##############################################################################
  
  file<-paste0("outputs/GHS2017_EstimatesWarmGlow",sd,".rds" )
  
  if (!file.exists(file)) {
    
    set.seed(123)
    
    model<-"GHS2017VD_Heterogeneous.stan" |>
      stan_model()
    
    nunif<-100
    
    d<-list(
      N = dim(GHS2017)[1],
      VolunteerCount = GHS2017$VolunteerCount,
      DecisionCount = GHS2017$DecisionCount,
      GroupSizeID = GHS2017$GroupSizeID,
      
      nGroups = GHS2017$GroupSize |> unique() |> length(),
      GroupSize = GHS2017$GroupSize |> unique(),
      
      V = 1,
      c = 0.2,
      L = 0.2,
      
      w = 10000,
      
      UseData = 1,
      
      sdMax = sd,
      
      prior_lambda = c(log(10),1),
      prior_MU = c(0,1),
      prior_TAU = 1,
      
      nunif=nunif,
      unif = runif(nunif),
      
      UseData =1,
      
      WarmGlow = 1
  
    )
    
    # This one is slow enough that it is worth running the chains in parallel
    Fit<-model |>
      sampling(
        data=d,seed=42,
        # Running RStan's default options, I got a "Bulk ESS" warning,
        iter = 10000,
        # Running RStan's default options, I got 6 divergent transitions
        control = list(adapt_delta = 0.9),
        # don't store the normalized individual-level parameters
        pars = c("ztheta"), include=FALSE
      )
    
    
    # I'm only interested in saving the summary of this simulation
    FitSummary<-summary(Fit)$summary
    
    FitSummary |>
      saveRDS(file)
    
  }
  
  file<-paste0("outputs/GHS2017_EstimatesDuplicateAversion",sd,".rds")
  
  if (!file.exists(file)) {
    
    set.seed(123)
    
    model<-"GHS2017VD_Heterogeneous.stan" |>
      stan_model()
    
    nunif<-100
    
    d<-list(
      N = dim(GHS2017)[1],
      VolunteerCount = GHS2017$VolunteerCount,
      DecisionCount = GHS2017$DecisionCount,
      GroupSizeID = GHS2017$GroupSizeID,
      
      nGroups = GHS2017$GroupSize |> unique() |> length(),
      GroupSize = GHS2017$GroupSize |> unique(),
      
      V = 1,
      c = 0.2,
      L = 0.2,
      
      w = 10000,
      
      UseData = 1,
      
      sdMax = 1,
      
      prior_lambda = c(log(10),1),
      prior_MU = c(0,1),
      prior_TAU = 1,
      
      nunif=nunif,
      unif = runif(nunif),
      
      UseData =1,
      
      WarmGlow = 0
      
    )
    
    # This one is slow enough that it is worth running the chains in parallel
    Fit<-model |>
      sampling(
        data=d,seed=42,
        # Running RStan's default options, I got a "Bulk ESS" warning,
        iter = 10000,
        # Running RStan's default options, I got 6 divergent transitions
        control = list(adapt_delta = 0.9),
        # don't store the normalized individual-level parameters
        pars = c("ztheta"), include=FALSE
      )
    
    
    # I'm only interested in saving the summary of this simulation
    FitSummary<-summary(Fit)$summary
    
    FitSummary |>
      saveRDS(file)
    
  }
}