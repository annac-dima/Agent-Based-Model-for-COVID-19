## GLOBAL PARAMETERS
'''
Script which defines function to set the global parameters used throughout the simulations. These functions and parameters are then recalled in the overall SIR and ABM executions.   

The file contains the definition of the followings: 
- ** (1) Calibrate p_document mild Function**: function to set per-day probability of documentation for mild cases such that the cumulative probability of documentation before recovery matches p_target
- ** (2) Probability to Infect Household Function**: function to set per-day probability of infecting each household member so that the probability of infecting before becoming isolated matches the target secondary attack rate
'''

# import libraries and modules 
import numpy as np
import scipy.stats
from sample_comorbidities import *  # import the Sample Comorbidities script and functions
from sample_households import * # import the Sample Households script and functions

# (1) Calibrate p_document mild Function 
def calibrate_p_document_mild(p_target, country, p_mild_severe, mean_time_mild_recovery, mean_time_to_severe):
    '''
    Sets per-day probability of documentation for mild cases such that the cumulative
    probability of documentation before recovery matches p_target
    '''
    n  = 10000
   
    def total_p_document(p_document):
        time_document = np.random.geometric(p_document, size=n)
        time_to_recovery = np.random.exponential(mean_time_mild_recovery, size=n)
        return (time_document < time_to_recovery).mean()
    #binary search
    eps = 0.0001 # epsilon parameter
    ub = 1 # upper bound
    lb = 0 # lower bound
    p_document = (lb + ub)/2. # initialize p_document equal to = 0.5
    cumulative_sar = total_p_document(p_document)
    while np.abs(cumulative_sar - p_target) > eps:
        if cumulative_sar < p_target:
            lb = p_document
            p_document = (p_document + ub)/2
        else:
            ub = p_document
            p_document = (p_document + lb)/2
        cumulative_sar = total_p_document(p_document)
    return p_document

# (2) Probability to Infect Household Function 
def get_p_infect_household(n, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std, asymptomatic_transmissibility):
    '''
    Sets per-day probability of infecting each household member so that the probability
    of infecting before becoming isolated matches the target secondary attack rate
    '''
    
    def threshold_exponential(mean, num):
        return 1 + np.round(np.random.exponential(mean-1, size=num))
    
    def threshold_log_normal(mean, sigma, num):
        x = np.random.lognormal(mean, sigma, size=num)
        x[x <= 0] = 1
        return np.round(x)

    def total_p_infection(p_infect, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std):
        time_to_isolate = threshold_exponential(mean_time_to_isolate, 5000)
        time_to_activate = threshold_log_normal(time_to_activation_mean, time_to_activation_std, 5000)
        time_infect_asymp = np.random.geometric(p_infect*asymptomatic_transmissibility, size=5000)
        time_infect_symp = np.random.geometric(p_infect, size=5000)
        return (1 - (time_infect_asymp > time_to_activate)*(time_infect_symp > time_to_isolate)).mean()

    eps = 0.0001
    
    #probability for an infected patient to infect each member of the household each day
    #calibrate to match
    #https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30462-1/fulltext
    #binary search
    target_secondary_attack_rate = 0.35 # TARGET SECONDARY ATTACK RATE
    ub = 1 # upper bound
    lb = 0 # lower bound 
    p_infect_household = 0.5 # initialize p_infect_house = 0.5
    cumulative_sar = total_p_infection(p_infect_household, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std)
    while np.abs(cumulative_sar - target_secondary_attack_rate) > eps:
        if cumulative_sar < target_secondary_attack_rate:
            lb = p_infect_household
            p_infect_household = (p_infect_household + ub)/2
        else:
            ub = p_infect_household
            p_infect_household = (p_infect_household + lb)/2
        cumulative_sar = total_p_infection(p_infect_household, mean_time_to_isolate, time_to_activation_mean, time_to_activation_std)
    
    p_infect_household_array = np.zeros(n) # create an empy array of length n
    p_infect_household_array[:] = p_infect_household # store p_infect_household n times in the array
     
    return p_infect_household_array