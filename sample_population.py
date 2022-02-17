## SAMPLE SYNTHETIC POPULATION

###### CREATE THE SIMULATED POPULATION PICKLE FILE 
'''
Sample a population of n agents 
The file will contain for each country the following elements

- **AGE** : a np.array of length = n  which represents the age for each of the agents sampled 
- **HOUSEHOLDS**: a np.array of lenght = n indicating the type of household for each of the agents samples
- **DIABETES**: a np.array of length = n indicating whether the agent has or does not have diabetes  --> values 1 or 0 
- **HYPERTENSION**: a np.array of length = n indicating whether the agent has or does not have hypertension --> values 1 or 0 
- **AGE-GROUPS**: a np.array of length = 101 indicating the agents that have age == i 
'''

import pandas as pd
import numpy as np
import ipynb
from numba import jit
from numba.typed import List
import pickle
import os
import json
from sample_households import *  # import the Sample Comorbidities script and functions
from sample_comorbidities import * # import the Sample Households script and functions

b_dir = './'  # set base directory

## Function to Sample Population 
def sample_population(country,n = 10000000):
    '''FUNCTION to sample a synthetic population'''

    #np.random.seed(seed) # set the seed 
    n = int(n) # set the number of agents to simulate
    
    # Create households for the specific country --> RETURNS: Households and Age for each agent 
    print("Making households... ")
    if country == 'Italy':
        households, age, households_tot = sample_households_italy(n)
    elif country == 'Spain':
        households, age, households_tot= sample_households_spain(n)
    elif country == 'Germany':
        households, age, households_tot= sample_households_germany(n)
    elif country == 'France':
        households, age, households_tot = sample_households_france(n)
    else: 
        print('Function not defined for country %s' %country)
        
    households = households.astype('int64')
    age = age.astype('int64')
    print("Done.")
    
    # Create age_groups --> RETURNS: List of number of agents for each age value that have that age
    print("Creating age group sector array... ")
    n_age = 101 
    age_groups = tuple([np.where(age == i)[0] for i in range(0, n_age)]) # gives the position of the elements for a given age value 
    print("Done.")

    # Sample comorbidities
    print("Sampling comorbidities... ")
    diabetes, hypertension = None, None
    diabetes, hypertension = sample_joint_comorbidities(age, country)
    diabetes = diabetes.astype('int64')
    hypertension = hypertension.astype('int64')
    print("Done.")
    
    # Save and export to a pickle file 
    print("Saving... ")
    pickle.dump((age, households, households_tot, diabetes, hypertension, age_groups), open(os.path.join(b_dir,'{}_population_{}.pickle'.format(country, int(n))), 'wb'), protocol=4)
    print("Done.")
    print('####')
    
## SIMULATE SYNTHETIC POPULATION ITALY; 500k
#sample_population('Italy',500000)

## SIMULATE SYNTHETIC POPULATION SPAIN; 500k
#sample_population('Spain',500000)

## SIMULATE SYNTHETIC POPULATION GERMANY; 500k
#sample_population('Germany',500000)

## SIMULATE SYNTHETIC POPULATION FRANCE; 500k
#sample_population('France',500000)

## READ THE SIMULATED SYNTHETIC POPULATION FILE
#country = 'Italy' 
#n = 500000
#age, households, households_tot, diabetes, hypertension, age_groups = pd.read_pickle(os.path.join(b_dir,'{0}_population_{1}.pickle'.format(country,n)))