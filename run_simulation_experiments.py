### CODE TO RUN SIMULATIONS FOR ITALY, SPAIN, GERMANY AND FRANCE
### EXPLORATIVE ANALYSIS TO INFER THE FREE MODEL PARAMETERS -- p_inf, d0, dmult

import numpy as np
from global_parameters import *
import scipy.special
import csv
from datetime import date
import numba
from seir_individual import run_complete_simulation
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
import time
import math
from multiprocessing import Pool
from multiprocessing import freeze_support
from itertools import product 
import joblib
from joblib import Parallel, delayed

#################################################################################

def simulation(tot_params_list):
    '''INPUT: tot_params_list >> list containinng the values for: [d0,pinf,dmult,country]'''
    time.sleep(50) # sleep time 
    ### INITIALIZATION 
    # Retrieve Simulation Parameters 
    d0 = tot_params_list[0]
    p_inf = tot_params_list[1]
    dmult = tot_params_list[2]
    country = tot_params_list[3]
    load_population = True # Either Load a population or re-generate it everytime
    n = 500000. # 500K SYNTHETIC POPULATION SIZES
    
    if country == 'Italy':
        real_pop_size = 60000000 # 60Millions ; real population size for the country
    elif country == 'Spain':
        real_pop_size = 47000000 # 47Millions ; real population size for the country
    elif country == 'Germany':
        real_pop_size = 80000000 # 80Millions ; real population size for the country
    elif country == 'France':
        real_pop_size = 66000000 # 66Millions ; real population size for the country
        
    # INITIALIZE THE PARAMETERS DICTIONARY
    params = numba.typed.Dict() # initialize the Dictionary
    params['n'] = n
    
    """n_ages -> int: Number of ages (0-100)"""
    n_ages = 101
    params['n_ages'] = float(n_ages)
    """seed -> int: Seed for random draws; arbitrary, just to uniform the simulations"""
    seed = 0
    params['seed'] = float(seed)
    np.random.seed(seed)
    
    if country== 'Italy':
        d0 = d0 # first case was registered on 23 January 2020
        d_lockdown = date(2020, 3, 8) # lockdown
        d_end = date(2020, 4, 15) # stop
        d_stay_home_start = date(3000, 3, 8) 
    elif country == 'Spain':
        d0 = d0 # first case was registered on 31 January 2020
        d_lockdown = date(2020, 3, 14) 
        d_end = date(2020, 4, 15) # stop
        d_stay_home_start = date(3000, 3, 8) 
    elif country == 'Germany':
        d0 = d0 # first case was registered on 27 January 2020 
        d_lockdown = date(2020, 3, 13)
        d_end = date(2020, 4, 15) # stop
        d_stay_home_start = date(3000, 3, 8) 
    elif country == 'France':
        d0 = d0 # first case was registered on 24 January 2020 
        d_lockdown = date(2020, 3, 16)
        d_end = date(2020, 4, 15) # stop
        d_stay_home_start = date(3000, 3, 8) 
     
    # PARSE REAL DATA FOR COMPUTING MSE DURING SWEEP 
    # Note: validation data are rescaled
    data = pd.read_csv('validation_data/Experiments/%s_deaths.csv' %country) # read validation dataset
    dates = [] 
    actual_deaths = []
    for i in range(len(data)):
        dates.append(pd.to_datetime(data['Date'][i],format='%Y-%m-%d').date()) # append the dates information 
        #actual_deaths.append(((data['Deaths'][i])*n)/real_pop_size) # append the deaths scaled by n 
        actual_deaths.append(((data['Deaths'][i])/real_pop_size)*100000) # real mortality rate / 100k people 

    time_from_d0 = []
    for i in range(len(dates)):
        time_from_d0.append((dates[i] - d0).days) # compute the time from d0 up to the simulated date
        
    """t_stayhome_start -> float: Number of days since stay home"""
    params['t_stayhome_start'] = float((d_stay_home_start - d0).days)
    
    """fraction_stay_home -> array: Fraction of stay home by age"""
    fraction_stay_home = np.zeros(n_ages)
    fraction_stay_home[:] = 0
    
    """T -> int: Number (total) of timesteps"""
    params['T'] = float((d_end - d0).days + 1)
    
    """initial_infected_fraction -> set the initial fraction of infected people as = 5/n for all countries"""
    # write the IFs statements for eventual subsequent changes 
    if country == 'Italy':
        params['initial_infected_fraction'] = 5./params['n']
    elif country == 'Spain':
        params['initial_infected_fraction'] = 5./params['n']
    elif country == 'Germany':
        params['initial_infected_fraction'] = 5./params['n']
    elif country == 'France':
        params['initial_infected_fraction'] = 5./params['n']
    
    """t_lockdown -> float: Number of days since lockdown"""
    params['t_lockdown'] = float((d_lockdown - d0).days)
    
    """ factor -> factor by which contacts are reduced after lockdown; factor that shapes the severity of the lockdown was
    The factor is set looking at the after factor imposition on the overall contact matrix in the country. 
    In addition, to get the overall qualitatively assessment across the country, it is used the Stringency Index information """
    factor = 1 # initialize the factor 
    if country == 'Italy':
        factor = 20.0
    elif country == 'Spain':
        factor = 10.0
    elif country == 'Germany':
        factor = 10.0
    elif country == 'France':
        factor = 10.0 
    params['lockdown_factor'] = factor
    # all age groups contacts are reduced by the same factor after lockdown
    lockdown_factor_age = ((0, 14, factor), (15, 24, factor), (25, 39, factor), (40, 69, factor), (70, 100, factor))
    lockdown_factor_age = np.array(lockdown_factor_age)
    
    params['mean_time_to_isolate_asympt'] = 10000 # assume that asymptomatic individuals do not isolate 
    """mean_time_to_isolate -> float: Time from symptom onset to isolation ; Source: https://www.nejm.org/doi/pdf/10.1056/NEJMoa2001316?articleTools=true"""
    params['mean_time_to_isolate'] = 4.6
    """asymptomatic_transmissibility -> float: How infectious are asymptomatic cases relative to symptomatic ones; Source: https://science.sciencemag.org/content/early/2020/03/13/science.abb3221"""
    params['asymptomatic_transmissibility'] = 0.55
    
    """p_infect_given_contact -> float: Probability of infection given contact between two individuals"""
    if country == 'China' or country == 'Republic of Korea':
        params['p_infect_given_contact'] = 0.020
    else: 
        # iterate over simulation combo parameters 
        params['p_infect_given_contact'] = p_inf
    
    """mortality_multiplier -> float: increase probability of death for all ages and comorbidities by this amount"""
    if country == 'Italy':
        params['mortality_multiplier'] = dmult
    elif country == 'Spain':
        params['mortality_multiplier'] = dmult
    elif country == 'Germany':
        params['mortality_multiplier'] = dmult
    elif country == 'France':
        params['mortality_multiplier'] = dmult
    
    """contact_tracing,p_trace_household,p_trace_outside -> Whether contact tracing happens, and if so the probability of successfully identifying each within and between household infected individual"""
    params['contact_tracing'] = float(False) # whether to do contact tracing or not 
    params['p_trace_household'] = 1 # probability of doing tracing in-household
    params['p_trace_outside'] = 0.8 # probability of doing tracing out-household
    d_tracing_start = date(2020, 2, 28) 
    params['t_tracing_start'] = float((d_tracing_start - d0).days)
    
    """cumulative_documentation_mild -> Target for the cumulative fraction of mild cases which never become severe that 
    are documnted. This is used to calibrate p_documented_in_mild, which is the per-day probability"""
    cumulative_documentation_mild = 0.00000000001
    
    """mean_time_to_isolate_factor -> Factor representing the mean time that takes to isolate individuals"""
    mean_time_to_isolate_factor = ((0, 14, 1), (14, 24, 1), (25, 39, 1), (40, 69, 1), (70, 100, 1))
    mean_time_to_isolate_factor = np.array(mean_time_to_isolate_factor)
    
    """p_documented_in_mild -> Probability of documenting the infection if it is still in a Mild State"""
    params['p_documented_in_mild'] = 0.2
    
    #############################
    """SET TRANSITION TIMES BETWEEN STATES ; Assume that all state transitions are exponentially distributed"""
    
    """mean_time_to_severe; Source: https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf"""
    params['mean_time_to_severe'] = 7.
    params['mean_time_mild_recovery'] = 14.
    """p_documented_in_mild -> Probability of documentation (outside of contact tracing) for a mild case which never becomes severe """
    params['p_documented_in_mild'] = calibrate_p_document_mild(cumulative_documentation_mild, country, None, params['mean_time_mild_recovery'], None)
    
    #Time from illness to mechanical ventilation is 14.5 days from (Source: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext)
    # There are subtracted off the 7 to get to critical.This gives a mean time to critical of 7.5
    params['mean_time_to_critical'] = 7.5
    
    #Based on WHO estimates there are used 4 weeks as mean for severe and 5 weeks as mean for critical
    params['mean_time_severe_recovery'] = 28. - params['mean_time_to_severe']
    params['mean_time_critical_recovery'] = 35. - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
    
    #Median time onset to death from (Source: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext)
    params['mean_time_to_death'] = 18.5 - params['mean_time_to_severe'] - params['mean_time_to_critical'] 
    
    #probability of exposed individual becoming infected each time step (Source: https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported)
    params['time_to_activation_mean'] = 1.621
    params['time_to_activation_std'] = 0.418
    
    # use get_p_infect_household function from Global_Parameters Script 
    p_infect_household = get_p_infect_household(int(params['n']), 4.6, params['time_to_activation_mean'], params['time_to_activation_std'], params['asymptomatic_transmissibility'])
    
    """overall_p_critical_death -> float: Probability that a critical individual dies. Indicates how many individuals end up in critical state; Source: http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51"""
    overall_p_critical_death = 0.49
    
    ######################
    """3. CONTACT MATRICES"""
    
    """Construct CONTACT MATRICES
    Idea: based on his/her age, each individuals has a different probability
          of contacting other individuals depending on their age
    Goal: construct contact_matrix, which states that an individual of age i
         contacts Poission(contact[i][j]) contacts with individuals of age j"""
    
    """contact_matrix_age_groups_dict -> dict: Mapping from interval names to age ranges."""
    contact_matrix_age_groups_dict = {
        'infected_1': '0-4', 'contact_1': '0-4', 'infected_2': '5-9',
        'contact_2': '5-9', 'infected_3': '10-14', 'contact_3': '10-14',
        'infected_4': '15-19', 'contact_4': '15-19', 'infected_5': '20-24',
        'contact_5': '20-24', 'infected_6': '25-29', 'contact_6': '25-29',
        'infected_7': '30-34', 'contact_7': '30-34', 'infected_8': '35-39',
        'contact_8': '35-39', 'infected_9': '40-44', 'contact_9': '40-44',
        'infected_10': '45-49', 'contact_10': '45-49', 'infected_11': '50-54',
        'contact_11': '50-54', 'infected_12': '55-59', 'contact_12': '55-59',
        'infected_13': '60-64', 'contact_13': '60-64', 'infected_14': '65-69',
        'contact_14': '65-69', 'infected_15': '70-74', 'contact_15': '70-74',
        'infected_16': '75-79', 'contact_16': '75-79'}
    
    # Define a read_contact_matrix(country) function
    def read_contact_matrix(country):
        """Create a country-specific contact matrix from stored data.
    
        Read a stored contact matrix based on age intervals. Return a matrix of
        expected number of contacts for each pair of raw ages. Extrapolate to age
        ranges that are not covered.
        Args -> country (str): country name.
        Returns -> float n_ages x n_ages matrix: expected number of contacts between of a person
                  of age i and age j is Poisson(matrix[i][j])."""
        matrix = np.zeros((n_ages, n_ages))
        # open the contact matrix
        with open('Contact_Matrices/{}/All_{}.csv'.format(country, country), 'r') as f:
            csvraw = list(csv.reader(f)) # read all the csv file by row: line by line 
        col_headers = csvraw[0][1:-1] # retrieve columns headers
        row_headers = [row[0] for row in csvraw[1:]] # retrieve row headers 
        data = np.array([row[1:-1] for row in csvraw[1:]]) # retrieve contact data 
        for i in range(len(row_headers)):
            for j in range(len(col_headers)):
                interval_infected = contact_matrix_age_groups_dict[row_headers[i]]
                interval_infected = [int(x) for x in interval_infected.split('-')]
                interval_contact = contact_matrix_age_groups_dict[col_headers[j]]
                interval_contact = [int(x) for x in interval_contact.split('-')]
                for age_infected in range(interval_infected[0], interval_infected[1]+1):
                    for age_contact in range(interval_contact[0], interval_contact[1]+1):
                        matrix[age_infected, age_contact] = float(data[i][j])/(interval_contact[1] - interval_contact[0] + 1)
    
        # extrapolate from 79yo out to 100yo
        # start by fixing the age of the infected person and then assuming linear decrease
        # in their number of contacts of a given age, following the slope of the largest
        # pair of age brackets that doesn't contain a diagonal term (since those are anomalously high)
        for i in range(interval_infected[1]+1):
            if i < 65: # 0-65
                slope = (matrix[i, 70] - matrix[i, 75])/5
            elif i < 70: # 65-70
                slope = (matrix[i, 55] - matrix[i, 60])/5
            elif i < 75: # 70-75
                slope = (matrix[i, 60] - matrix[i, 65])/5
            else: # 75-80
                slope = (matrix[i, 65] - matrix[i, 70])/5
    
            start_age = 79
            if i >= 75:
                start_age = 70
            for j in range(interval_contact[1]+1, n_ages):
                matrix[i, j] = matrix[i, start_age] - slope*(j - start_age)
                if matrix[i, j] < 0:
                    matrix[i, j] = 0
    
        # fix diagonal terms
        for i in range(interval_infected[1]+1, n_ages):
            matrix[i] = matrix[interval_infected[1]]
        for i in range(int((100-80)/5)):
            age = 80 + i*5
            matrix[age:age+5, age:age+5] = matrix[79, 79]
            matrix[age:age+5, 75:80] = matrix[75, 70]
        matrix[100, 95:] = matrix[79, 79]
        matrix[95:, 100] = matrix[79, 79]
    
        return matrix
    
    """contact_matrix -> n_ages x n_ages matrix: expected number of contacts between of a person
        of age i and age j is Poisson(matrix[i][j])."""
    contact_matrix = read_contact_matrix(country)
    
    ######################
    """4. SET TRANSITION PROBABILITIES BETWEEN DISEASE SEVERITIES"""
    
    """Construct TRANSITION PROBABILITIES BETWEEN DISEASE SEVERITIES
    There are three disease states: MILD, SEVERE and CRITICAL.
    - Mild represents sub-hospitalization.
    - Severe is hospitalization.
    - Critical is ICU.
    
    The key results of this section are:
    - p_mild_severe: n_ages x 2 x 2 matrix. For each age and comorbidity state
        (length two bool vector indicating whether the individual has diabetes and/or
        hypertension), what is the probability of the individual transitioning from
        the mild to severe state.
    - p_severe_critical, p_critical_death are the same for the other state transitions.
    
    All of these probabilities are proportional to the base progression rate
    for an (age, diabetes, hypertension) state which is stored in p_death_target
    and estimated via logistic regression.
    """
    
    """
    p_mild_severe_verity -> n_ages vector: The probability of transitioning from the mild to
        severe state for a patient of age i is p_mild_severe_verity[i];
    Overall probability of progressing to severe infection (hospitalization) for each age
    Using the estimates from Verity et al; Source: https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
    ** Page 675
    """
    p_mild_severe_verity = np.zeros(n_ages)
    p_mild_severe_verity[0:10] = 0
    p_mild_severe_verity[10:20] = 0.0408
    p_mild_severe_verity[20:30] = 1.04
    p_mild_severe_verity[30:40] = 3.43
    p_mild_severe_verity[40:50] = 4.25
    p_mild_severe_verity[50:60] = 8.16
    p_mild_severe_verity[60:70] = 11.8
    p_mild_severe_verity[70:80] = 16.6
    p_mild_severe_verity[80:] = 18.4
    p_mild_severe_verity = p_mild_severe_verity/100
    
    """
    overall_p_severe_critical -> n_ages vector: Overall probability of get severe infection 
    Overall probability of ICU admission given severe infection for each age.
    Using p(ICU)/p(ICU or hospitalized) from CDC, taking midpoint of intervals
    https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
    ** Table: Hospitalization, intensive care unit (ICU) admission, and case–fatality percentages for reported COVID–19 cases, by age group —United States, February 12–March 16, 2020
    """
    icu_cdc = np.zeros(n_ages)
    hosp_cdc = np.zeros(n_ages)
    icu_cdc[:20] = 0
    icu_cdc[20:45] = (2 + 4.2)/2
    icu_cdc[45:55] = (5.4 + 10.4)/2
    icu_cdc[55:65] = (4.7 + 11.2)/2
    icu_cdc[65:75] = (8.1 + 18.8)/2
    icu_cdc[75:] = (10.5+31.0	)/2
    icu_cdc = icu_cdc/100
    
    hosp_cdc[:20] = 0
    hosp_cdc[20:45] = (14.3 + 20.8)/2
    hosp_cdc[45:55] = (21.2 + 28.3)/2
    hosp_cdc[55:65] = (20.5 + 30.1)/2
    hosp_cdc[65:75] = (28.6 + 43.5)/2
    hosp_cdc[75:] = (30.5 + 58.7)/2
    hosp_cdc = hosp_cdc/100
    overall_p_severe_critical = icu_cdc/(icu_cdc + hosp_cdc)
    overall_p_severe_critical[:20] = 0
                
    # Find severe-critical and critical-deaths multiplies wrt mild-severe
    severe_critical_multiplier = overall_p_severe_critical / p_mild_severe_verity
    critical_death_multiplier = overall_p_critical_death / p_mild_severe_verity
    severe_critical_multiplier[:20] = 1
    critical_death_multiplier[:20] = 1
    
    # get the overall CFR for each age/comorbidity combination by running the logistic model
    """
    Mortality model. We fit a logistic regression to estimate p_mild_death from (age, diabetes, hypertension).
    The results of the logistic regression are used to set the disease severity
    transition probabilities.
    """
    """float vector: Logistic regression weights for each age bracket."""
    c_age = np.loadtxt('c_age.txt', delimiter=',').mean(axis=0)
    """float: Logistic regression weight for diabetes."""
    c_diabetes = np.loadtxt('c_diabetes.txt', delimiter=',').mean(axis=0)
    """float: Logistic regression weight for hypertension."""
    c_hyper = np.loadtxt('c_hypertension.txt', delimiter=',').mean(axis=0)
    intervals = np.loadtxt('comorbidity_age_intervals.txt', delimiter=',')
    
    def age_to_interval(i):
        """Return the corresponding comorbidity age interval for a specific age.
        Args -> i (int): age.
        Returns -> int: index of interval containing i in intervals."""
        for idx, a in enumerate(intervals):
            if i >= a[0] and i < a[1]:
                return idx
        return idx
    
    p_death_target = np.zeros((n_ages, 2, 2))
    for i in range(n_ages):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                if i < intervals[0][0]:
                    p_death_target[i, diabetes_state, hyper_state] = 0
                else:
                    # Apply logistic-sigmoid function
                    # .expit(x): x ndarray to expit to element-wise 
                    p_death_target[i, diabetes_state, hyper_state] = scipy.special.expit(
                        c_age[age_to_interval(i)] + diabetes_state * c_diabetes +
                        hyper_state * c_hyper)
    
    #calibrate the probability of the severe -> critical transition to match the overall CFR for each age/comorbidity combination
    #age group, diabetes (0/1), hypertension (0/1)
    progression_rate = np.zeros((n_ages, 2, 2))
    """p_mild_severe -> float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from mild to severe state."""
    p_mild_severe = np.zeros((n_ages, 2, 2))
    """p_severe_critical -> float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from severe to critical state."""
    p_severe_critical = np.zeros((n_ages, 2, 2))
    """p_critical_death -> float n_ages x 2 x 2 vector: Probability a patient with a particular age combordity
        profile transitions from critical to dead state."""
    p_critical_death = np.zeros((n_ages, 2, 2))
    
    for i in range(n_ages):
        for diabetes_state in [0,1]:
            for hyper_state in [0,1]:
                # Pm_s(a,c) = (Pm_d(a,c)/Gamma(s_c)*Gamma(c_d))**1/3
                progression_rate[i, diabetes_state, hyper_state] = (p_death_target[i, diabetes_state, hyper_state]
                                                                    / (severe_critical_multiplier[i]
                                                                       * critical_death_multiplier[i])) ** (1./3)
                p_mild_severe[i, diabetes_state, hyper_state] = progression_rate[i, diabetes_state, hyper_state]
                # Ps_c(a,c) = Gamma(s_c)*Pm_s(a,c)
                p_severe_critical[i, diabetes_state, hyper_state] = severe_critical_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
                # Pc_d(a,c) = Gamma(c_d)*Pm_s(a,c)
                p_critical_death[i, diabetes_state, hyper_state] = critical_death_multiplier[i]*progression_rate[i, diabetes_state, hyper_state]
    
    # Assume no critical cases under 20; set Pc_d and Ps_c = 0 for Under20
    p_critical_death[:20] = 0
    p_severe_critical[:20] = 0

    #for now, just cap 80+yos with diabetes and hypertension
    p_critical_death[p_critical_death > 1] = 1
    
    #Scale up all transitions proportional to the mortality_multiplier parameter
    p_mild_severe *= params['mortality_multiplier']**(1/3)
    p_severe_critical *= params['mortality_multiplier']**(1/3)
    p_critical_death *= params['mortality_multiplier']**(1/3)
    p_mild_severe[p_mild_severe > 1] = 1
    p_severe_critical[p_severe_critical > 1] = 1
    p_critical_death[p_critical_death > 1] = 1
    
    ######################
    """5. RUN THE SIMULATION AND SAVE OUT FINE-GRAINED RESULTS"""
    
    num_runs = 100 # SET the number of runs per simulation combo 
    n = int(params['n']) # retrieve population size 
    T = int(params['T']) # retrieve number of timesteps 
    all_c = []
    all_s = []
    all_d = []
    
    # n x 1 datatypes
    r0_total = np.zeros((num_runs,1))
    mse_list = np.zeros((num_runs,1))
    
    # n x T datatypes 
    S_per_time = np.zeros((num_runs, T))
    E_per_time = np.zeros((num_runs, T))
    D_per_time = np.zeros((num_runs, T))
    
    Mild_per_time = np.zeros((num_runs, T))
    Severe_per_time = np.zeros((num_runs, T))
    Critical_per_time = np.zeros((num_runs, T))
    R_per_time = np.zeros((num_runs, T))
    Q_per_time = np.zeros((num_runs, T))
    
    r_0_over_time = np.zeros((num_runs, int(params['T'])))
    cfr_over_time = np.zeros((num_runs, int(params['T'])))
    fraction_over_70_time = np.zeros((num_runs, int(params['T'])))
    fraction_below_30_time = np.zeros((num_runs, int(params['T'])))
    median_age_time = np.zeros((num_runs, int(params['T'])))
    total_infections_time = np.zeros((num_runs, int(params['T'])))
    total_documented_time = np.zeros((num_runs, int(params['T'])))
    dead_by_age = np.zeros((num_runs, n_ages))
    total_deaths_time = np.zeros((num_runs, int(params['T'])))
    all_final_s = np.zeros((num_runs, int(params['n'])))
    
    # n x n_age_groups x T datatypes
    age_groups = ((0, 14), (15, 24), (25, 39), (40, 69), (70, 100))
    infected_by_age_by_time = np.zeros((num_runs, len(age_groups), T))
    total_age_by_time = np.zeros((num_runs, len(age_groups), T))
    CFR_by_age_by_time = np.zeros((num_runs, len(age_groups), T))
    dead_by_age_by_time = np.zeros((num_runs, len(age_groups), T))
    
    for i in range(num_runs):
        # RUN the simulations of the SEIR ABModel
        #print(i)
        S, E, Mild, Documented, Severe, Critical, R, D, Q, num_infected_by, time_documented, \
        time_to_activation, time_to_death, time_to_recovery, time_critical, time_exposed, \
        num_infected_asympt, age, time_infected, time_to_severe \
            =  run_complete_simulation(seed + i ,country, contact_matrix, p_mild_severe, p_severe_critical, \
                                       p_critical_death, mean_time_to_isolate_factor, \
                                       lockdown_factor_age, p_infect_household, fraction_stay_home, params, load_population)
        S_per_time[i] = S.sum(axis=1)
        E_per_time[i] = E.sum(axis=1)
        D_per_time[i] = D.sum(axis=1)
    
        Mild_per_time[i] = Mild.sum(axis=1)
        Severe_per_time[i] = Severe.sum(axis=1)
        Critical_per_time[i] = Critical.sum(axis=1)
        R_per_time[i] = R.sum(axis=1)
        Q_per_time[i] = Q.sum(axis=1)
    
        r0_total[i] = float(num_infected_by[np.logical_and(time_exposed <= 20, time_exposed > 0)].mean())
        
        '''
        #This is all for analyzing the age distribution of infection, CFR over time, etc.
        for t in range(T):
            if (time_exposed == t).sum() > 0:
                r_0_over_time[i, t] = num_infected_by[time_exposed == t].mean()
                cfr_over_time[i, t] = D[-1, time_exposed == t].sum()/(D[-1, time_exposed == t].sum() + R[-1, time_exposed == t].sum())
                fraction_over_70_time[i, t] = (age[time_exposed == t] >= 70).mean()
                fraction_below_30_time[i, t] = (age[time_exposed == t] < 30).mean()
                median_age_time[i, t] = np.median(age[time_exposed == t])
        total_infections_time[i] = (params['n'] - S.sum(axis=1) - E.sum(axis=1))
        total_documented_time[i] = Documented.sum(axis=1)
        
        for idx, (lower, upper) in enumerate(age_groups):
                total = 0
                infected = 0
                age_group_array = np.logical_and(age >= lower, age <= upper)
    
                age_group_total = age_group_array.sum()
    
                age_group_susceptible_this_timestep = np.logical_and(S[t], age_group_array).sum()
                infected = age_group_total - age_group_susceptible_this_timestep
    
                infected_by_age_by_time[i, idx, t] = infected
                total_age_by_time[i, idx, t] = age_group_total
    
                d_end_per_age_per_timestep = np.logical_and(D[-1], time_exposed == t)
                d_end_per_age_per_timestep = np.logical_and(d_end_per_age_per_timestep, age_group_array).sum()
    
                r_end_per_age_per_timestep = np.logical_and(R[-1], time_exposed == t)
                r_end_per_age_per_timestep = np.logical_and(r_end_per_age_per_timestep, age_group_array).sum()
    
                CFR_by_age_by_time[i, idx, t] = d_end_per_age_per_timestep / (d_end_per_age_per_timestep + r_end_per_age_per_timestep)
    
                dead_by_age_by_time[i, idx, t] = np.logical_and(D[t], age_group_array).sum()
        '''
        for patient_age in range(n_ages):
            dead_by_age[i, patient_age] = D[-1, age == patient_age].sum()
        total_deaths_time[i] = D.sum(axis=1)
        all_final_s[i] = S[-1]
        
    f = time_from_d0[0] 
    l = time_from_d0[-1] + 1
    
    for i,D in enumerate(D_per_time):
        #mse = mean_squared_error(D[f:l], actual_deaths) # MSE over n population
        mse = mean_squared_error((D[f:l]/n)*100000, actual_deaths) # MSE evaluated on mortality rate / 100k individuals 
        mse_list[i] = float(mse)
    #print(mse_list)
    
    fname = '%s_n%s_p%s_m%s_s%s'%(country, params['n'], params['p_infect_given_contact'], params['mortality_multiplier'], d0.day)
    fname += '_%s.csv'
    
    datatypes_n_1 = [
        {
            'name':'r0_tot', 
            'data':r0_total
        },
        {
            'name':'mse',
            'data':mse_list
        }]
    
    datatypes_n_t = [
        {
            'name':'susceptible', 
            'data':S_per_time
        },
        {
            'name':'exposed', 
            'data':E_per_time
        },
        {
            'name':'deaths', 
            'data':D_per_time
        },
        {
            'name':'mild', 
            'data':Mild_per_time
        },
        {
            'name':'severe', 
            'data':Severe_per_time
        },
        {
            'name':'critical', 
            'data':Critical_per_time
        },
        {
            'name':'recovered', 
            'data':R_per_time
        },
        {
            'name':'quarantine', 
            'data':Q_per_time
        },
    
        #{
        #    'name':'r0_time', 
        #    'data':r_0_over_time
        #},
        #{
        #    'name':'cfr_time', 
        #    'data':cfr_over_time
        #},
        #{
        #    'name':'frac_over_70_time', 
        #    'data':fraction_over_70_time
        #},
        #{
        #    'name':'frac_below_30_time', 
        #    'data':fraction_below_30_time
        #},
        #{
        #    'name':'median_age_time', 
        #    'data':median_age_time
        #}
        ]
    
    # These will be the biggest storage burden. Cut where you can.
    """datatypes_n_a_t = [
        {
            'name':'infected_age_time', 
            'data':infected_by_age_by_time
        },
        {
            'name':'total_age_time',
            'data':total_age_by_time
        },
        {
            'name':'cfr_age_time',
            'data':CFR_by_age_by_time
        },
        {
            'name':'dead_age_time',
            'data':dead_by_age_by_time
        }]"""
    
    
    for d in datatypes_n_1:
        df = pd.DataFrame(d['data'])
        df.to_csv("./Paramsweep/%s/"%(country)+fname%d['name'],sep=',',index=False,header=False, na_rep='NA')
    
    for d in datatypes_n_t:
        df = pd.DataFrame(d['data'])
        df.to_csv("./Paramsweep/%s/"%(country)+fname%d['name'],sep=',',index=False,header=False, na_rep='NA')
    
    
    """for d in datatypes_n_a_t:
        # Flatten the data -- remember to unflatten for analysis
        d['data'] = d['data'].reshape(num_runs, len(age_groups)*T)
        df = pd.DataFrame(d['data'])
        df.to_csv("./Paramsweep/%s/"%(country)+fname%d['name'],sep=',',index=False,header=False, na_rep='NA')"""
    
    #pickle.dump((r_0_over_time, cfr_over_time, fraction_over_70_time, fraction_below_30_time, median_age_time, total_infections_time, total_documented_time, dead_by_age, total_deaths_time, all_final_s), open('results_{}_{}_{}.pickle'.format(country, params['p_infect_given_contact'], d0.day), 'wb'))
    print('Done with this combo simulation')
    print(f"D0:{d0}, PINF: {params['p_infect_given_contact']}, DMULT:{dmult}")
    print('################')

### PARALLELIZATION -- USING JOBLIB
country_list = ['Italy','Spain','Germany','France'] # pass the list of the countries of interest to the function and to the parallelization
for country in country_list:
    if country == 'Italy':
        d0_list = [date(2020, 1, 3),date(2020, 1, 10),date(2020, 1, 17), date(2020, 1, 19),date(2020, 1, 21),date(2020, 1, 23),date(2020, 2, 6)]
        p_inf_list = [0.014,0.015,0.016,0.017,0.018,0.019,0.020,0.021,0.022,0.024]
        dmult_list = [0.8,1,2,3,4]
        tot_params = list(product(d0_list,p_inf_list,dmult_list))
        tot_params_list = []
        for combo in tot_params: 
            combo = list(combo)
            combo += [country]
            tot_params_list.append(combo)
        #print(tot_params_list)
        #simulation(tot_params_list)
        Parallel(n_jobs=18)(delayed(simulation)(i) for i in tot_params_list)
    elif country == 'Spain':
        time.sleep(300)
        d0_list = [date(2020, 1, 11),date(2020, 1, 18),date(2020, 1, 25), date(2020, 1, 27),date(2020, 1, 29),date(2020, 1, 31),date(2020, 2, 10)]
        p_inf_list = [0.016,0.017,0.018,0.019,0.020,0.021,0.022,0.024,0.026,0.028]
        dmult_list = [0.8,1,2,3] 
        tot_params = list(product(d0_list,p_inf_list,dmult_list))
        tot_params_list = []
        for combo in tot_params: 
            combo = list(combo)
            combo += [country]
            tot_params_list.append(combo)
        #print(tot_params_list)
        #simulation(tot_params_list)
        Parallel(n_jobs=18)(delayed(simulation)(i) for i in tot_params_list)
    elif country == 'Germany':
        time.sleep(300)
        d0_list = [date(2020, 1, 7),date(2020, 1, 14),date(2020, 1, 21),date(2020, 1, 23),date(2020, 1, 25), date(2020, 1, 27),date(2020,2,10)]
        p_inf_list = [0.021,0.022,0.024,0.026,0.028,0.029,0.030,0.031,0.032,0.033]
        dmult_list = [0.8,1,2,3]
        tot_params = list(product(d0_list,p_inf_list,dmult_list))
        tot_params_list = []
        for combo in tot_params: 
            combo = list(combo)
            combo += [country]
            tot_params_list.append(combo)
        #print(tot_params_list)
        #simulation(tot_params_list)
        Parallel(n_jobs=18)(delayed(simulation)(i) for i in tot_params_list)
    elif country == 'France':
        time.sleep(300)
        d0_list = [date(2020, 1, 4),date(2020, 1, 11),date(2020, 1, 18),date(2020, 1, 20),date(2020, 1, 22), date(2020, 1, 24),date(2020, 2, 7)]
        p_inf_list =[0.014,0.015,0.016,0.017,0.018,0.019,0.020,0.021,0.022,0.024]
        dmult_list = [0.8,1,2,3] 
        tot_params = list(product(d0_list,p_inf_list,dmult_list))
        tot_params_list = []
        for combo in tot_params: 
            combo = list(combo)
            combo += [country]
            tot_params_list.append(combo)
        #print(tot_params_list)
        #simulation(tot_params_list)
        Parallel(n_jobs=18)(delayed(simulation)(i) for i in tot_params_list)
            
            