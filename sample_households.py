## SAMPLE POPULATION 

'''
Script uses the *World_Age_2019.csv* and *AgeSpecificFertility.csv* -built using the *AgeDistribution_FertilityAgeDistribution* script- dataset in order to get the age distribution for each country of interest and then uses it as well as additional data on the country-specific household composition to build the household structure for all the countries of interest.   
The Script comprehends country-specific functions **sample_househol_country(n)** which, given an input of population size *n* outputs the household composition for that country population.
'''

# import libraries
import numpy as np
import pandas as pd
import csv
import math
import os 

# set the base directory
b_dir = './Population Datasets' # define the source directory 

def get_age_distribution(country):
    """Get the population age distribution for each specific country
    INPUT: string indicating country name"""
    age_distribution=[]
    with open(os.path.join(b_dir,'World_Age_2019.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]==country:
                for i in range(101):
                    age_distribution.append(float(row[i+1]))
                break
    return np.array(age_distribution)

def get_mother_birth_age_distribution(country):
    """Get the age for the mothers for each country
    INPUT: string indicating country name """
    mother_birth_age_distribution=[]
    with open(os.path.join(b_dir,'AgeSpecificFertility.csv'),encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]==country:
                #15-19; 20-24; 25-29; 30-34; 35-39; 40-44; 45-49
                for i in range(7):
                    mother_birth_age_distribution.append(float(row[i+1]))
                break
    return np.array(mother_birth_age_distribution)

###################################
##### ITALY HOUSEHOLD_AGE 
def sample_households_italy(n):
    max_household_size = 6
    
    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    
    age = np.zeros(n, dtype=np.int)    
    n_ages = 101
        
    # Age distribution in Italy
    age_distribution = get_age_distribution("Italy")
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
        
    # List of household types
    # single household, couple without children, single parent +1/2/3 children, 
    # couple +1/2/3 children, family without a nucleus, nucleus with other persons, 
    # households with two or more nuclei (a and b)
    household_probs = np.array([0.309179, 0.196000, 0.0694283, 0.0273065, 0.00450268, 0.152655, 0.132429, 0.0200969, 
                       0.049821, 0.033, 0.017])
    household_probs /= household_probs.sum()
    households_tot = np.zeros(len(household_probs), dtype=np.int)
    
    # Keeping track of the number of agents
    num_generated = 0
    
    # Age of the mother at first birth, as obtained from fertility data
    mother_birth_age_distribution = get_mother_birth_age_distribution("Italy")    
    renormalized_mother = mother_birth_age_distribution/mother_birth_age_distribution.sum()
    
    renormalized_adult = age_distribution[18:]
    renormalized_adult = renormalized_adult/renormalized_adult.sum()
    
    # Age = 30 considered as the time when children leave the family home
    # Note: older children in Italy often live with their parents longer than elsewhere
    adult_child_random = 30 + np.random.randint(-1,4)
    renormalized_child = age_distribution[:adult_child_random]
    renormalized_child = renormalized_child/renormalized_child.sum()
    renormalized_adult_older = age_distribution[adult_child_random:]
    renormalized_adult_older /= renormalized_adult_older.sum()
    # Age = 60 considered as retirement threshold (as a first approximation; it could potentially be larger)
    age_grandparent_random = 60 + np.random.randint(0,4)
    renormalized_grandparent = age_distribution[age_grandparent_random:]
    renormalized_grandparent = renormalized_grandparent/renormalized_grandparent.sum()
    
    while num_generated < n:
        if n - num_generated < (max_household_size+1):
            i = 0
            households_tot[i] += 1
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        # Single person household
        if i == 0:
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult)
            age[num_generated]= np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 1
        # Couple with one of the two being 3 years older
        elif i == 1:  
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age_adult = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated] = age_adult
            # For heterosexual couples, the man is older than the woman on average
            age[num_generated+1] = min(n_ages-1,age_adult+np.random.randint(-2,4))
            generated_this_step = 2
        # Single parent + 1 child
        elif i == 2: 
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random, p=renormalized_child)
            age[num_generated] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            generated_this_step = 2
        # Single parent + 2 children
        elif i == 3:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            generated_this_step = 3
        # Single parent + 3 children
        elif i == 4:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            age[num_generated + 3] = mother_current_age
            generated_this_step = 4
            
        # Couple with one of the two being 3 years older + 1 child
        elif i == 5:
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random, p=renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 3
        
        # Couple with one of the two being 3 years older + 2 children
        elif i == 6:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 4            
        
        # Couple with one of the two being 3 years older + 3 children
        elif i == 7:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            # Populate age for parents
            age[num_generated + 3] = mother_current_age
            age[num_generated + 4] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 5
        
        # Family without nucleus (2 adults >= 30)
        elif i == 8:
            households_tot[i] += 1
            age[num_generated] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated+1] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 2         
                
        # Nucleus with other persons (couple with one of the two being three years older + 2 children + 1 adult >= 60)
        elif i == 9:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Populate age for adult >= 60
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)
            generated_this_step = 5
            
        # Households with 2 or more nuclei
        # Assumption: couple with one of the two being three years older + 2 children <= 30 + 2 grand-parents
        
        elif i == 10:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random, p=renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Grand-parents
            grandmother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            grandmother_current_age = min(n_ages-1,grandmother_age_at_birth + mother_current_age)
            # Populate age for grand-parents
            age[num_generated + 4] = grandmother_current_age
            age[num_generated + 5] = min(n_ages-1,grandmother_current_age+np.random.randint(-2,2))   
            generated_this_step = 6
            
        # Update list of household contacts accordingly 
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        
    return households, age, households_tot

#households, age, households_tot = sample_households_italy(1000000)

###################################
##### SPAIN HOUSEHOLD_AGE
def sample_households_spain(n):
    max_household_size = 6
    
    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    
    age = np.zeros(n, dtype=np.int)    
    n_ages = 101
        
    # Age distribution in Spain
    age_distribution = get_age_distribution("Spain")
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
    
    ##!! HOUSEHOLDS DISTRIBUTION: https://www.ine.es/en/prensa/ech_2019_en.pdf
    # List of household types
    # 0) single household
    # 1) couple without children
    # 2) single parent + child/children, 
    # 3) 4) 5) couple + 1/2/3 children, 
    # 6) family without a nucleus
    # 7) nucleus with other persons, 
    # 8) households with two or more nuclei (a and b)
    household_probs = np.array([0.257, 0.211 , 0.101, 0.157, 0.148, 0.030, 0.030, 0.043, 0.023])
    household_probs /= household_probs.sum()
    households_tot = np.zeros(len(household_probs), dtype=np.int)
    
    # Keeping track of the number of agents
    num_generated = 0
    
    #https://www.worldometers.info/demographics/spain-demographics/
    #solve for probability of 1 child to get average 1.3 children per woman
    p_one_child = 1.3/3
    
    # Age of the mother at first birth, as obtained from fertility data
    mother_birth_age_distribution=get_mother_birth_age_distribution("Spain")    
    renormalized_mother = mother_birth_age_distribution/mother_birth_age_distribution.sum()
    
    ## ADULT = AGED >= 18
    renormalized_adult = age_distribution[18:]
    renormalized_adult = renormalized_adult/renormalized_adult.sum()
    
    ## CHILD LIVING WITH FAMILITY = AGED <= 30 
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    adult_child_random = 30 + np.random.randint(-3,4)
    renormalized_child = age_distribution[:adult_child_random]
    renormalized_child = renormalized_child/renormalized_child.sum()
    
    ## ADULT NOT LIVING WITH FAMILY = AGED 30 +
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    renormalized_adult_older = age_distribution[adult_child_random:]
    renormalized_adult_older /= renormalized_adult_older.sum()
    
    ## RETIREMENT AGE = AGED 65 + 
    # Age = 65 considered as retirement threshold (as a first approximation; it could potentially be larger)
    age_grandparent_random = 65 + np.random.randint(1,4)
    renormalized_grandparent = age_distribution[age_grandparent_random:]
    renormalized_grandparent = renormalized_grandparent/renormalized_grandparent.sum()
    
    while num_generated < n:
        if n - num_generated < (max_household_size+1):
            i = 0
            households_tot[i] += 1
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        # Single person household
        if i == 0:
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age[num_generated]=np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 1
            
        # Couple with one of the two being 3 years older and no children
        elif i == 1:  
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age_adult = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated] = age_adult
            # For heterosexual couples, the man is older than the woman on average, let the age diff be random between 0 and 3 years
            age[num_generated+1] = min(n_ages-1,age_adult+np.random.randint(-2,4))
            generated_this_step = 2
            
        # Single parent + child/children
        elif i == 2:   
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            generated_this_step = 2
            # Generate another child younger than the first
            if np.random.rand() < 1 - p_one_child and child_age > 0:
                offset = min((5, child_age))
                renormalized = age_distribution[child_age-offset:child_age]
                renormalized = renormalized/renormalized.sum()
                child_age = np.random.choice(offset, p=renormalized) + child_age
                age[num_generated+generated_this_step] = child_age
                generated_this_step += 1    
                
        # Couple with one of the two being 3 years older + 1 child
        elif i == 3: 
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 3
        
        # Couple with one of the two being 3 years older + 2 children
        elif i == 4:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 4            
        
        # Couple with one of the two being 3 years older + 3 children
        elif i == 5:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            # Populate age for parents
            age[num_generated + 3] = mother_current_age
            age[num_generated + 4] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 5
        
        # Family without nucleus (2 adults >= 30)
        elif i == 6:
            households_tot[i] += 1
            age[num_generated] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated+1] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 2         
                
        # Nucleus with other persons (couple with one of the two being older + 2 children + 1 adult in retirement)
        elif i == 7:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Populate age for adult >= 60
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)
            generated_this_step = 5
            
        # Households with 2 or more nuclei
        # Assumption: couple with one of the two older + 2 children + 2 grand-parents
        
        elif i == 8:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Grand-parents
            grandmother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            grandmother_current_age = min(n_ages-1,grandmother_age_at_birth + mother_current_age)
            # Populate age for grand-parents
            age[num_generated + 4] = grandmother_current_age
            age[num_generated + 5] = min(n_ages-1,grandmother_current_age+np.random.randint(-1,2)) 
            generated_this_step = 6
            
        # Update list of household contacts accordingly 
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        
    return households, age, households_tot

# households, age = sample_households_spain(1000000)

###################################
##### GERMANY HOUSEHOLD_AGE
def sample_households_germany(n):
    max_household_size = 5
    
    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    
    age = np.zeros(n, dtype=np.int)    
    n_ages = 101
        
    # Age distribution in Spain
    age_distribution = get_age_distribution("Germany")
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
    
    ##!! HOUSEHOLDS DISTRIBUTION:
    # 0) Single Household
    # 1) Couples Without Children
    # 2) 3) 4) Single Parent + 1/2/3 Children
    # 5) 6) 7) Couple + 1/2/3 Children
    # 8) Other Household No Children
    # 9) 10) 11) Other Household + 1/2/3 Children
    household_probs = np.array([0.4184, 0.2828, 0.0226, 0.0103, 0.0028, 0.0684, 0.0665, 0.0222, 0.0780, 0.0192, 0.0061, 0.0021])
    household_probs /= household_probs.sum()
    households_tot = np.zeros(len(household_probs), dtype=np.int)
    
    # Keeping track of the number of agents
    num_generated = 0
    
    # Age of the mother at first birth, as obtained from fertility data
    mother_birth_age_distribution=get_mother_birth_age_distribution("Germany")    
    renormalized_mother = mother_birth_age_distribution/mother_birth_age_distribution.sum()
    
    ## ADULT = AGED >= 18
    renormalized_adult = age_distribution[18:]
    renormalized_adult = renormalized_adult/renormalized_adult.sum()
    
    ## CHILD LIVING WITH FAMILITY = AGED <= 30 
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    adult_child_random = 24 + np.random.randint(-1,3)
    renormalized_child = age_distribution[:adult_child_random]
    renormalized_child = renormalized_child/renormalized_child.sum()
    
    ## ADULT NOT LIVING WITH FAMILY = AGED 30 +
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    renormalized_adult_older = age_distribution[adult_child_random:]
    renormalized_adult_older /= renormalized_adult_older.sum()
    
    ## RETIREMENT AGE = AGED 65 + 
    # https://ec.europa.eu/social/main.jsp?catId=1111&langId=en&intPageId=4554
    # Age = 65 considered as retirement threshold (as a first approximation; it could potentially be larger)
    age_grandparent_random = 65 + np.random.randint(1,4)
    renormalized_grandparent = age_distribution[age_grandparent_random:]
    renormalized_grandparent = renormalized_grandparent/renormalized_grandparent.sum()
    
    while num_generated < n:
        if n - num_generated < (max_household_size+1):
            i = 0
            households_tot[i] += 1
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        # Single person household
        if i == 0:
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age[num_generated]=np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 1
            
        # Couple with one of the two being 3 years older and no children
        elif i == 1:  
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age_adult = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated] = age_adult
            # For heterosexual couples, the man is older than the woman on average, let the age diff be random between 0 and 3 years
            age[num_generated+1] = min(n_ages-1,age_adult+np.random.randint(-2,4))
            generated_this_step = 2
            
        # Single parent + 1 child
        elif i == 2: 
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            generated_this_step = 2
            
        # Single parent + 2 children
        elif i == 3:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            generated_this_step = 3
            
        # Single parent + 3 children
        elif i == 4:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            age[num_generated + 3] = mother_current_age
            generated_this_step = 4  
                
        # Couple with one of the two being older + 1 child
        elif i == 5:
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 3
        
        # Couple with one of the two being older + 2 children
        elif i == 6:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 4            
        
        # Couple with one of the two being older + 3 children
        elif i == 7:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            # Populate age for parents
            age[num_generated + 3] = mother_current_age
            age[num_generated + 4] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 5
        
        # Other Household No Children 
        elif i == 8:
            households_tot[i] += 1
            age[num_generated] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated+1] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 2         
                
        # Nucleus with other persons (couple with one of the two being older + 1 children + 2 adults in retirement)
        elif i == 9:
            households_tot[i] += 1
            # Children
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Populate age for adult >= 60
            age[num_generated + 3] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)
            generated_this_step = 5
            
        # Nucleus with other persons (couple with one of the two being older + 2 children + 1 adults in retirement)
        elif i == 10:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Grand-parents
            grandmother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            grandmother_current_age = min(n_ages-1,grandmother_age_at_birth + mother_current_age)
            # Populate age for grand-parents
            age[num_generated + 4] = grandmother_current_age   
            generated_this_step = 5
        
        # Nucleus with other persons (single parent + 3 children + 1 adults in retirement)
        elif i == 11:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parent
            age[num_generated + 3] = mother_current_age
            # Grand-parent
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)
            generated_this_step = 5
            
        # Update list of household contacts accordingly 
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        
    return households, age, households_tot

#households, age, households_tot = sample_households_germany(1000000)

###################################
##### FRANCE HOUSEHOLD_AGE
def sample_households_france(n):
    max_household_size = 5
    
    households = np.zeros((n, max_household_size), dtype=np.int)
    households[:] = -1
    
    age = np.zeros(n, dtype=np.int)    
    n_ages = 101
        
    # Age distribution in France
    age_distribution = get_age_distribution("France")
    age_distribution = np.array(age_distribution)
    age_distribution = age_distribution/age_distribution.sum()
    
    ##!! HOUSEHOLDS DISTRIBUTION:
    # 0) Single Household
    # 1) Couples Without Children
    # 2) 3) 4) Single Parent + 1/2/3 Children
    # 5) 6) 7) Couple + 1/2/3 Children
    # 8) Other Household No Children
    # 9) 10) 11) Other Household + 1/2/3 Children

    household_probs = np.array([0.3622,0.2616, 0.0309, 0.0218, 0.0086, 0.0707,0.0909, 0.0410, 0.0768, 0.0217, 0.0086,0.0046])
    household_probs /= household_probs.sum()
    households_tot = np.zeros(len(household_probs), dtype=np.int)
    
    # Keeping track of the number of agents
    num_generated = 0
    
    # Age of the mother at first birth, as obtained from fertility data
    mother_birth_age_distribution=get_mother_birth_age_distribution("France")    
    renormalized_mother = mother_birth_age_distribution/mother_birth_age_distribution.sum()
    
    ## ADULT = AGED >= 18
    renormalized_adult = age_distribution[18:]
    renormalized_adult = renormalized_adult/renormalized_adult.sum()
    
    ## CHILD LIVING WITH FAMILITY = AGED <= 30 
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    adult_child_random = 24 + np.random.randint(-1,3)
    renormalized_child = age_distribution[:adult_child_random]
    renormalized_child = renormalized_child/renormalized_child.sum()
    
    ## ADULT NOT LIVING WITH FAMILY = AGED 30 +
    # https://ec.europa.eu/eurostat/web/products-eurostat-news/-/edn-20180515-1
    renormalized_adult_older = age_distribution[adult_child_random:]
    renormalized_adult_older /= renormalized_adult_older.sum()
    
    ## RETIREMENT AGE = AGED 65 + 
    # https://ec.europa.eu/social/main.jsp?catId=1111&langId=en&intPageId=4554
    # Age = 65 considered as retirement threshold (as a first approximation; it could potentially be larger)
    age_grandparent_random = 65 + np.random.randint(1,4)
    renormalized_grandparent = age_distribution[age_grandparent_random:]
    renormalized_grandparent = renormalized_grandparent/renormalized_grandparent.sum()
    
    while num_generated < n:
        if n - num_generated < (max_household_size+1):
            i = 0
            households_tot[i] += 1
        else:
            i = np.random.choice(household_probs.shape[0], p=household_probs)
        # Single person household
        if i == 0:
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age[num_generated]=np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 1
            
        # Couple with one of the two being 3 years older and no children
        elif i == 1:  
            households_tot[i] += 1
            # Sample from left-truncated age distribution (adult aged >= 30)
            age_adult = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated] = age_adult
            # For heterosexual couples, the man is older than the woman on average, let the age diff be random between 0 and 3 years
            age[num_generated+1] = min(n_ages-1,age_adult+np.random.randint(-2,4))
            generated_this_step = 2
            
        # Single parent + 1 child
        elif i == 2:     
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            age[num_generated + 1] = mother_current_age
            generated_this_step = 2
            
        # Single parent + 2 children
        elif i == 3:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            age[num_generated + 2] = mother_current_age
            generated_this_step = 3
            
        # Single parent + 3 children
        elif i == 4:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            age[num_generated + 3] = mother_current_age
            generated_this_step = 4  
                
        # Couple with one of the two being older + 1 child
        elif i == 5: 
            households_tot[i] += 1
            # Child
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + child_age)
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 3
        
        # Couple with one of the two being older + 2 children
        elif i == 6:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 4            
        
        # Couple with one of the two being older + 3 children
        elif i == 7:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+3)]))
            # Populate age for parents
            age[num_generated + 3] = mother_current_age
            age[num_generated + 4] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            generated_this_step = 5
        
        # Family without nucleus (2 adults >= 30)
        elif i == 8:
            households_tot[i] += 1
            age[num_generated] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            age[num_generated+1] = np.random.choice(np.arange(adult_child_random,101), p=renormalized_adult_older)
            generated_this_step = 2         
                
        # Nucleus with other persons (couple with one of the two being older + 1 children + 2 adults in retirement)
        elif i == 9:
            households_tot[i] += 1
            # Children
            child_age = np.random.choice(adult_child_random,p= renormalized_child)
            age[num_generated] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 1] = mother_current_age
            age[num_generated + 2] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Populate age for adult >= 60
            age[num_generated + 3] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent) 
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent)  
            generated_this_step = 6
            
        # Nucleus with other persons (couple with one of the two being older + 2 children + 1 adults in retirement)
        elif i == 10:
            households_tot[i] += 1
            # Children
            for j in range(2):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parents
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parents
            age[num_generated + 2] = mother_current_age
            age[num_generated + 3] = min(n_ages-1,mother_current_age+np.random.randint(-2,4))
            # Populate age for grand-parents
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent) 
            generated_this_step = 5
        
        # Nucleus with other persons (single parent + 3 children + 1 adults in retirement)
        elif i == 11:
            households_tot[i] += 1
            # Children
            for j in range(3):                
                child_age = np.random.choice(adult_child_random,p= renormalized_child)
                age[num_generated+j] = child_age
            # Parent
            mother_age_at_birth = (np.random.choice(7, p=renormalized_mother) + 3)*5+np.random.randint(5)
            mother_current_age = min(n_ages-1,mother_age_at_birth + max(age[num_generated:(num_generated+2)]))
            # Populate age for parent
            age[num_generated + 3] = mother_current_age
            # Grand-parent
            age[num_generated + 4] = np.random.choice(np.arange(age_grandparent_random,101), p=renormalized_grandparent) 
            generated_this_step = 5
            
        # Update list of household contacts accordingly 
        for i in range(num_generated, num_generated+generated_this_step):
            curr_pos = 0
            for j in range(num_generated, num_generated+generated_this_step):
                if i != j:
                    households[i, curr_pos] = j
                    curr_pos += 1
        num_generated += generated_this_step
        
    return households, age, households_tot

#households, age, households_tot = sample_households_france(1000000)