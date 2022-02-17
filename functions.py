## FUNCTIONS SCRIPT 
'''
Script containing the definitions of some useful general and global functions which are then recalled in the overall SIR and ABM executions

The file contains the definition of the following functions: 
- ** (1) categorical_sample**: function which allows to create a sample categorical from a set of elements
- ** (2) threshold_exponential**: allows to set the threshold of an exponential distribution 
- ** (3) threshold_log_normal**: set a threshold for a lognormal distribution 
- ** (4) resevoir_sample**: function used to sample the initial set of infected individual. It is used a *reservoir sampling* algorithm to select a set of *k* elements selected from the *n* available ones 
'''

# import libraries
import numpy as np # import numpy library 
from numba import jit # library to optimize the code 

## (1) Categorial Sample Function 
def categorical_sample(p):
    """Definition of a categorical sample function"""
    threshold = np.random.rand() # one sample from a uniform distribution over (0,1)
    current = 0 
    for i in range(p.shape[0]):
        current += p[i]
        if current > threshold:
            return i
        
## (2) Threshold Exponential Function   
#@jit(nopython=True,nogil=True)
def threshold_exponential(mean):
    """Definition of a function to get the threshold for an exponential shaped distribution
    np.random.exponential(mean):Sample from the exponential distribution with scale parameter = mean """
    return np.round(np.random.exponential(mean)) 

## (3) Threshold Log Normal Function 
#@jit(nopython=True,nogil=True)
def threshold_log_normal(mean, sigma):
    """ Definition of a function to set a threshold to a lognormal distribution
    Draw samples from a log-normal distribution with (mean,sigma)
    """
    x = np.random.lognormal(mean, sigma)
    if x <= 0:
        return 1
    else:
        return np.round(x)

## (4) Resevoir Sample Function  
'''
**Reservoir sampling** is a family of randomized algorithms for choosing a simple random sample, without replacement, of k items from a population of unknown size n in a single pass over the items.
The basic algorithm works by maintaining a reservoir of size k, which initially contains the first k items of the input. It then iterates over the remaining items until the input is exhausted. 
'''
#@jit(nopython=True)
#functions.resevoir_sample(n, int(initial_infected_fraction*n))

def resevoir_sample(n, k):
    """Definition of a function using resevoir resampling to sample the intial infected individuals
    Output: array of lenght k of elements selected from the n available ones {range from 0 to n-1}
    Optimal algorithm for RESEVOIR SAMPLING == Algorithm L
    """
    R = np.zeros(k, dtype=np.int32) # define an array of k elements = 0
    if k == 0:
        return R # return an empty array
    for i in range(k):
        R[i] = i # fill the resevoir array
    W = np.exp(np.log(np.random.rand())/k)
    while i < n:
        i = i + int(np.floor(np.log(np.random.rand())/np.log(1-W))) + 1
        if i < n:
            # replace a random item of the reservoir with item i
            R[np.random.randint(0, k)] = i # random index between 1 and k, inclusive
            W = W * np.exp(np.log(np.random.rand())/k) 
    return R