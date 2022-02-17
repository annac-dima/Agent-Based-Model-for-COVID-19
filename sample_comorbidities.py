## SAMPLE COMORBIDITIES

# import libraries 
import numpy as np
import pandas as pd

### DATA   
'''
**ITALY HYPERTENSION** PREVALENCE (%) BY AGE: https://www.ncbi.nlm.nih.gov/pubmed/28487768
- 35-39y: 14
- 40-44y: 10
- 45-49y: 16
- 50-54y: 30
- 55-100y : 34  

**SPAIN HYPERTENSION** PREVALENCE (%) BY AGE: https://www.revespcardiol.org/en-prevalence-diagnosis-treatment-and-control-articulo-S1885585716000505?redirect=true
- 18-30y: 9.3
- 31-45y: 17.2
- 46-60y: 44.4
- 61-75y: 75.4
- 75y-100 : 88.7   
  
**GERMANY HYPERTENSION** PREVALENCE (%) BY AGE:   
https://www.nature.com/articles/jhh201482.pdf -- Table 2 2008-11 Data;
- 18-29y: 5
- 30-44y: 11.7
- 45-64y: 38
- 65-74y: 71
- 75-84y: 79.9
- 85-100y: 73.6  

**FRANCE HYPERTENSION** PREVALENCE (%) BY AGE:   
https://academic.oup.com/ajh/article/11/6/759/111907
- 18-34y: 8
- 35-49y: 28
- 50-64y: 57
- 65-79y: 76
- 80-100y : 79  

**ITALY-SPAIN-GERMANY-FRANCE DIABETES PREVALENCE BY AGE**: http://ghdx.healthdata.org/gbd-results-tool  
*See also next cell with the code to import and read diabetes dataset from Global Burden 2019*

## Import and read diabetes dataset from Global Burden 2019
country = 'Italy'
diabetes_data = pd.read_csv("C:/Users/anna9/Desktop/TESI/CODE/COVID19-Demography TEST/GlobalBurden_DiabetesData/IHME-GBD_2019_DATA-310f0f2a-1.csv")
prev = data[data['location'] == country].val.to_list()
age = data[data['location'] == country].age.to_list()
data[data['location']==country].head(10)
'''

## FUNCTIONS
# (1) Sample Joint Function 
def sample_joint(age, p_diabetes, p_hyp):
    #https://pubmed.ncbi.nlm.nih.gov/28701739/ 
    # retrieve the p of hypertension GIVEN DIABETES = 0.5 
    p_hyp_given_diabetes = 0.5
    # get the p of hypertension GIVEN NOT DIABETES
    p_hyp_given_not_diabetes = (p_hyp - p_hyp_given_diabetes*p_diabetes)/(1 - p_diabetes)
    # retrieve the diabetes status using the p_diabetes 
    diabetes_status = (np.random.rand(age.shape[0]) < p_diabetes[age]).astype(np.int)
    hyp_status = np.zeros(age.shape[0], dtype=np.int) # buid a zeros-array to retrieve hypertension statuses
    # retrieve the hypertension status
    # for those that have diabetes_status == 1 use p_hyp_given_diab
    hyp_status[diabetes_status == 1] = np.random.rand((diabetes_status == 1).sum()) < p_hyp_given_diabetes
    # for those that have diabetes_status == 0 retrieve the p_hyp_given_not_diabetes age-specific rate and sample 
    hyp_status[diabetes_status == 0] = np.random.rand((diabetes_status == 0).sum()) < p_hyp_given_not_diabetes[age[diabetes_status == 0]]
    return diabetes_status, hyp_status

# (2) Sample Joint for age, country Function 
def sample_joint_comorbidities(age, country='Italy'):
    """
    Default country is Italy.
    For other countries pass value for country from {Italy,Spain,Germany,France}
    """
    return sample_joint(age, p_comorbidity(country, 'diabetes'), p_comorbidity(country, 'hypertension'))

# (3) Comorbidity Probability Function 
def p_comorbidity(country, comorbidity, warning=False):

    """
    Input:
        -country: a string input belonging to- {Italy,Spain,Germany,France}
        -comorbidity: a string input belonging to- {diabetes, hypertension}
        -warning: optional, If set to True, prints out the underlying assumptions/approximations
    Returns:
        -prevalence, sampled from a prevalence array of size 100, where prevalence[i] is the prevalence rate at age between {i, i+1}
    """

    prevalence = np.zeros(101)
    warning_string= " "

    ######################################  ITALY #############################
    if country=='Italy':

        if comorbidity=='diabetes':
            # Global Burden of Disease Study 2019 (GBD 2019) Results.
            # Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
            # Available from http://ghdx.healthdata.org/gbd-results-tool.

            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.00064
                elif i <= 9:
                    prevalence[i] = 0.0022
                elif i <= 14:
                    prevalence[i] = 0.0038
                elif i <= 19:
                    prevalence[i] = 0.0058
                elif i <= 24:
                    prevalence[i] = 0.0107
                elif i <= 29:
                    prevalence[i] = 0.0180
                elif i <= 34:
                    prevalence[i] = 0.0261
                elif i <= 39:
                    prevalence[i] = 0.0360
                elif i <= 44:
                    prevalence[i] = 0.0496
                elif i <= 49:
                    prevalence[i] = 0.0729
                elif i <= 54:
                    prevalence[i] = 0.1080
                elif i <= 59:
                    prevalence[i] = 0.1575
                elif i <= 64:
                    prevalence[i] = 0.2153
                elif i <= 69:
                    prevalence[i] = 0.2634
                elif i <= 74:
                    prevalence[i] = 0.2969
                elif i <= 79:
                    prevalence[i] = 0.3086
                elif i <= 84:
                    prevalence[i] = 0.3031
                elif i <= 89:
                    prevalence[i] = 0.2877
                else:
                    prevalence[i] = 0.2615

        elif comorbidity=='hypertension':
            #https://www.ncbi.nlm.nih.gov/pubmed/28487768
            for i in range(101):
                if i<35:
                    prevalence[i]= 0.14*(i/35.)
                elif i<39:
                    prevalence[i]=0.14
                elif i<44:
                    prevalence[i]=0.1
                elif i<49:
                    prevalence[i]=0.16
                elif i<54:
                    prevalence[i]=0.3
                else:
                    prevalence[i]=0.34
                    
    ######################################  SPAIN #############################
    elif country=='Spain':

        if comorbidity=='diabetes':
            # Global Burden of Disease Study 2019 (GBD 2019) Results.
            # Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
            # Available from http://ghdx.healthdata.org/gbd-results-tool.
            
            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.00085
                elif i <= 9:
                    prevalence[i] = 0.0023
                elif i <= 14:
                    prevalence[i] = 0.0040
                elif i <= 19:
                    prevalence[i] = 0.0060
                elif i <= 24:
                    prevalence[i] = 0.0141
                elif i <= 29:
                    prevalence[i] = 0.0231
                elif i <= 34:
                    prevalence[i] = 0.0312
                elif i <= 39:
                    prevalence[i] = 0.0408
                elif i <= 44:
                    prevalence[i] = 0.0540
                elif i <= 49:
                    prevalence[i] = 0.0765
                elif i <= 54:
                    prevalence[i] = 0.1094
                elif i <= 59:
                    prevalence[i] = 0.1556
                elif i <= 64:
                    prevalence[i] = 0.2101
                elif i <= 69:
                    prevalence[i] = 0.2601
                elif i <= 74:
                    prevalence[i] = 0.2996
                elif i <= 79:
                    prevalence[i] = 0.3168
                elif i <= 84:
                    prevalence[i] = 0.3124
                elif i <= 89:
                    prevalence[i] = 0.2981
                else:
                    prevalence[i] = 0.2752

        elif comorbidity=='hypertension':
            #https://www.revespcardiol.org/en-prevalence-diagnosis-treatment-and-control-articulo-S1885585716000505?redirect=true
            # Table 2
            for i in range(101):
                if i<18:
                    prevalence[i]=9.3*(i/18.)
                elif i<30:
                    prevalence[i]=9.3
                elif i<45:
                    prevalence[i]=17.2
                elif i<60:
                    prevalence[i]=44.4
                elif i<75:
                    prevalence[i]=75.4
                else:
                    prevalence[i]=88.7
            prevalence = [p/100 for p in prevalence]
            
    ######################################  GERMANY #############################
    elif country=='Germany':

        if comorbidity=='diabetes':
            # Global Burden of Disease Study 2019 (GBD 2019) Results.
            # Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
            # Available from http://ghdx.healthdata.org/gbd-results-tool.
            
            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.00059
                elif i <= 9:
                    prevalence[i] = 0.0021
                elif i <= 14:
                    prevalence[i] = 0.0036
                elif i <= 19:
                    prevalence[i] = 0.0072
                elif i <= 24:
                    prevalence[i] = 0.0163
                elif i <= 29:
                    prevalence[i] = 0.0266
                elif i <= 34:
                    prevalence[i] = 0.0367
                elif i <= 39:
                    prevalence[i] = 0.0490
                elif i <= 44:
                    prevalence[i] = 0.0658
                elif i <= 49:
                    prevalence[i] = 0.0929
                elif i <= 54:
                    prevalence[i] = 0.1315
                elif i <= 59:
                    prevalence[i] = 0.1823
                elif i <= 64:
                    prevalence[i] = 0.2409
                elif i <= 69:
                    prevalence[i] = 0.2956
                elif i <= 74:
                    prevalence[i] = 0.3406
                elif i <= 79:
                    prevalence[i] = 0.3615
                elif i <= 84:
                    prevalence[i] = 0.3595
                elif i <= 89:
                    prevalence[i] = 0.3458
                else:
                    prevalence[i] = 0.3230

        elif comorbidity=='hypertension':
            # https://www.nature.com/articles/jhh201482.pdf -- Table 2 2008-11 Data;
            for i in range(101):
                if i<18:
                    prevalence[i]=5*(i/18.)
                elif i<29:
                    prevalence[i]=5
                elif i<44:
                    prevalence[i]=11.7
                elif i<64:
                    prevalence[i]=38
                elif i<74:
                    prevalence[i]=71
                elif i<84:
                    prevalence[i]=79.9
                else:
                    prevalence[i]=73.6
            prevalence = [p/100 for p in prevalence]


    ######################################  FRANCE #############################
    elif country=='France':

        if comorbidity=='diabetes':
            # Global Burden of Disease Study 2019 (GBD 2019) Results.
            # Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
            # Available from http://ghdx.healthdata.org/gbd-results-tool.
        
            for i in range(101):
                if i <= 4:
                    prevalence[i] = 0.00059
                elif i <= 9:
                    prevalence[i] = 0.0020
                elif i <= 14:
                    prevalence[i] = 0.0036
                elif i <= 19:
                    prevalence[i] = 0.0047
                elif i <= 24:
                    prevalence[i] = 0.0074
                elif i <= 29:
                    prevalence[i] = 0.0118
                elif i <= 34:
                    prevalence[i] = 0.0161
                elif i <= 39:
                    prevalence[i] = 0.0212
                elif i <= 44:
                    prevalence[i] = 0.0280
                elif i <= 49:
                    prevalence[i] = 0.0387
                elif i <= 54:
                    prevalence[i] = 0.0537
                elif i <= 59:
                    prevalence[i] = 0.0738
                elif i <= 64:
                    prevalence[i] = 0.0978
                elif i <= 69:
                    prevalence[i] = 0.1221
                elif i <= 74:
                    prevalence[i] = 0.1436
                elif i <= 79:
                    prevalence[i] = 0.1536
                elif i <= 84:
                    prevalence[i] = 0.1518
                elif i <= 89:
                    prevalence[i] = 0.1429
                else:
                    prevalence[i] = 0.1268

        elif comorbidity=='hypertension':
            # https://academic.oup.com/ajh/article/11/6/759/111907
            for i in range(101):
                if i<18:
                    prevalence[i]=8*(i/18.)
                elif i<34:
                    prevalence[i]=8
                elif i<49:
                    prevalence[i]=28
                elif i<64:
                    prevalence[i]=57
                elif i<79:
                    prevalence[i]=76
                else:
                    prevalence[i]=79
            prevalence = [p/100 for p in prevalence]

    if warning:
        print ("Warning: \n", warning_string)

    return prevalence


