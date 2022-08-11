# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:40:55 2022

@author: marsh
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from BLI_functions import *

# =============================================================================
# input parameters
# =============================================================================
folder = 'C:\\Users\\marsh\\Dropbox (University of Michigan)\\Michigan\\assorted antigens\\bcl2\\BLI data\\bfl1 bcl2 sort1,5-pe bim-p5-pe 10aug22\\bfl1 bcl2 sort1,5-pe bim-p5-pe 10aug22\\'
filename = 'RawData.xlsx'
data = pd.read_excel(folder+filename)
# experiment_outline = pd.read_table('220718_ExpMethod.fmf')

#manual parameters
t_b = 60 #baseline
t_l = 900 #load
t_w= 900 #wash
t_a = 1200 #association
t_d = 3600 #diassociation

#automatic parameters
# experiment_details = experiment_outline.iloc[0].values[0]
#TODO: extract time values from this nightmare


#fits: which steps to fit curves to
steps = ['Assoc1', 'Disassoc1', 'Assoc2', 'Disassoc2', 
         'Assoc3', 'Disassoc3', 'Assoc4', 
        'Disassoc4', 'Assoc5', 'Disassoc5']

#sensors: which sensors to fit (i.e. ignore the blanks). has the shape (n_steps,n_sensors)
sensors = ['A1_normalized', 'B1_normalized', 'C1_normalized', 'D1_normalized',
        'E1_normalized','F1_normalized']
# sensors=['F1_normalized','G1_normalized','H1_normalized']

#time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(steps),len(sensors)))
time_bounds[0,1] = 0

#functions: which functions to fit to which steps
functions= [association,disassociation,association,disassociation,
            association,disassociation,association,disassociation,
            association,disassociation]

#get processed data, split by time step
data_dict = preprocess(data,t_b,t_l,t_w,t_a,t_d)

#get fit parameters for each time step
parameters_dict = fitAll(data_dict,steps,time_bounds,sensors,functions)

#plot parameters to manually evaluate fit
plot_size = (4,2)
plot_fit_all(data_dict,steps,time_bounds,sensors,functions,parameters_dict,plot_size)


# =============================================================================
# improve fit quality
# =============================================================================
#time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(steps),len(sensors)))

time_bounds[[1,3,5,7,9],2]=400
time_bounds[[1,3,5,7,9],4]=1500
time_bounds[[1,3,5,7,9],5]=1000
time_bounds[[0,2,4,6,8],4]=500
time_bounds[[0,2,4,6,8],5]=600

#y0_bounds: matrix of y0 | t=0
y0_bounds = np.zeros((len(steps),len(sensors)))
y0_bounds[:,4]=1


parameters_dict_upd = fitAll(data_dict,steps,time_bounds,sensors,functions,y0_bounds)
plot_fit_all(data_dict,steps,time_bounds,sensors,functions,parameters_dict_upd,plot_size)

# =============================================================================
# #plot final parameters from fit
# =============================================================================
#manually remove bad fits before plotting
parameters_dict_upd['A1_normalized'].loc[:,:]=np.nan
parameters_dict_upd['B1_normalized'].loc[:,:]=np.nan
parameters_dict_upd['D1_normalized'].loc[:,:]=np.nan
parameters_dict_upd['C1_normalized'].loc[['Assoc1','Assoc2'],'K']=np.nan
parameters_dict_upd['E1_normalized'].loc[['Assoc1'],'K']=np.nan
parameters_dict_upd['F1_normalized'].loc[['Assoc1'],'K']=np.nan
# #concentrations of association phases - [M]
concs = [10e-9,25e-9,100-9,250e-9,1000e-9]
plot_fit_parameters(parameters_dict_upd,steps, ['C1_normalized','E1_normalized','F1_normalized'],time_bounds,functions,concs)
