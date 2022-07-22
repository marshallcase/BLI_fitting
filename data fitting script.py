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
data = pd.read_excel('RawData.xlsx')
experiment_outline = pd.read_table('220718_ExpMethod.fmf')

#manual parameters
t_b = 60 #baseline
t_l = 900 #load
t_w= 900 #wash
t_a = 600 #association
t_d = 3600 #diassociation

#atuomatic parameters
experiment_details = experiment_outline.iloc[0].values[0]
#TODO: extract time values from this nightmare


#fits: which steps to fit curves to
fits = ['Assoc1', 'Disassoc1', 'Assoc2', 'Disassoc2', 
         'Assoc3', 'Disassoc3', 'Assoc4', 
        'Disassoc4', 'Assoc5', 'Disassoc5']

#sensors: which sensors to fit (i.e. ignore the blanks). has the shape (n_steps,n_sensors)
sensors = ['A1_normalized', 'B1_normalized', 'C1_normalized', 'D1_normalized',
       'E1_normalized', 'F1_normalized']

#time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(fits),len(sensors)))
time_bounds[0,1] = 0

#functions: which functions to fit to which steps
functions= [association,disassociation,association,disassociation,
            association,disassociation,association,disassociation,
            association,disassociation]

#get processed data, split by time step
data_dict = preprocess(data,t_b,t_l,t_w,t_a,t_d)

#get fit parameters for each time step
parameters_dict = fitAll(data_dict,fits,time_bounds,sensors,functions)

#plot parameters to manually evaluate fit
plot_size = (4,2)
plot_fit_all(data_dict,fits,time_bounds,sensors,functions,parameters_dict,plot_size)


# =============================================================================
# improve fit quality
# =============================================================================
#time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(fits),len(sensors)))
time_bounds[1,1] = 1500
time_bounds[1,2] = 1000
time_bounds[3,1] = 1000
time_bounds[3,2] = 1000
time_bounds[3,3] = 2000
time_bounds[5,2] = 1000
time_bounds[7,2] = 500
time_bounds[7,4] = 1500
time_bounds[9,2] = 500
time_bounds[9,4] = 500

time_bounds[[1,3,5,7,9],1]=1000
time_bounds[[1,3,5,7,9],2]=1000
time_bounds[[1,3,5,7,9],3]=2000
time_bounds[[1,3,5,7,9],4]=1000

parameters_dict_upd = fitAll(data_dict,fits,time_bounds,sensors,functions)
plot_fit_all(data_dict,fits,time_bounds,sensors,functions,parameters_dict_upd,plot_size)