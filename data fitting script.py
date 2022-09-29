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
# folder = 'C:\\Users\\marsh\\Dropbox (University of Michigan)\\Michigan\\bcl2\\BLI data\\27sep22 bim-p5-pe,bim-p7-pe\\'
folder = '/Users/marshallcase/Dropbox (University of Michigan)/Michigan/bcl2/BLI data/27sep22 bim-p5-pe,bim-p7-pe/'
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
        'E1_normalized','F1_normalized','G1_normalized','H1_normalized']
# sensors=['F1_normalized',

#time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(steps),len(sensors)))
# time_bounds[0,1] = 0

#y0_bounds: matrix of starting values
y0_bounds = np.zeros((len(steps),len(sensors)))

#alphas: coefficients of baseline drift
alphas = np.zeros((len(steps),len(sensors)))
alphas = np.where(alphas==0,None,0)

functions= [association_linerror,disassociation_linerror,association_linerror,disassociation_linerror,
            association_linerror,disassociation_linerror,association_linerror,disassociation_linerror,
            association_linerror,disassociation_linerror]


#get processed data, split by time step
data_dict = preprocess(data,t_b,t_l,t_w,t_a,t_d)

# get fit parameters for each time step
parameters_dict = fitAll(data_dict,steps,time_bounds,sensors,functions,y0_bounds,alphas)

# plot parameters to manually evaluate fit
plot_size = (4,2)
plot_fit_all(data_dict,steps,time_bounds,sensors,functions,parameters_dict,plot_size)


# =============================================================================
# improve fit quality
# =============================================================================
# #time_bounds: matrix of upper time limits (for baseline drift)
time_bounds = np.zeros((len(steps),len(sensors)))
alphas=np.zeros((len(steps),len(sensors)))
alphas = np.where(alphas==0,None,0)

time_bounds[[1,3,5,7,9],0]=0

#y0_bounds: matrix of y0 | t=0
y0_bounds = np.zeros((len(steps),len(sensors)))
# for i,step in enumerate(steps):
#     for j,sensor in enumerate(sensors):
#         if 'Assoc' in step:
#             y0_bounds[i,j] = min(data_dict[step][sensor].values)
#         elif 'Disassoc' in step:
#             y0_bounds[i,j] = max(data_dict[step][sensor].values)

y0_bounds[[1,3,5,7,9],0] = [max(data_dict[j]['A1_normalized'].values) for j in steps if 'Disassoc' in j]
y0_bounds[[4,6,8],0] = [min(data_dict[j]['A1_normalized'].values) for j in steps if 'Assoc' in j][2:]
y0_bounds[[1,3,5,7,9],2] = [max(data_dict[j]['C1_normalized'].values) for j in steps if 'Disassoc' in j]
y0_bounds[[1,3,5,7,9],3] = [max(data_dict[j]['D1_normalized'].values[:100]) for j in steps if 'Disassoc' in j]
y0_bounds[[1,3,5,7,9],4] = [max(data_dict[j]['E1_normalized'].values[:100]) for j in steps if 'Disassoc' in j]
y0_bounds[[4,6,8],4] = [min(data_dict[j]['E1_normalized'].values[:100]) for j in steps if 'Assoc' in j][2:]
y0_bounds[[1,3,5,7,9],6] = [max(data_dict[j]['G1_normalized'].values[:100]) for j in steps if 'Disassoc' in j]
y0_bounds[[0,2,4,6,8],6] = [min(data_dict[j]['G1_normalized'].values[:100]) for j in steps if 'Assoc' in j]
y0_bounds[[1,3,5,7,9],7] = [max(data_dict[j]['H1_normalized'].values[:100]) for j in steps if 'Disassoc' in j]
y0_bounds[[0,2,4,6,8],7] = [min(data_dict[j]['H1_normalized'].values[:100]) for j in steps if 'Assoc' in j]

alphas[9,[6,7]]=0
alphas[[2,3,4],6]=0
alphas[[4,6],0]=0
alphas[[2,4,6],7]=0
alphas[[4,6,8],4]=0
alphas[7,2]=0
alphas[0,[6,7]]=0
time_bounds[[2,4,6],7]=200
time_bounds[0,[6,7]]=200
time_bounds[[4,6,8],4]=200

parameters_dict_upd = fitAll(data_dict,steps,time_bounds,sensors,functions,y0_bounds,alphas)
plot_fit_all(data_dict,steps,time_bounds,sensors,functions,parameters_dict_upd,plot_size)

# =============================================================================
# #plot final parameters from fit
# =============================================================================
##manually remove bad fits before plotting
parameters_dict_upd['B1_normalized'].loc[:,:]=np.nan
parameters_dict_upd['C1_normalized'].loc[['Assoc1'],:]=np.nan
parameters_dict_upd['F1_normalized'].loc[:,:]=np.nan
parameters_dict_upd['G1_normalized'].loc[['Disassoc1'],:]=np.nan
# #concentrations of association phases - [M]
concs = [10e-9,25e-9,100e-9,250e-9,800e-9]
parameters_consolidated = plot_fit_parameters(parameters_dict_upd,steps, ['A1_normalized', 'B1_normalized', 'C1_normalized', 'D1_normalized',
        'E1_normalized','F1_normalized','G1_normalized','H1_normalized'],time_bounds,functions,concs)

