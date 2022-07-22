#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:58:54 2022

@author: marshallcase
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# =============================================================================
# preprocess
# =============================================================================
def preprocess(data,t_b,t_l,t_w,t_a,t_d):
    '''
    preprocess: convert the raw data matrix from BLI into a dictionary of steps
    Input:
        data: a DataFrame object with columns Time and sensors for each channel
        t_b: time of baseline step
        t_l: time of load step
        t_w: time of wash step
        t_a: time of association step
        t_d: time of disassociation step
    Output:
        data_dict: a dictionary of data broken down by steps
    '''
    
    steps = pd.DataFrame(columns = ['Step','Time'])
    steps['Step'] = ['Baseline_start','Load','Wash','Baseline1',
                     'Assoc1','Disassoc1','Baseline2','Assoc2',
                     'Disassoc2','Baseline3','Assoc3','Disassoc3',
                     'Baseline4','Assoc4','Disassoc4','Baseline5',
                     'Assoc5','Disassoc5','End']
    steps['Time']=[0,t_b,t_b+t_l,t_b+t_l+t_w,t_b*2+t_l+t_w,
                   t_b*2+t_l+t_w+t_a,t_b*2+t_l+t_w+t_a+t_d,
                   t_b*3+t_l+t_w+t_a+t_d,t_b*3+t_l+t_w+t_a*2+t_d,
                   t_b*3+t_l+t_w+t_a*2+t_d*2,
                   t_b*4+t_l+t_w+t_a*2+t_d*2,
                   t_b*4+t_l+t_w+t_a*3+t_d*2,
                   t_b*4+t_l+t_w+t_a*3+t_d*3,
                   t_b*5+t_l+t_w+t_a*3+t_d*3,
                   t_b*5+t_l+t_w+t_a*4+t_d*3,
                   t_b*5+t_l+t_w+t_a*4+t_d*4,
                   t_b*6+t_l+t_w+t_a*4+t_d*4,
                   t_b*6+t_l+t_w+t_a*5+t_d*4,
                   t_b*6+t_l+t_w+t_a*5+t_d*5]
    
    steps['data_index'] = [data['Time'].sub(time).abs().idxmin() for time in steps['Time']]
    
    num_sensors = len(data.columns)-1
    sensor_normalized_column_titles = [str(data.columns[i+1]) + '_normalized' for i in range(num_sensors)]
    
    data_dict = dict(zip(steps['Step'].values,[pd.DataFrame() for step in steps['Step']]))
    for index,step in enumerate(steps['Step']):
        if index == len(steps['Step'])-1:
            break
        data_dict[step] = data.loc[steps.loc[index,'data_index']:(steps.loc[index+1,'data_index']-1),:]
        data_dict[step].loc[:,'Time0'] = data_dict[step]['Time']-data_dict[step].iloc[0,0]
        for i in range(num_sensors):
            if data_dict[step].iloc[0,i+1] == 0:
                data_dict[step][sensor_normalized_column_titles[i]] = data_dict[step].iloc[:,i+1]
            else:
                data_dict[step][sensor_normalized_column_titles[i]] = data_dict[step].iloc[:,i+1]/data_dict[step].iloc[0,i+1]
    return data_dict


# =============================================================================
# curvefit all sensors + steps
# =============================================================================
def fitAll(data_dict,fits,time_bounds,sensors,functions):
    '''
    fitAll: fit every step in an BLI experiment
    Input:
        data_dict: a dictionary of data broken down by steps (from preprocess)
        fits: an array of steps to fit
        time_bounds: an array of dimension (# sensors, # fits) with upper time limits,
                                            in case of baseline drift
        sensors: an array of sensors to fit
        functions: which functions correspond to which steps
    Output:
        parameters_dict: fit parameters (Y0, Plateau, K) and their standard deviations
    '''
    parameters_dict = dict(zip(sensors,[pd.DataFrame(index=fits,
                      columns=['Y0','Plateau','K','Y0_std','Plateau_std','K_std']) for s in sensors]))
    
    for i,step in enumerate(fits):
        for j,sensor in enumerate(sensors):
            func = functions[i]
            time_bound = time_bounds[i,j]
            if time_bound == 0:
                time_bound=None
            popt,perr = get_curve_fit(data_dict,step,sensor,func,time_bound)
            parameters_dict[sensor].loc[step,:] = np.hstack((popt,perr))
            
    return parameters_dict

# =============================================================================
# plot any step function
# =============================================================================
def plot_step(data_dict,step,sensors=None):
    '''
    plot_step: plot raw data from a given BLI step
    Input:
        data_dict: a dictionary of data broken down by steps (from preprocess)
        step: which step to plot
        sensors: which sensor(s) to fit, defaults to all
        functions: which functions correspond to which steps
    Output:
        None
    '''
    plot_data = data_dict[step]
    num_sensors = int((len(plot_data.columns)-2)/2)
    if sensors is None:
        sensors = plot_data.columns[(2+num_sensors):].values
    for s in sensors:
        plt.scatter([plot_data['Time0']], plot_data[s],label=s)
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Normalized Signal')
    plt.tight_layout()

def plot_fit(data_dict,step,sensor,func,popt,time_bound=None,ax=None,**kwargs):
    '''
    plot_fit: plot raw data with its fit from a given BLI step
    Input:
        data_dict: a dictionary of data broken down by steps (from preprocess)
        step: which step to plot
        sensors: which sensor(s) to fit, defaults to all
        func: which function to plot
        popt: parameters from a curve_fit
        time_bound: upper time limit for plotting, default to None
        ax: ax object for subplotting, default to None
        **kwargs: kwargs for plotting
    Output:
        ax: optional, if ax object supplied as input. Only for multiple subplots
    '''
    if time_bound is not None:
        raw_data = data_dict[step]
        fit_data = data_dict[step].loc[data_dict[step]['Time0'] <= time_bound][['Time0',sensor]]
    else:
        raw_data = data_dict[step]
        fit_data = data_dict[step]
    t_0 = 0.0
    t_end = data_dict[step]['Time0'].values[-1]
    time = np.linspace(t_0,t_end,int(np.round(t_end*5)))
    fit = func(time,popt[0],popt[1],popt[2])
    if ax is None:
        plt.plot(time,fit,label='fit',linewidth=5)
        plt.scatter(raw_data['Time0'],raw_data[sensor],s=1,**kwargs)
    else:
        ax.plot(time,fit,label='fit',linewidth=1,c='black')
        ax.scatter(raw_data['Time0'],raw_data[sensor],s=1,**kwargs)
        return ax

def plot_fit_all(data_dict,fits,time_bounds,sensors,functions,parameters_dict,plot_size=(4,4)):
    '''
    plot_fit_all: plot an entire BLI experiment with its fits
    Input:
        data_dict: a dictionary of data broken down by steps (from preprocess)
        fits: which steps to plot
        time_bounds: an array of dimension (# sensors, # fits) with upper time limits,
                                            in case of baseline drift
        sensors: which sensor(s) to fit, defaults to all
        functions: which functions correspond to which steps
        parameter_dict: parameters from fitAll
        plot_size: size of every individual plot. default to (4,4)
        **kwargs: kwargs for plotting
    Output:
        None
    '''
    plt.figure(figsize=(plot_size[0]*len(sensors), plot_size[1]*len(fits)))
    # plt.ioff()
    for i,step in enumerate(fits):
        for j,sensor in enumerate(sensors):
            ax = plt.subplot(len(fits),len(sensors),i*len(sensors)+j+1)
            if i == (len(fits)-1):
                ax.set_xlabel(sensor.split('_')[0])
            if j == 0:
                ax.set_ylabel(step)
            func = functions[i]
            time_bound = time_bounds[i,j]
            if time_bound == 0:
                time_bound=None
            popt = parameters_dict[sensor].loc[step][:3].values
            ax = plot_fit(data_dict,step,sensor,func,popt,time_bound,ax)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            
    plt.tight_layout()
    # plt.savefig("test.png",dpi=1000)
# =============================================================================
# define differential equation forms
# =============================================================================
def association(t, Y0, Plateau, K):
    '''
    association: general equation for single phase association
    Input:
        t: time (seconds)
        Y0: starting value (normalized distance)
        Plateau: asymptote value (normalized distance)
        K: rate constant (1/M/sec)
    Output:
        Y0 + (Plateau-Y0)*(1-np.exp(-K*t)): one phase association equation
    '''
    return Y0 + (Plateau-Y0)*(1-np.exp(-K*t))

def disassociation(t,Y0,Plateau,K):
    '''
    disassociation: general equation for single phase disassociation
    Input:
        t: time (seconds)
        Y0: starting value (normalized distance)
        Plateau: asymptote value (normalized distance)
        K: rate constant (1/M/sec)
    Output:
        (Y0 - Plateau)*np.exp(-K*t) + Plateau: one phase disassociation equation
    '''
    return (Y0 - Plateau)*np.exp(-K*t) + Plateau

# =============================================================================
# function to fit equations
# =============================================================================
def get_curve_fit(data_dict,step,sensor,func,time_bound=None,**kwargs):
    '''
    get_curve_fit: get fit for a given BLI step
    Input:
        data_dict: a dictionary of data broken down by steps (from preprocess)
        step: which step to plot
        sensor: which sensor to fit
        func: which function to use for fit
        time_bound: set an upper limit in case of baseline drift, defaults to None
    Output:
        popt: fit parameters, outputs 0 if max function evaluations hits cap (default maxfev=800)
        perr: standard deviation fit parameters, outputs 0 if max function evaluations hits cap (default maxfev=800)
    '''
    if time_bound is not None:
        fit_data = data_dict[step].loc[data_dict[step]['Time0'] <= time_bound][['Time0',sensor]]
    else:
        fit_data = data_dict[step]
    try:
        popt, pcov = curve_fit(func, fit_data['Time0'], fit_data[sensor], **kwargs)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print("Optimal parameters not found: Number of calls to function has reached maxfev = 800.")
        popt = np.array([0,0,0])
        perr = np.array([0,0,0])
    return popt, perr

# =============================================================================
# output data by timestep
# =============================================================================
##filepath
# root = 'outputData'
## if ~os.path.isfile(os.getcwd()+'\\'+root):
# os.makedirs(root)
# for index in range(len(steps['Step'])):
#     if index == len(steps['Step'])-1:
#         break
#     output_data = data.loc[steps.loc[index,'data_index']:steps.loc[index+1,'data_index'],:]
#     output_data['Time0'] = output_data['Time']-output_data.iloc[0,0]
#     for i in range(num_sensors):
#         output_data[sensor_normalized_column_titles[i]] = output_data.iloc[:,i+1]/output_data.iloc[0,i+1]
    
#     output_data.to_excel(root + '/' + steps.loc[index,'Step'] + '.xlsx')