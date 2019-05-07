#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:54:51 2019

@author: BrianTCook
"""

import numpy as np
import pandas as pd

'''
data goes from February 8 2013 to Febraury 7 2018
to get standard quarters (Jan/Feb/Mar and so on), will only use 14/15/16/17 data
'''

raw = pd.read_csv('all_stocks_5yr.csv')
data = raw.loc[:, ('date','open','high','low','close','volume','Name')]

def inputvec(company, year):
    
    '''
    takes the relevant info from data (company is a string, year is an int)
    generates a 4D vector Q_{i} for the ith quarter
    
    Q_{i} elements:
        v_{i}: standard deviation of opening price
        s_{i}: mean(high price - low price)
        n_{y,i}: mean(traded shares)
        h_{y,i}: mean(high price)
    
    output is concatenation of Q_{1}, Q_{2}, Q_{3}, and Q_{4}
    '''

    #filters out by the company and relevant year
    filter1 = data.loc[data['Name'] == company]
    filter2 = filter1.loc[filter1['date'].str.contains(str(year))]
    
    def Q(i):
        
        if i == 1:
            
            month_strs = ['-01-', '-02-', '-03-']
            
        if i == 2:
            
            month_strs = ['-04-', '-05-', '-06-']
            
        if i == 3:
            
            month_strs = ['-07-', '-08-', '-09-']
            
        if i == 4:
            
            month_strs = ['-10-', '-11-', '-12-']
        
        filterquarter = filter2
        months = [filterquarter.loc[filterquarter['date'].str.contains(month_str)] for month_str in month_strs]
    
        opens = [month['open'].tolist() for month in months] 
        opens = [j for i in opens for j in i] #concatenates months
        
        highs = [month['high'].tolist() for month in months]
        highs = [j for i in highs for j in i]
        
        lows = [month['low'].tolist() for month in months]
        lows = [j for i in lows for j in i]
        
        volumes = [month['high'].tolist() for month in months]
        volumes = [j for i in volumes for j in i]
        
        spreads = [highs[i] - lows[i] for i in range(len(highs))]
        
        #ignores NaNs
        v, s = np.nanstd(opens), np.nanmean(spreads)
        n, h = np.nanmean(volumes), np.mean(highs)
    
        return [v, s, n, h]
    
    return Q(1) + Q(2) + Q(3) + Q(4)

print(inputvec('GOOG', 2014))

    