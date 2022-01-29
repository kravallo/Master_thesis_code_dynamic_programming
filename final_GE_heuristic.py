#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:14:38 2021

@author: maltekemeter
"""

# load general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.linalg import block_diag
import time
import datetime
import os

# load modules related to this exercise
import tools as tools
from final_model import model_dc_multidim 

#=============================================================================
# get balance given price tupel function
#=============================================================================

def get_balance(input_premium_0, input_premium_1, mandate):
        
    # set up model
    model = model_dc_multidim()
    model.setup()
    model.create_grids()
    par = model.par
    
    # overwrite premium parameters
    par.lambdaa_0 = input_premium_0
    par.lambdaa_1 = input_premium_1
    
    #ggf. overwrite mandate default = 0
    par.mandate = mandate
    
    # solve model for given paramters
    model.solve_VFI_inf_horizon()
    
    # simulate from model 
    sim = model.simulate()
    
    #transform dataset
    df = sim.df 
    df.columns = ['delta m', 'm_t', 'm_t-1', 'c', 'a', 'p_lag', 'claims','theta', 'z','z_lag']
    df = pd.concat([df, pd.get_dummies(df['theta'])], axis=1)
    df.columns = ['delta m', 'm_t', 'm_t-1', 'c', 'a', 'p_lag', 'claims','theta', 'z','z_lag','dummy_type0','dummy_type1']
    df["z1_theta1"] = df['theta']*df['z']           # only 1 if both theta == 1 AND z == 1 otherwise 0
    df["z1_theta0"] = df['dummy_type0']*df['z']     # only 1 if both theta == 0 AND z == 1 otherwise 0
    df["z0_theta1"] = df['theta']*(1-df['z'])       # only 1 if both theta == 1 AND z == 0 otherwise 0
    df["z0_theta0"] = df['dummy_type0']*(1-df['z']) # only 1 if both theta == 0 AND z == 0 otherwise 0
    
    # create sub dfs
    df_theta0 = df.loc[df['theta'] == 0]
    df_theta1 = df.loc[df['theta'] == 1]
    df_insured = df.loc[df['z'] == 1]
    df_uninsured = df.loc[df['z'] == 0]
    df_insured_type0 = df_insured.loc[df['z1_theta0']==1] #insured but bad type
    df_insured_type1 = df_insured.loc[df['z1_theta1']==1] # insured but good type
    df_uninsured_type0 = df_uninsured.loc[df['z0_theta0']==1] # uninsured but bad type
    df_uninsured_type1 = df_uninsured.loc[df['z0_theta1']==1] # uninsured but good type
    
    # collect claims given type shape (10000,)
    claims_type0 = df_theta0['claims']
    claims_type1= df_theta1['claims']
    #calculate number of agents in portfolio
    portfolio_size_0 = np.count_nonzero(claims_type0)
    portfolio_size_1 = np.count_nonzero(claims_type1)

    # caluclate revenue from premiums
    Revenue =  par.lambdaa_0 * portfolio_size_0 + par.lambdaa_1 * portfolio_size_1
    Cost =  portfolio_size_0*0.5 + portfolio_size_1*0.25 

    # caluclate balance
    balance = Revenue - Cost - par.O
    
    # calculate other 
    ins_rate = df["z"].mean()
    ins_rate_theta0 = df_theta0["z"].mean()
    ins_rate_theta1= df_theta1["z"].mean()
    mean_consumption = df["c"].mean() 
    consumption_z0_theta0 = df_uninsured_type0["c"].mean()
    consumption_z0_theta1 = df_uninsured_type1["c"].mean() 
    consumption_z1_theta0 = df_insured_type0["c"].mean() 
    consumption_z1_theta1 = df_insured_type1["c"].mean() 
    
    # return list
    lst = []
    lst.append(balance)
    lst.append(ins_rate)
    lst.append(ins_rate_theta0)
    lst.append(ins_rate_theta1)
    lst.append(mean_consumption)
    lst.append(consumption_z0_theta0)
    lst.append(consumption_z0_theta1)
    lst.append(consumption_z1_theta0)
    lst.append(consumption_z1_theta1)
    
    return lst


#=============================================================================
# balance_matrix function 
#=============================================================================

def balance_matrix(x_coordinates,y_coordinates,par,mandate,regime):
    ''' 
    mode is either benchmark or single_payer.
    If mode is single_payer, x_coordinates =! y_coordinates => tax coordinates
    '''
    
    # set shape
    if regime == "single_payer":
        shape = par.Ntax
    else:
        shape = [par.Npremium,par.Npremium]
    
    balance_matrix = np.zeros(shape)
    ins_rate = np.zeros(shape)
    ins_rate_theta0 = np.zeros(shape)
    ins_rate_theta1 = np.zeros(shape)
    mean_consumption = np.zeros(shape)
    consumption_z0_theta0 = np.zeros(shape)
    consumption_z0_theta1 = np.zeros(shape)
    consumption_z1_theta0 = np.zeros(shape)
    consumption_z1_theta1 = np.zeros(shape)
    
    if regime == "single_payer":
        
        mean_savings = np.zeros(shape)
        
        tax_grid = x_coordinates
        
        for i, tax in enumerate(tax_grid):
            
            return_lst = get_single_payer_balance(tax)
                
            balance_matrix[i] = return_lst[0]
            mean_savings[i] = return_lst[1]           
            mean_consumption = return_lst[2]
            #c_theta0 = return_lst[3]
            #c_theta1 = return_lst[4]
                
            print(f'Current iteration ({i}) with tax = {tax}  ')
            print('===========================================')

    else: 
        
        
        for i, x in enumerate(x_coordinates):
            for j, y in enumerate(y_coordinates):
                
                return_lst = get_balance(x,y, mandate)
                
                balance_matrix[i,j] = return_lst[0]
                ins_rate[i,j] = return_lst[1]
                ins_rate_theta0[i,j] = return_lst[2]
                ins_rate_theta1[i,j] = return_lst[3]
                mean_consumption = return_lst[4] 
                consumption_z0_theta0[i,j] = return_lst[5]
                consumption_z0_theta1[i,j] = return_lst[6]
                consumption_z1_theta0[i,j] = return_lst[7]
                consumption_z1_theta1[i,j] = return_lst[8]
                
                
                print(f'Current iteration ({i,j}) with premium_0 = {x} | premium_1 = {y} ')
            
    return balance_matrix, ins_rate, ins_rate_theta0, ins_rate_theta1,mean_consumption,consumption_z0_theta0,consumption_z0_theta1,consumption_z1_theta0,consumption_z1_theta1


#=============================================================================
# save dfs to file 
#=============================================================================
def save_alldfs_to_file(version, results, premium_0_grid, premium_1_grid):
    ''' 
    version is either benchmark_case, ACA_reform or single_payer
    '''
    
    #construct filenames
    if version == "single_payer":
        
        tax_grid = premium_0_grid
        
        # balance
        filename_balance = 'balance_matrix_dim_{0}_{1}.csv'.format(tax_grid.size, version) # premium_0_grid as first inout to function eqauls tax_grid
        # savings
        filename_savings = 'savings_dim_{0}_{1}.csv'.format(tax_grid.size,  version)
        # consumption
        filename_consumption = 'consumption_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        
        # convert all to df
        balance_df = pd.DataFrame(data = np.round(results[0],3),
                                index = [str(np.round(i,2)) 
                                for i in tax_grid])
        savings_df = pd.DataFrame(data = np.round(results[1],3),
                                index = [str(np.round(i,2)) 
                                for i in tax_grid])
        consumption_df = pd.DataFrame(data = np.round(results[2],3),
                                index = [str(np.round(i,2)) 
                                for i in tax_grid])

        # save all dfs to folder
        balance_df.to_csv(filename_balance)
        savings_df.to_csv(filename_savings)
        consumption_df.to_csv(filename_consumption)

    
    else:
        # balance
        filename_balance = 'balance_matrix_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # insrate
        filename_insrate = 'insrate_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # insrate_theta0
        filename_insrate_theta0 = 'insrate_theta0_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # insrate_theta1
        filename_insrate_theta1 = 'insrate_theta1_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # consumption
        filename_consumption = 'consumption_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # c_z0
        filename_consumption_z0_theta0 = 'consumption_z0_theta0_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # c_z1
        filename_consumption_z0_theta1 = 'consumption_z0_theta1_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
         # c_theta0
        filename_consumption_z1_theta0 = 'consumption_z1_theta0_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
        # c_theta1
        filename_consumption_z1_theta1 = 'consumption_z1_theta1_dim_{0}x{1}_{2}.csv'.format(premium_0_grid.size, premium_0_grid.size, version)
    
        # convert all to df
        balance_df = pd.DataFrame(data = np.round(results[0],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        insrate_df = pd.DataFrame(data = np.round(results[1],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        insrate_theta0_df = pd.DataFrame(data = np.round(results[2],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        insrate_theta1_df = pd.DataFrame(data = np.round(results[3],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        
        consumption_df = pd.DataFrame(data = np.round(results[4],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        consumption_z0_theta0_df = pd.DataFrame(data = np.round(results[5],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        consumption_z0_theta1_df = pd.DataFrame(data = np.round(results[6],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        consumption_z1_theta0_df = pd.DataFrame(data = np.round(results[7],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        consumption_z1_theta1_df = pd.DataFrame(data = np.round(results[8],3),
                                index = [str(np.round(i,2)) 
                                for i in premium_0_grid],
                                columns = [str(np.round(i,2)) 
                                for i in premium_1_grid])
        


        # save all dfs to folder
        balance_df.to_csv(filename_balance)
        insrate_df.to_csv(filename_insrate)
        insrate_theta0_df.to_csv(filename_insrate_theta0)
        insrate_theta1_df.to_csv(filename_insrate_theta1)
        consumption_df.to_csv(filename_consumption)
        consumption_z0_theta0_df.to_csv(filename_consumption_z0_theta0)
        consumption_z0_theta1_df.to_csv(filename_consumption_z0_theta1)
        consumption_z1_theta0_df.to_csv(filename_consumption_z1_theta0)
        consumption_z1_theta1_df.to_csv(filename_consumption_z1_theta1)



#=============================================================================
# balance_matrix function for government tax optimization 
#=============================================================================

def get_single_payer_balance(tax):
        
    # set up model
    model = model_dc_multidim()
    model.setup()
    model.create_grids()
    par = model.par
    
    # overwrite tax and regime parameters
    par.tax = tax
    par.single_payer_regime = 1
    par.lambdaa_0 = 0
    par.lambdaa_1 = 0
    par.mandate = 0
    
    # solve model for given paramters
    model.solve_VFI_inf_horizon()
    
    # simulate from model 
    sim = model.simulate()
    
    #transform dataset
    df = sim.df 
    df.columns = ['delta m', 'm_t', 'm_t-1', 'c', 'a', 'p_lag', 'claims','theta']

    # create sub dfs
    df_theta0 = df.loc[df['theta'] == 0]
    df_theta1 = df.loc[df['theta'] == 1]

    # collect claims given type shape (10000,)
    claims_type0 = df_theta0['claims']
    claims_type1= df_theta1['claims']

    # caluclate revenue from premiums
    n = par.grid_theta.size
    Revenue = par.simN*par.tax

    # calculate costs from claims 
    Cost = (np.sum(par.Y*claims_type0) +  np.sum(par.Y*claims_type1 ))

    # caluclate balance
    balance = par.O + Revenue - Cost
    
    # calculate other 
    c_theta0 = df_theta0["c"].mean()
    c_theta1= df_theta1["c"].mean()

    mean_savings = df["a"].mean() 
    mean_consumption = df["c"].mean() 

    
    # return list
    lst = []
    lst.append(balance)
    lst.append(mean_savings)
    lst.append(mean_consumption)
    lst.append(c_theta0)
    lst.append(c_theta1)

    
    return lst




# =============================================================================
def operating_costs(pool_size,par):
    # for making operating costs depening on number of people in the portfolio
    return par.O + np.sqrt(pool_size)
# 
# =============================================================================
