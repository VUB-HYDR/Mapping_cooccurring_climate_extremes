# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:00:00 2023

@author: Derrick Muheki
"""

import os
import global_funcs as fn
import xarray as xr
import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.stats import mode


# Start recording processing time
start_time = datetime.now()

#%% SETTING UP THE CURRENT WORKING DIRECTORY; for both the input and output folders
cwd = os.getcwd()

#%% LIST OF PARAMETERS FOR A NESTED LOOP


scenarios_of_datasets = ['historical','rcp26', 'rcp60', 'rcp85']

time_periods_of_datasets = ['1861-1910', '1956-2005', '2050-2099']

extreme_event_categories = ['floodedarea', 'driedarea', 'heatwavedarea', 'burntarea', 'tropicalcyclonedarea', 'cropfailedarea']

compound_events = [['floodedarea', 'burntarea'], ['floodedarea', 'heatwavedarea'], ['heatwavedarea', 'burntarea'], 
                   ['heatwavedarea', 'cropfailedarea'], ['driedarea', 'burntarea'], ['driedarea', 'heatwavedarea'],
                   ['cropfailedarea','burntarea'], ['floodedarea', 'driedarea'], ['floodedarea', 'cropfailedarea'],
                   ['driedarea', 'cropfailedarea'], ['heatwavedarea', 'tropicalcyclonedarea'], ['burntarea', 'tropicalcyclonedarea'],
                   ['floodedarea', 'tropicalcyclonedarea'], ['driedarea', 'tropicalcyclonedarea'], ['cropfailedarea', 'tropicalcyclonedarea']]



# list of bias-adjusted Global Climate Models available for all the Impact Models
gcms = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']



#%% MASK: FOR UNIFORMITY IN THE PLOTS, A MASKING WILL BE DONE ON ALL THE EXTREME EVENTS DATA TO ENSURE NaN VALUES OVER THE OCEAN. 
#         This is because it was noticed that some events such as heatwaves and crop failure had zero values over the ocean instead of NaN values in some Global Impact Models 
#         FOR OUR CASE, ONLY FOR MASKING PURPOSES, FLOODS DATA FROM THE 'ORCHIDEE' IMPACT MODEL WILL BE USED TO MASK ALL THE OTHER EXTREME EVENTS. 

floods_data = os.path.join(cwd, 'floodedarea') #folder containing all flooded area data for all available impact models
floods_orchidee_dataset = os.path.join(floods_data, 'orchidee') # considering orchidee data

# For the period 1861 UNTIL 2005
one_file_historical_floods_orchidee_data =  os.path.join(floods_orchidee_dataset, 'orchidee_gfdl-esm2m_historical_floodedarea_global_annual_landarea_1861_2005.nc4')
start_year_of_historical_floods_data = fn.read_start_year(one_file_historical_floods_orchidee_data) #function to get the starting year of the data from file name
historical_floods_orchidee_data = fn.nc_read(one_file_historical_floods_orchidee_data, start_year_of_historical_floods_data, 'floodedarea', time_dim=True) # reading netcdf files based on variables
occurrence_of_historical_floods = fn.extreme_event(historical_floods_orchidee_data) # occurrence of floods..as a boolean...true or false. returns 1 where floods were recorded in that location during that year

# For the period 2006 UNTIL 2099
one_file_projected_floods_orchidee_data =  os.path.join(floods_orchidee_dataset, 'orchidee_gfdl-esm2m_rcp26_floodedarea_global_annual_landarea_2006_2099.nc4')
start_year_of_projected_floods_data = fn.read_start_year(one_file_projected_floods_orchidee_data) #function to get the starting year of the data from file name
projected_floods_orchidee_data = fn.nc_read(one_file_projected_floods_orchidee_data, start_year_of_projected_floods_data, 'floodedarea', time_dim=True) # reading netcdf files based on variables
occurrence_of_projected_floods = fn.extreme_event(projected_floods_orchidee_data) # occurrence of floods..as a boolean...true or false. returns 1 where floods were recorded in that location during that year


# mask for map uniformity purposes: to apply NaN values over the ocean
mask_for_historical_data = occurrence_of_historical_floods
mask_for_projected_data = occurrence_of_projected_floods

#%% FILE WITH ENTIRE GLOBE GRID CELL AREA in m2

entire_globe_grid_cell_areas_in_netcdf = 'entire_globe_grid_cell_areas.nc'

entire_globe_grid_cell_areas_in_xarray = (xr.open_dataset(entire_globe_grid_cell_areas_in_netcdf)).to_array()


#%% FULL DATASETS

occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] # Where order of list is the six extreme event catergories and within these: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5 and extreme event name] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']   
no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] # Where order of list is the six extreme event catergories and within these: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5 and extreme event name] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']   


for extreme_event in extreme_event_categories:
    
    # Name of extreme event for graphs/plots
    extreme_event_name = fn.event_name(extreme_event)
    
    # All GCMs data on timeseries (50-year periods) of occurrence of extreme event for all scenarios
    all_gcms_timeseries_50_years_of_occurrence_of_events = []
    
    # NOTE:
    # Full dataset of occurrence of an extreme event considering all impact models driven by the same GCM, for all GCMs
    # Where: order of list [[], [], [], []] is order of the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
    #  occurrence of extreme event from 1861 until 1910
    
    # 0. Extreme event occurrence
    all_gcm_data_about_occurrence_of_extreme_event_from_1861_until_1910 = [[],[],[],[]] # Order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
    all_gcm_data_about_occurrence_of_extreme_event_from_1956_until_2005 = [[],[],[],[]]
    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26 = [[],[],[],[]] # Under RCP2.6
    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60 = [[],[],[],[]] # Under RCP6.0
    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85 = [[],[],[],[]] # Under RCP8.5
    
    # 1. Total no. of years with an extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMS.
    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910 = [[],[],[],[]] # Order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005 = [[],[],[],[]]
    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26 = [[],[],[],[]] # Under RCP2.6
    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60 = [[],[],[],[]] # Under RCP6.0
    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85 = [[],[],[],[]] # Under RCP8.5
    
    
    # Iterating through each of the 4 GCMs, such that data from impact models driven by the same GCM is collected, after which it is appended to the LISTS ABOVE that include impact data from all the available GCMs
    for i in range(len(gcms)):
        
        # Considering all the impact models (with the same driving GCM)
        extreme_event_impact_models = fn.impact_model(extreme_event, gcms[i])
        
        # Timeseries (entire time periods) of occurrence of an extreme event for all scenarios
        timeseries_of_occurrence_of_extreme_event = []
        
        
        for scenario in scenarios_of_datasets:
            

            if scenario == 'historical':
                
                all_impact_model_data_about_an_extreme_event_from_1861_until_1910 = [] # List with occurrence of an extreme event during 50 years accross the multiple impact models driven by the same GCM
                all_impact_model_data_about_an_extreme_event_from_1956_until_2005 = []
                
                all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910 = [] # List with total no. of years with occurrence of an extreme event accross the multiple impact models driven by the same GCM
                all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005 = []
                

                
                # Dataset of extreme event occurrence per grid for a given scenario and given time period considering all impact models driven by same GCM
                extreme_event_occurrence_dataset = fn.extreme_event_occurrence(extreme_event, extreme_event_impact_models, scenario)
                
                
                for impact_model_data in extreme_event_occurrence_dataset[0]:  # [0] reresents the occurrence_of_extreme_event_considering_ensemble_of_gcms_from_1861_until_2005
                    

                        
                        if len(impact_model_data) == 0: # checking for an empty array representing no data
                            print('No data available on occurrence of extreme event for selected impact model during the period '+ time_periods_of_datasets[0] + 'or' + time_periods_of_datasets[1] + '\n')
                        else:
                            
                            ## EARLY INDUSTRIAL/ HISTORICAL / 50 YEARS / FROM 1861 UNTIL 1910
                            extreme_event_from_1861_until_1910_unmasked =  impact_model_data[0:50] # (UNMASKED) occurrence of extreme event considering one impact model 
                            # Mask extreme event data for map uniformity purposes: to apply NaN values over the ocean
                            extreme_event_from_1861_until_1910 = xr.where(np.isnan(mask_for_historical_data[0:50]), np.nan, extreme_event_from_1861_until_1910_unmasked) # (MASKED) occurrence of event considering one impact model  
                            all_impact_model_data_about_an_extreme_event_from_1861_until_1910.append(extreme_event_from_1861_until_1910)
                            
                            # Total no. of years with occurrence of extreme event from 1861 until 1910. (Annex to empty array above to later on determine the average total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM)
                            no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910 = fn.total_no_of_years_with_occurrence_of_extreme_event(extreme_event_from_1861_until_1910)
                            all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910.append(no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910) # Appended to the list above with total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM
                            
                        
                        
                            ## PRESENT DAY / HISTORICAL / 50 YEARS / FROM 1956 UNTIL 2005
                            extreme_event_from_1956_until_2005_unmasked =  impact_model_data[95:] # (UNMASKED) occurrence of extreme event considering one impact model 
                            # Mask extreme event data for map uniformity purposes: to apply NaN values over the ocean
                            extreme_event_from_1956_until_2005 = xr.where(np.isnan(mask_for_historical_data[95:]), np.nan, extreme_event_from_1956_until_2005_unmasked) # (MASKED) occurrence of event considering one impact model  
                            all_impact_model_data_about_an_extreme_event_from_1956_until_2005.append(extreme_event_from_1956_until_2005)
                            
                            # Total no. of years with occurrence of extreme event  (Annex to empty array above to later on determine the average total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM)
                            no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005 = fn.total_no_of_years_with_occurrence_of_extreme_event(extreme_event_from_1956_until_2005)
                            all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005.append(no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005) # Appended to the list above with total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM
                            

                # Early Industrial     
                
                # Extreme event occurence; across the different impact models driven by the same GCM to one list
                all_gcm_data_about_occurrence_of_extreme_event_from_1861_until_1910[i].append(all_impact_model_data_about_an_extreme_event_from_1861_until_1910)
                # Append total no. of years with occurrence of extreme events from 1861 until 1910 across the different impact models driven by the same GCM to one list
                all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910[i].append(all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910)
                
                # Present day
                # Extreme event occurence; across the different impact models driven by the same GCM to one list
                all_gcm_data_about_occurrence_of_extreme_event_from_1956_until_2005[i].append(all_impact_model_data_about_an_extreme_event_from_1956_until_2005)
                # Append total no. of years with occurrence of extreme events from 1956 until 2005 across the different impact models driven by the same GCM to one list
                all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005[i].append(all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005)        
                
            else:
                
                # End-of-century scenarios (2050-2099)
                
                all_impact_model_data_about_an_extreme_event_from_2050_until_2099 = [] # List with occurrence of an extreme event during 50 years accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099 = [] # List with total no. of years with occurrence of an extreme event accross the multiple impact models driven by the same GCM

                all_impact_model_data_about_length_of_spell_with_occurrence_of_extreme_event_from_2050_until_2099 = [] # List with length of spell of an extreme event accross the multiple impact models driven by the same GCM
                
                # Dataset of extreme event occurrence per grid for a given scenario and given time period considering all impact models driven by same GCM
                extreme_event_occurrence_dataset = fn.extreme_event_occurrence(extreme_event, extreme_event_impact_models, scenario)
                
                
                for impact_model_data in extreme_event_occurrence_dataset[1]:  # [1] reresents the occurrence_of_extreme_event_considering_ensemble_of_gcms_from_2006_until_2099
                                        
                        if len(impact_model_data) == 0: # checking for an empty array representing no data
                            print('No data available on occurrence of extreme event for selected impact model during the period '+ time_periods_of_datasets[2] + '\n')
                        else:
                            
                            ## EARLY INDUSTRIAL/ HISTORICAL / 50 YEARS / FROM 2050 UNTIL 2050
                            extreme_event_from_2050_until_2099_unmasked =  impact_model_data[44:] # (UNMASKED) occurrence of extreme event considering one impact model 
                            # Mask extreme event data for map uniformity purposes: to apply NaN values over the ocean
                            extreme_event_from_2050_until_2099 = xr.where(np.isnan(mask_for_projected_data[44:]), np.nan, extreme_event_from_2050_until_2099_unmasked) # (MASKED) occurrence of event considering one impact model  
                            all_impact_model_data_about_an_extreme_event_from_2050_until_2099.append(extreme_event_from_2050_until_2099)
                            
                            # Total no. of years with occurrence of extreme event from 2050 until 2099. (Annex to empty array above to later on determine the average total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM)
                            no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099 = fn.total_no_of_years_with_occurrence_of_extreme_event(extreme_event_from_2050_until_2099)
                            all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099.append(no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099) # Appended to the list above with total no. of years with occurrence of extreme event accross the multiple impact models driven by the same GCM                          
               
                
                if scenario == 'rcp26':
                    
                    # Extreme event occurence; across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26[i].append(all_impact_model_data_about_an_extreme_event_from_2050_until_2099)
                    # Append total no. of years with occurrence of extreme events from 2050 until 2099 across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26[i].append(all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099)
                              
                if scenario == 'rcp60':
                    # Extreme event occurence; across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60[i].append(all_impact_model_data_about_an_extreme_event_from_2050_until_2099)
                    # Append total no. of years with occurrence of extreme events from 2050 until 2099 across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60[i].append(all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099)
                             
                if scenario == 'rcp85':
                    # Extreme event occurence; across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85[i].append(all_impact_model_data_about_an_extreme_event_from_2050_until_2099)
                    # Append total no. of years with occurrence of extreme events from 2050 until 2099 across the different impact models driven by the same GCM to one list
                    all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85[i].append(all_impact_model_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099)
                
     
    position_of_extreme_event_in_list = extreme_event_categories.index(extreme_event)
    
    ## Append data from all the gcms and scenatios such that: The order of list is [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5 , extreme event name] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']    
    
    # Occurrence of extreme event across 50 yerar period per grid for a given scenario and given time period considering an ensemble of GCMS
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_occurrence_of_extreme_event_from_1861_until_1910) # # Early indutrial # Where order of list is [early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_occurrence_of_extreme_event_from_1956_until_2005) # Present-day
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26) # Under RCP2.6
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60) # Under RCP6.0
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85) # Under RCP8.5
    occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(extreme_event_name)
    
    # Total no. of years with an extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMS
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1861_until_1910) # # Early indutrial # Where order of list is [early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_1956_until_2005) # present day
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp26) # RCP2.6
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp60) # RCP6.0
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(all_gcm_data_about_no_of_years_with_occurrence_of_extreme_event_from_2050_until_2099_under_rcp85) # RCP8.5
    no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[position_of_extreme_event_in_list].append(extreme_event_name)                                                                                      

    
   
#%% PROPENSITY 
    
# Compute single-gridbox propensity
# Propensity = 1 + ln(tot. number of occurrences of extreme "X"/average number of occurrences of all extremes)
# Where average number of occurrences of all extremes is the sum of all extreme event occurrences at that gridpoint 
# divided by 6 (the classes of extreme events). Using 1 + the natural logarithm nicely separates events more 
# frequent than the average or less frequent than the average as having a propensity > 1 or < 1, with propensity = 1 
# meaning that the chosen event class perfectly matches the average extreme event frequency at that gridpoint.


# Initialize storage for propensities
propensity_data_per_extreme_event = [[None] * 5 for _ in range(len(extreme_event_categories))]

scenarios = ['Early-industrial','Present Day', 'RCP2.6', 'RCP6.0', 'RCP8.5']

# Loop through scenarios
for scenario_index, scenario in enumerate(scenarios):  # Exclude the last item (event name)
    if scenario == 'RCP8.5':
        print("Processing scenario: RCP8.5 (with missing data for crop failures).")
        
    # Outer list to store GCM-level averages for each extreme event
    all_gcm_averages = [[] for _ in range(len(extreme_event_categories))]
    
    # Loop through GCMs
    for gcm_index, gcm in enumerate(gcms):
        print(f"Processing Scenario={scenario}, GCM={gcm}")

        # Placeholder for GCM-level propensities
        gcm_propensities = [[] for _ in range(len(extreme_event_categories))]

        # Loop through extreme events
        for extreme_event_index, extreme_event in enumerate(extreme_event_categories):
            if extreme_event == 'cropfailedarea' and scenario == 'RCP8.5':
                print(f"Skipping {extreme_event} for {scenario} due to missing data.")
                continue

            # Access pre-calculated impact models with number of years for the current extreme event, scenario, and GCM
            try:
                number_of_years_data_for_gcm = no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[
                    extreme_event_index][scenario_index][gcm_index][0]
            except IndexError:
                print(f"Data missing for {extreme_event}, Scenario={scenario}, GCM={gcm}")
                continue

            if not number_of_years_data_for_gcm or len(number_of_years_data_for_gcm) == 0:
                print(f"No data for {extreme_event}, Scenario={scenario}, GCM={gcm}")
                continue

            # Generate combinations of number of years for all extreme events
            all_events_years_data_by_event = [
                no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[event_index][scenario_index][gcm_index][0]
                for event_index in range(len(extreme_event_categories))
            ]

            # Initialize placeholder for the running average of impact model propensities
            running_average_propensity = None           

            # Generate cross-category combinations
            for cross_category_combination in product(*[data for data in all_events_years_data_by_event if data]):
                # Flatten and validate the data within the combination
                valid_data = [
                    data for data in cross_category_combination
                    if isinstance(data, (xr.DataArray, xr.Dataset)) and data.size > 0
                ]
                if not valid_data:
                    print("No valid data for this combination.")
                    continue

                # Concatenate valid xarray objects to calculate the denominator
                try:
                    denominator = xr.concat(valid_data, dim='extreme_events').mean(dim='extreme_events', skipna=True)
                except Exception as e:
                    print(f"Error in concatenating data: {e}")
                    continue

                # Flatten `number_of_years_data_for_gcm` to create a 2D or 3D DataArray
                for data_index, data_array in enumerate(number_of_years_data_for_gcm):
                    try:
                        # Convert the current array to an xarray.DataArray
                        flat_numerator = xr.DataArray(data_array)

                        # Align dimensions in the script before calling `propensity`
                        try:
                            aligned_denominator = denominator.broadcast_like(flat_numerator)
                        except ValueError as e:
                            print(f"Dimension alignment failed for data_index {data_index}: {e}")
                            continue

                        # Calculate propensity for this specific combination
                        try:
                            propensity = fn.propensity(flat_numerator, aligned_denominator)
                            
                            # Replace inf and -inf with NaN
                            propensity = propensity.where(~np.isinf(propensity), other=np.nan)
                        
                            # Update the running average
                            if running_average_propensity is None:
                                running_average_propensity = propensity
                            else:
                                combined_propensity_with_running_average = xr.concat([running_average_propensity, propensity], dim="combined_propensity_with_running_average")
                                running_average_propensity = combined_propensity_with_running_average.mean(dim="combined_propensity_with_running_average", skipna = True)
                                

                        except Exception as e:
                            print(f"Error in propensity calculation for data_index {data_index}: {e}")
                            continue

                    except Exception as e:
                        print(f"Error in processing data_index {data_index}: {e}")
                        continue

            # Average propensities across all combinations for this extreme event and impact model
            # Finalize the average for the current impact model-level propensities
            if running_average_propensity is not None:
                gcm_propensities[extreme_event_index].append(running_average_propensity)
            else:
                gcm_propensities[extreme_event_index].append(None)
        
        # Append GCM-level averages for each extreme event to the outer list
        for extreme_event_index, gcm_average in enumerate(gcm_propensities):
            if gcm_average:
                all_gcm_averages[extreme_event_index].append(gcm_average[0])

    # Ensemble averaging for each extreme event across GCMs
    for extreme_event_index, gcm_averages in enumerate(all_gcm_averages):
        if gcm_averages:
            try:
                ensemble_average_propensity = xr.concat(
                    [p for p in gcm_averages if p is not None],
                    dim='gcms'
                ).mean(dim='gcms', skipna=True)
                propensity_data_per_extreme_event[extreme_event_index][scenario_index] = ensemble_average_propensity
            except Exception as e:
                print(f"Error in ensemble averaging for {extreme_event_index}: {e}")
                propensity_data_per_extreme_event[extreme_event_index][scenario_index] = None
        else:
            propensity_data_per_extreme_event[extreme_event_index][scenario_index] = None

# Plot the propensity data for all extreme events and scenarios
fn.plot_propensity_of_extreme_events(propensity_data_per_extreme_event, extreme_event_categories)





#%% C0-0CCURRENCE RATIO & LENGTH OF SPELL OF EXTREME EVENTS      
               
# Compute co-occurrence ratio
# Co-occurrence ratio = 1 + ln(tot. number of extreme events occuring not in isolation/tot. number of occurrences of extreme events occurring in isolation). 
# Here, by "occurring in isolation" I mean that there is a single class of extremes occurring at a given gridbox in a given year, while "occuring not in isolation" means that there are two or more extremes occurring at a given gridbox in a given year. 
# As for propensity, the cutoff value is 1. A co-occurrence ratio > 1 means that there are more co-occurring extremes than isolated extremes, and vice-versa for a ratio < 1.           

# The loop below analyses each GCM, timeperiod and impact model separately.
# One could also code this in a much simpler fashion if one does not wish to 
# loop over all possible configurations. I commented below on the part that 
# actually computes the metric. Note that for the co-occurrence ratio it
# does not make sense to aggergate occurrences across different impact
# models, as then one would count the same event class occurring in
# different impact models as an extreme events occuring not in 
# isolation. Since there is a huge number of possible combinations of
# different impact models for different event classes, in this loop I am
# simply taking the first impact model from each GCM.

                


full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario = [[[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]] # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the six extreme event catergories 


for extreme_event_occurrence in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models)): # one event's data out of the six categories of extremes
    
    # Event name
    extreme_event_category = occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][5] # name of extreme event
       
    for scenario in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence]) - 1): # -1 is to avoid the last member of the list which is the extreme event name (String)
        
        if extreme_event_category == 'Crop Failures' and scenario == 4:
            print('No data available on occurrence of crop failures for selected impact model during the period '+ time_periods_of_datasets[2] + ' under RCP 8.5 \n')
        else:
            
            
            for gcm in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][scenario])):
                
                gcm_data_on_occurrence_of_extreme_event = occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][scenario][gcm][0] #gcm[0] because it is a list with one element (that is a list of arrays)
                
                if extreme_event_occurrence <= 5: # To avoid the name of the extreme event in the list
                    
                    #full_dataset_occurrence_of_extreme_event_considering_all_gcms_per_scenario[gcm][scenario].insert(extreme_event_occurrence, gcm_data_on_occurrence_of_extreme_event)
                
                    full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm][scenario].append(gcm_data_on_occurrence_of_extreme_event) 
                    
                    


# CO-OCCURRENCE RATIO FOR EXTREME EVENTS considering all scenarios mapped on a single plot

# Initialize the structure to store co-occurrence ratios
cooccurrence_ratios_per_scenario = [[[] for _ in range(5)] for _ in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario))]  # 5 scenarios for 4 GCMs

# Loop through GCMs and scenarios
for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    gcm_data_on_occurrence_of_extreme_events = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]
    
    for scenario in range(len(gcm_data_on_occurrence_of_extreme_events)):
        scenario_data = gcm_data_on_occurrence_of_extreme_events[scenario]

        if scenario == 4:  # Handle RCP8.5 with missing crop failures
            print(f"No data available for crop failures in RCP8.5 for GCM {gcm}.")
            continue

        # Placeholder for GCM-level co-occurrence ratios
        running_average_cooccurrence_ratios = None

        # Generate combinations of impact models across extreme events
        for cross_category_impact_models in itertools.product(*scenario_data):
            # Calculate total occurrences per year across impact models
            total_occurrences_per_year = xr.concat(cross_category_impact_models, dim='impact_models').sum(dim='impact_models', skipna=True)

            # Calculate isolated occurrences (only 1 event per grid point)
            isolated_occurrences = xr.where(
                total_occurrences_per_year == 1, 1, xr.where(np.isnan(total_occurrences_per_year), np.nan, 0)
            ).sum(dim='time', skipna=True)

            # Calculate co-occurring events (more than 1 event per grid point)
            cooccurring_events = xr.where(
                total_occurrences_per_year > 1, total_occurrences_per_year, xr.where(np.isnan(total_occurrences_per_year), np.nan, 0)
            ).sum(dim='time', skipna=True)

            # Compute the co-occurrence ratio for this combination
            try:
                cooccurrence_ratio = fn.cooccurrence_ratio(cooccurring_events, isolated_occurrences)
                
                # Update the running average
                if running_average_cooccurrence_ratios is None:
                    running_average_cooccurrence_ratios = cooccurrence_ratio
                else:
                    combined_cooccurrence_ratio_with_running_average = xr.concat([running_average_cooccurrence_ratios, cooccurrence_ratio], dim="combined_cooccurrence_ratio_with_running_average")
                    
                    # Mask inf values as nan. These inf are a result of log error (inf) in the co-occurrence ratio calculation within some grid cells
                    masked_ratios = combined_cooccurrence_ratio_with_running_average.where(
                        ~np.isinf(combined_cooccurrence_ratio_with_running_average), other=np.nan)

                    # Compute the running mean, skipping NaN and inf values
                    running_average_cooccurrence_ratios = masked_ratios.mean(dim="combined_cooccurrence_ratio_with_running_average", skipna = True)
                    
            except Exception as e:
                print(f"Error computing co-occurrence ratio: {e}")
                running_average_cooccurrence_ratios.append(None)

        # Average co-occurrence ratio across impact models for this GCM
        if running_average_cooccurrence_ratios is not None:
            try:
                cooccurrence_ratios_per_scenario[gcm][scenario].append(running_average_cooccurrence_ratios)
            except Exception as e:
                print(f"Error averaging co-occurrence ratio for GCM {gcm}, scenario {scenario}: {e}")
                cooccurrence_ratios_per_scenario[gcm][scenario].append(None)


# Aggregate co-occurrence ratios across all GCMs
average_cooccurrence_ratios_per_scenario = []

for scenario_index in range(5):  # For each scenario
    scenario_cooccurrence_ratios = [
        cooccurrence_ratios_per_scenario[gcm][scenario_index] 
        for gcm in range(len(cooccurrence_ratios_per_scenario))
    ]
    
    # Flatten the list
    scenario_cooccurrence_ratios_flattened = [
        ratio 
        for gcm_ratios in scenario_cooccurrence_ratios 
        for ratio in gcm_ratios 
        if ratio is not None
    ]
    
    if scenario_cooccurrence_ratios_flattened:
        try:
            # Concatenate along 'gcms' dimension
            concatenated_ratios = xr.concat(
                scenario_cooccurrence_ratios_flattened, dim='gcms'
            )
            
            # Mask inf values as nan
            masked_ratios = concatenated_ratios.where(~np.isinf(concatenated_ratios), other=np.nan)
            
            # Compute mean while skipping nan (and inf)
            average_cooccurrence_ratio = masked_ratios.mean(dim='gcms', skipna=True)
        except Exception as e:
            print(f"Error averaging co-occurrence ratio for scenario {scenario_index}: {e}")
            average_cooccurrence_ratio = None
    else:
        average_cooccurrence_ratio = None

    average_cooccurrence_ratios_per_scenario.append(average_cooccurrence_ratio)


# Plot the average co-occurrence ratios
fn.plot_cooccurrence_ratio_considering_all_gcms_in_a_single_plot(average_cooccurrence_ratios_per_scenario, mask_for_historical_data[0])




#%%  LENGTH OF SPELLS OF INDIVIDUAL EXTREME EVENTS -- AGGREGATED ACROSS MODELS -- CONSIDERING ALL IMPACT MODELS AND THEIR DRIVING GCMs

# Initialize the structure to store 95th percentiles for each event and scenario
quantile_95th_per_event_per_scenario = [[None for _ in range(5)] for _ in range(len(extreme_event_categories))]  # 5 scenarios

# Process each extreme event individually
for event_index, event_name in enumerate(extreme_event_categories):
    print(f"Processing extreme event: {event_name}")
    
    for scenario_index in range(5):  # Loop through scenarios
        if event_index == 5 and scenario_index == 4:  # Skip crop failures under RCP8.5
            print(f"No data for {event_name} under RCP8.5; skipping.")
            continue

        # Initialize a list to collect spell lengths for this event and scenario
        lengths_to_pool = []

        for gcm_index, gcm_data in enumerate(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario):
            scenario_data = gcm_data[scenario_index]  # Extract scenario data for this GCM
            
            # Extract data for the current event
            event_data = scenario_data[event_index]
            
            for model_data in event_data:  # Loop through impact models
                # Compute the length of spells for the current model
                lengths_of_spells = fn.length_of_spell_with_occurrence_of_extreme_event(model_data)
                
                # Replace non-positive values with NaN to exclude invalid data
                lengths_of_spells = xr.where(lengths_of_spells > 0, lengths_of_spells, np.nan)
                
                # Append spell lengths to the pooled list
                lengths_to_pool.append(lengths_of_spells)

        # Compute the 95th percentile for the current event and scenario
        if lengths_to_pool:
            try:
                # Concatenate all lengths across 'combinations' (representing models)
                concatenated_lengths = xr.concat(lengths_to_pool, dim='combinations')

                # Compute the 95th quantile for each (lat, lon) grid cell by collapsing 'combinations' and 'years'
                quantile_95th = concatenated_lengths.reduce(
                    np.nanquantile, q=0.95, dim=('combinations', 'time')
                )

                # Assign the 95th quantile to the appropriate location in the structure
                quantile_95th_per_event_per_scenario[event_index][scenario_index] = quantile_95th
            except Exception as e:
                print(f"Error calculating 95th quantile for event {event_name}, scenario {scenario_index}: {e}")
                quantile_95th_per_event_per_scenario[event_index][scenario_index] = None
        else:
            quantile_95th_per_event_per_scenario[event_index][scenario_index] = None

# Plot the results
fn.plot_quantile_95th_of_length_of_spell_with_occurrence_considering_all_gcms_and_showing_all_extremes(
    quantile_95th_per_event_per_scenario, extreme_event_categories
)

        
#%% LENGTH OF SPELLS FOR COMPOUND EVENT PAIRS-- AGGREGATED ACROSS MODELS -- CONSIDERING ALL IMPACT MODELS AND THEIR DRIVING GCMs

# Initialize the structure to store pooled lengths of spells across all GCMs and scenarios
# Generate all possible pairs of extreme events
event_pairs = list(itertools.combinations(range(len(extreme_event_categories)), 2))

# Initialize the structure to store 95th percentiles for each event pair and scenario
quantile_95th_per_event_pair_per_scenario = [
    [None for _ in range(5)]  # 5 scenarios
    for _ in range(len(event_pairs))  # One entry for each event pair
]

# Process each event pair individually
for pair_index, (event_1, event_2) in enumerate(event_pairs):
    print(f"Processing extreme event pair: {extreme_event_categories[event_1]} and {extreme_event_categories[event_2]}")
    
    for scenario_index in range(5):  # Loop through scenarios
        if scenario_index == 4 and (event_1 == 5 or event_2 == 5):  # Skip crop failures under RCP8.5
            print(f"No data for crop failures in RCP8.5; skipping pair: {extreme_event_categories[event_1]} and {extreme_event_categories[event_2]}.")
            continue

        # Initialize a list to collect spell lengths for this event pair and scenario
        lengths_to_pool = []

        for gcm_index, gcm_data in enumerate(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario):
            
            for model_1, model_2 in itertools.product(gcm_data[scenario_index][event_1], gcm_data[scenario_index][event_2]): # MODEL 1 EVENT 1 = gcm_data[scenario_index][event_1]  AND MODEL 2 EVENT 2 = gcm_data[scenario_index][event_2]
                # Calculate compound event occurrence
                compound_occurrence = fn.compound_event_occurrence(model_1, model_2)
                
                # Calculate lengths of spells
                lengths_of_spells = fn.length_of_spell_with_occurrence_of_extreme_event(compound_occurrence)
                lengths_of_spells = xr.where(lengths_of_spells > 0, lengths_of_spells, np.nan)  # Replace non-positive values with NaN
                
                # Append spell lengths to the pooled list
                lengths_to_pool.append(lengths_of_spells)

        # Compute the 95th percentile for the current pair and scenario
        if lengths_to_pool:
            try:
                # Concatenate all lengths across 'combinations' (representing models)
                concatenated_lengths = xr.concat(lengths_to_pool, dim='combinations')

                # Compute the 95th quantile for each (lat, lon) grid cell by collapsing 'combinations' and 'time'
                quantile_95th = concatenated_lengths.reduce(
                    np.nanquantile, q=0.95, dim=('combinations', 'time')
                )

                # Assign the 95th quantile to the appropriate location in the structure
                quantile_95th_per_event_pair_per_scenario[pair_index][scenario_index] = quantile_95th
            except Exception as e:
                print(f"Error calculating 95th quantile for pair {extreme_event_categories[event_1]} and {extreme_event_categories[event_2]}, scenario {scenario_index}: {e}")
                quantile_95th_per_event_pair_per_scenario[pair_index][scenario_index] = None
        else:
            quantile_95th_per_event_pair_per_scenario[pair_index][scenario_index] = None

# Plot the quantiles for all extreme event pairs
fn.plot_quantile_95th_of_length_of_spell_with_compound_event_occurrence_considering_all_gcms_and_showing_all_pairs(
    quantile_95th_per_event_pair_per_scenario, extreme_event_categories, event_pairs
)


# Generate names of the compound events (in the same order as the calculations above)
compound_events_names = list(itertools.combinations(extreme_event_categories, 2))

#Plot figure of_average length of spells for only selected compound extreme events
selected_indices = [9, 11, 5, 1] # for the heatwaves and wildfires, heatwaves and crop failures, droughts and heatwaves, and river floods and heatwaves pairs
plot_of_average_length_of_spells_for_selected_events = fn.plot_quantile_95th_length_of_spell_for_event_pairs(
    quantile_95th_per_event_pair_per_scenario,
    compound_events_names,
    selected_indices)

second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_of_average_length_of_spells_for_second_selected_events = fn.plot_quantile_95th_length_of_spell_for_event_pairs_second_plot(quantile_95th_per_event_pair_per_scenario, compound_events_names, second_selected_indices)




#%% PLOT COMBINED EVENT OCCURRENCE/ frequency

summary_of_frequency_for_all_extreme_events_considering_all_GCMs_and_scenarios = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0 and RCP8.5

for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):  # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the six extreme event catergories 
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm] # All GCM data on length of spells
    
    # 95th quantile of length of spell with occurrence of extreme events considering all GCMS and impact models
    occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm = [[],[],[],[],[],[]] # Whereby the order of the extreme events follows the order in 'extreme_event_categories' at the beginning of this script
    
    for scenario in range(len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
         
        for i in range(len(occurrence_of_extreme_events_considering_one_scenario)):
            occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[i].append(occurrence_of_extreme_events_considering_one_scenario[i])
        
    for extreme_event in range(len(occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm)):
        
        for scenario in range(len(occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[extreme_event])):
            
            summary_of_frequency_for_all_extreme_events_considering_all_GCMs_and_scenarios[scenario][gcm].append(occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[extreme_event][scenario])

# Average frequency of extreme events across all GCMs and scenarios
average_frequency_of_all_extreme_events = []
for scenario in summary_of_frequency_for_all_extreme_events_considering_all_GCMs_and_scenarios:
    
    averaged_scenario = []
    
    for i in range(6):
        gcm_data = [gcm[i] for gcm in scenario if len(gcm) > i and gcm[i] is not None]
        
        if gcm_data:
            # Concatenate the data for each impact model across GCMs
            impact_model_averages = [xr.concat([model_data for model_data in impact_model], dim="impact_model").mean(dim="impact_model", skipna=True) for impact_model in zip(*gcm_data)]
            # Average the concatenated data across the impact models
            average_occurrence_of_one_extreme_event_category_across_gcms = xr.concat(impact_model_averages, dim="gcm").mean(dim="gcm", skipna=True)
            average_no_of_years_with_occurrence_of_one_extreme_event_category_accross_gcms_events = fn.total_no_of_years_with_occurrence_of_extreme_event(average_occurrence_of_one_extreme_event_category_across_gcms)
            #Given 50 year period per scenario, we calculate the frequency of these individual extreme events per scenario
            frequency_of_one_extreme_event_category_accross_gcms = average_no_of_years_with_occurrence_of_one_extreme_event_category_accross_gcms_events/50
            averaged_scenario.append(frequency_of_one_extreme_event_category_accross_gcms)
        else:
            averaged_scenario.append(None)
            
    
    average_frequency_of_all_extreme_events.append(averaged_scenario)
        
#Plot figure of combined frequency for all extreme events under different climate scenarios
plot_of_average_frequency_of_all_extreme_events = fn.plot_average_frequency_of_extreme_events_considering_all_gcms_and_showing_all_extremes(average_frequency_of_all_extreme_events, extreme_event_categories)




#%% AVERAGE PROBABILITY OF JOINT OCCURRENCE OF EXTREME EVENTS

# Initialize the structure to store frequencies across all GCMs and scenarios
#summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]] 
summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios = []
# scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0 and RCP8.5

for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]  # All GCM data on occurrence of extreme events
    
    # Initialize for each GCM across all scenarios
    frequency_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario = []  
    
    for scenario in range(len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        
        # Initialize the container for results per scenario and GCM
        frequency_of_compound_events_for_all_possible_combinations_and_GCM = []

        # Handle the missing cropfailedarea data for RCP8.5
        if scenario == 4:  # since 'cropfailedarea' has no data for RCP8.5, we need to insert an empty map where appropriate
            print('No data available for crop failures in RCP8.5; filling with empty map.')
            empty_map = np.nan * xr.zeros_like(occurrence_of_extreme_events_considering_one_scenario[0][0].mean(dim='time'))
            
            # Generate pairs of extreme events using the names of the categories
            event_pairs = list(itertools.combinations(extreme_event_categories, 2))
            
            for pair in event_pairs:
                event_1, event_2 = pair
                
                if 'cropfailedarea' in [event_1, event_2]:
                    # Insert an empty map for pairs involving cropfailedarea
                    frequency_of_compound_events_for_all_possible_combinations_and_GCM.append(empty_map)
                else:
                    # Calculate normally for other pairs
                    index_1 = extreme_event_categories.index(event_1)
                    index_2 = extreme_event_categories.index(event_2)
                    
                    models_event_1 = occurrence_of_extreme_events_considering_one_scenario[index_1]
                    models_event_2 = occurrence_of_extreme_events_considering_one_scenario[index_2]
                    
                    frequency_of_compound_events_across_impact_models = None
                    
                    for model_1, model_2 in itertools.product(models_event_1, models_event_2):
                        # Calculate occurrence of compound events
                        occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                        total_years_with_compound_events = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events)
                        
                        # Calculate the frequency over 50 years
                        frequency_of_compound_events = total_years_with_compound_events / 50
                        
                        if frequency_of_compound_events_across_impact_models is None:
                            frequency_of_compound_events_across_impact_models = frequency_of_compound_events
                        else:
                            frequency_of_compound_events_across_impact_models = xr.concat(
                                [frequency_of_compound_events_across_impact_models, frequency_of_compound_events],
                                dim='impact_models').mean(dim='impact_models', skipna=True)
                    
                    frequency_of_compound_events_for_all_possible_combinations_and_GCM.append(frequency_of_compound_events_across_impact_models)
        
        else:
            # Normal processing for other scenarios
            event_pairs = list(itertools.combinations(range(len(occurrence_of_extreme_events_considering_one_scenario)), 2))
            
            for pair in event_pairs:
                event_1, event_2 = pair
                
                models_event_1 = occurrence_of_extreme_events_considering_one_scenario[event_1]
                models_event_2 = occurrence_of_extreme_events_considering_one_scenario[event_2]
                
                frequency_of_compound_events_across_impact_models = None
                
                for model_1, model_2 in itertools.product(models_event_1, models_event_2):
                    # Calculate occurrence of compound events
                    occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                    total_years_with_compound_events = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events)
                    
                    # Calculate the frequency over 50 years
                    frequency_of_compound_events = total_years_with_compound_events / 50
                    
                    if frequency_of_compound_events_across_impact_models is None:
                        frequency_of_compound_events_across_impact_models = frequency_of_compound_events
                    else:
                        frequency_of_compound_events_across_impact_models = xr.concat(
                            [frequency_of_compound_events_across_impact_models, frequency_of_compound_events],
                            dim='impact_models').mean(dim='impact_models', skipna=True)
                
                frequency_of_compound_events_for_all_possible_combinations_and_GCM.append(frequency_of_compound_events_across_impact_models)
        
        frequency_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario.append(frequency_of_compound_events_for_all_possible_combinations_and_GCM)
    
    summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios.append(
        frequency_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario
    )



# Average frequency of compound events across all GCMs and scenarios
average_frequency_of_all_compound_events = []

# Iterate through the scenarios (Present Day, RCP2.6, RCP6.0, RCP8.5)
for scenario_idx in range(len(summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios[0])):
    
    averaged_scenario = []
    
    # Iterate through the 15 possible combinations of the 6 extreme events
    for i in range(15):  
        gcm_data = []
        
        # Collect data across all GCMs for the given scenario and event combination
        for gcm_idx in range(len(summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios)):
            if len(summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios[gcm_idx][scenario_idx]) > i:
                gcm_event_data = summary_of_frequency_for_all_compound_events_considering_all_GCMs_and_scenarios[gcm_idx][scenario_idx][i]
                if gcm_event_data is not None:
                    gcm_data.append(gcm_event_data)
        
        if gcm_data:
            average_frequency_of_compound_events_across_gcms = xr.concat(gcm_data, dim="gcm").mean(dim="gcm", skipna=True)
            averaged_scenario.append(average_frequency_of_compound_events_across_gcms)
        else:
            averaged_scenario.append(None)
    
    average_frequency_of_all_compound_events.append(averaged_scenario)

# Generate names of the compound events (in the same order as the calculations above)
compound_events_names = list(itertools.combinations(extreme_event_categories, 2))

# Plot the figure for the average frequency of all compound events
selected_indices = [9, 11, 5, 1]  # for the heatwaves and wildfires, heatwaves and crop failures, droughts and heatwaves, and river floods and heatwaves pairs
plot_of_average_frequency_of_all_compound_events = fn.plot_average_frequency_of_compound_events_considering_all_gcms_and_showing_all_compound_events(average_frequency_of_all_compound_events, compound_events_names, selected_indices)


second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_of_average_frequency_of_all_compound_events_for_second_selected_events = fn.plot_average_frequency_of_compound_events_considering_all_gcms_and_showing_all_compound_events(average_frequency_of_all_compound_events, compound_events_names, second_selected_indices)




#%% MOST DOMINANT COMPOUND EVENT
# Plot the most dominant compound event per scenario
plot_dominant_compound_event_per_scenario=fn.plot_dominant_compound_event_per_scenario(average_frequency_of_all_compound_events, compound_events_names)


#%% PROBABILITY RATIO

# Initialize lists to store PRs for individual and compound events
probability_ratios_individual_event_1 = []
probability_ratios_individual_event_2 = []
probability_ratios_compound_events = []

for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]  # All GCM data on occurrence of extreme events
    
    pr_for_all_possible_combinations_and_GCM_per_scenario_event_1 = []
    pr_for_all_possible_combinations_and_GCM_per_scenario_event_2 = []
    pr_for_all_possible_combinations_and_GCM_per_scenario_compound = []
    
    for scenario in range(1, len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):  # Start from scenario 1 (skip early industrial)
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        occurrence_of_extreme_events_in_early_industrial = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[0]  # Early industrial
        
        pr_for_event_1 = []
        pr_for_event_2 = []
        pr_for_compound_events = []
        
        # Generate pairs of extreme events using the indices of the categories
        event_pairs = list(itertools.combinations(range(len(extreme_event_categories)), 2))
        
        for pair in event_pairs:
            index_1, index_2 = pair  # These are now indices, not event names
            
            event_1_name = extreme_event_categories[index_1]
            event_2_name = extreme_event_categories[index_2]
            
            if scenario == 4 and ('cropfailedarea' in [event_1_name, event_2_name]):
                print("Empty map created for cropfailedarea in rcp8.5 scenario")
                # Insert an empty map for pairs involving cropfailedarea in RCP8.5
                empty_map = xr.full_like(occurrence_of_extreme_events_considering_one_scenario[0][0][0], fill_value=np.nan)
                pr_for_event_1.append(empty_map)
                pr_for_event_2.append(empty_map)
                pr_for_compound_events.append(empty_map)
                continue
            
            models_event_1 = occurrence_of_extreme_events_considering_one_scenario[index_1]
            models_event_2 = occurrence_of_extreme_events_considering_one_scenario[index_2]
            
            models_event_1_early = occurrence_of_extreme_events_in_early_industrial[index_1]
            models_event_2_early = occurrence_of_extreme_events_in_early_industrial[index_2]
            
            pr_across_impact_models_event_1 = None
            pr_across_impact_models_event_2 = None
            pr_across_impact_models_compound = None
            
            # Pairing model_1 with model_1_early, and model_2 with model_2_early
            paired_models_event_1 = zip(models_event_1, models_event_1_early)
            paired_models_event_2 = zip(models_event_2, models_event_2_early)
            
            for (model_1, model_1_early), (model_2, model_2_early) in itertools.product(paired_models_event_1, paired_models_event_2):
                # Calculate the occurrence of individual and compound events
                occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                occurrence_of_compound_events_early = fn.compound_event_occurrence(model_1_early, model_2_early)
                
                frequency_event_1 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1) / 50
                frequency_event_1_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1_early) / 50
                
                frequency_event_2 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2) / 50
                frequency_event_2_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2_early) / 50
                
                frequency_of_compound_events = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events) / 50
                frequency_of_compound_events_early = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events_early) / 50
                
                # Calculate the PRs
                pr_event_1 = frequency_event_1 / frequency_event_1_early
                pr_event_2 = frequency_event_2 / frequency_event_2_early
                pr_compound = ((frequency_of_compound_events / (frequency_event_1 * frequency_event_2)) /(frequency_of_compound_events_early / (frequency_event_1_early * frequency_event_2_early)))
                
                # Handle inf values in the PR calculations
                pr_event_1 = xr.where(np.isfinite(pr_event_1), pr_event_1, np.nan)
                pr_event_2 = xr.where(np.isfinite(pr_event_2), pr_event_2, np.nan)
                pr_compound = xr.where(np.isfinite(pr_compound), pr_compound, np.nan)
                
                # Store the PRs for individual events and compound event
                if pr_across_impact_models_event_1 is None:
                    pr_across_impact_models_event_1 = pr_event_1
                else:
                    pr_across_impact_models_event_1 = xr.concat(
                        [pr_across_impact_models_event_1, pr_event_1],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
                
                if pr_across_impact_models_event_2 is None:
                    pr_across_impact_models_event_2 = pr_event_2
                else:
                    pr_across_impact_models_event_2 = xr.concat(
                        [pr_across_impact_models_event_2, pr_event_2],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
                
                if pr_across_impact_models_compound is None:
                    pr_across_impact_models_compound = pr_compound
                else:
                    pr_across_impact_models_compound = xr.concat(
                        [pr_across_impact_models_compound, pr_compound],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
            
            pr_for_event_1.append(pr_across_impact_models_event_1)
            pr_for_event_2.append(pr_across_impact_models_event_2)
            pr_for_compound_events.append(pr_across_impact_models_compound)
        
        pr_for_all_possible_combinations_and_GCM_per_scenario_event_1.append(pr_for_event_1)
        pr_for_all_possible_combinations_and_GCM_per_scenario_event_2.append(pr_for_event_2)
        pr_for_all_possible_combinations_and_GCM_per_scenario_compound.append(pr_for_compound_events)
    
    probability_ratios_individual_event_1.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_1)
    probability_ratios_individual_event_2.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_2)
    probability_ratios_compound_events.append(pr_for_all_possible_combinations_and_GCM_per_scenario_compound)



# Calculate the averages for each scenario across all GCMs
average_pr_for_event_1 = []
average_pr_for_event_2 = []
average_pr_for_compound_events = []

for scenario in range(4):  # We have 4 scenarios excluding early industrial
    averaged_scenario_event_1 = []
    averaged_scenario_event_2 = []
    averaged_scenario_compound = []
    
    for i in range(15):  # 15 possible combinations of the 6 extreme events
        # Gather data across GCMs for the current scenario and event combination
        gcm_data_event_1 = []
        gcm_data_event_2 = []
        gcm_data_compound = []
        
        for gcm in probability_ratios_individual_event_1:
            data = gcm[scenario][i]
            if data is not None:
                gcm_data_event_1.append(data)
        
        for gcm in probability_ratios_individual_event_2:
            data = gcm[scenario][i]
            if data is not None:
                gcm_data_event_2.append(data)
        
        for gcm in probability_ratios_compound_events:
            data = gcm[scenario][i]
            if data is not None:
                gcm_data_compound.append(data)
        
        # Ensure there is data to average across
        if gcm_data_event_1:
            if all(data.shape == gcm_data_event_1[0].shape for data in gcm_data_event_1):
                average_pr_across_gcms_event_1 = xr.concat(gcm_data_event_1, dim="gcm").mean(dim="gcm", skipna=True)
            else:
                print(f"Shape mismatch found in gcm_data_event_1 for scenario {scenario}, event combination {i}")
                average_pr_across_gcms_event_1 = None
            averaged_scenario_event_1.append(average_pr_across_gcms_event_1)
        else:
            averaged_scenario_event_1.append(None)
        
        if gcm_data_event_2:
            if all(data.shape == gcm_data_event_2[0].shape for data in gcm_data_event_2):
                average_pr_across_gcms_event_2 = xr.concat(gcm_data_event_2, dim="gcm").mean(dim="gcm", skipna=True)
            else:
                print(f"Shape mismatch found in gcm_data_event_2 for scenario {scenario}, event combination {i}")
                average_pr_across_gcms_event_2 = None
            averaged_scenario_event_2.append(average_pr_across_gcms_event_2)
        
        if gcm_data_compound:
            if all(data.shape == gcm_data_compound[0].shape for data in gcm_data_compound):
                average_pr_across_gcms_compound = xr.concat(gcm_data_compound, dim="gcm").mean(dim="gcm", skipna=True)
            else:
                print(f"Shape mismatch found in gcm_data_compound for scenario {scenario}, event combination {i}")
                average_pr_across_gcms_compound = None
            averaged_scenario_compound.append(average_pr_across_gcms_compound)
        else:
            averaged_scenario_compound.append(None)
    
    average_pr_for_event_1.append(averaged_scenario_event_1)
    average_pr_for_event_2.append(averaged_scenario_event_2)
    average_pr_for_compound_events.append(averaged_scenario_compound)



# Generate names of the compound events (in the same order as the calculations above)
compound_events_names = list(itertools.combinations(extreme_event_categories, 2))

selected_indices = [9, 11, 5, 1]  # Indices for selected compound events

# Function `plot_probability_ratios` that can handle three columns
plot_probability_ratios_for_selected_indices_main = fn.plot_probability_ratios(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    selected_indices
)



second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_probability_ratios_for_second_selected_events = fn.plot_probability_ratios_for_second_selected_events(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    second_selected_indices
)





#%% SECOND OPTION TO PLOT PROBABILITY RATIOS WITH HATCHING
# Initialize lists to store PRs and inf locations for compound events
probability_ratios_individual_event_1 = []
probability_ratios_individual_event_2 = []
probability_ratios_compound_events = []

inf_locations_compound_events = []

for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]  # All GCM data on occurrence of extreme events
    
    pr_for_all_possible_combinations_and_GCM_per_scenario_event_1 = []
    pr_for_all_possible_combinations_and_GCM_per_scenario_event_2 = []
    pr_for_all_possible_combinations_and_GCM_per_scenario_compound = []
    
    inf_for_all_possible_combinations_and_GCM_per_scenario_compound = []
    
    for scenario in range(1, len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):  # Start from scenario 1 (skip early industrial)
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        occurrence_of_extreme_events_in_early_industrial = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[0]  # Early industrial
        
        pr_for_event_1 = []
        pr_for_event_2 = []
        pr_for_compound_events = []
        
        inf_for_compound_events = []
        
        # Generate pairs of extreme events using the indices of the categories
        event_pairs = list(itertools.combinations(range(len(extreme_event_categories)), 2))
        
        for pair in event_pairs:
            index_1, index_2 = pair  # These are now indices, not event names
            
            event_1_name = extreme_event_categories[index_1]
            event_2_name = extreme_event_categories[index_2]
            
            if scenario == 4 and ('cropfailedarea' in [event_1_name, event_2_name]):
                print("Empty map created for cropfailedarea in rcp8.5 scenario")
                # Insert an empty map for pairs involving cropfailedarea in RCP8.5
                empty_map = xr.full_like(occurrence_of_extreme_events_considering_one_scenario[0][0][0], fill_value=np.nan)
                pr_for_event_1.append(empty_map)
                pr_for_event_2.append(empty_map)
                pr_for_compound_events.append(empty_map)
                
                inf_for_compound_events.append(np.zeros_like(empty_map, dtype=bool))
                continue
            
            models_event_1 = occurrence_of_extreme_events_considering_one_scenario[index_1]
            models_event_2 = occurrence_of_extreme_events_considering_one_scenario[index_2]
            
            models_event_1_early = occurrence_of_extreme_events_in_early_industrial[index_1]
            models_event_2_early = occurrence_of_extreme_events_in_early_industrial[index_2]
            
            pr_across_impact_models_event_1 = None
            pr_across_impact_models_event_2 = None
            pr_across_impact_models_compound = None
            
            inf_across_impact_models_compound = None
            
            # Pairing model_1 with model_1_early, and model_2 with model_2_early
            paired_models_event_1 = zip(models_event_1, models_event_1_early)
            paired_models_event_2 = zip(models_event_2, models_event_2_early)
            
            for (model_1, model_1_early), (model_2, model_2_early) in itertools.product(paired_models_event_1, paired_models_event_2):
                # Calculate the occurrence of individual and compound events
                occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                occurrence_of_compound_events_early = fn.compound_event_occurrence(model_1_early, model_2_early)
                
                frequency_event_1 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1) / 50
                frequency_event_1_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1_early) / 50
                
                frequency_event_2 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2) / 50
                frequency_event_2_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2_early) / 50
                
                frequency_of_compound_events = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events) / 50
                frequency_of_compound_events_early = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events_early) / 50
                
                # Calculate the PRs
                pr_event_1 = frequency_event_1 / frequency_event_1_early
                pr_event_2 = frequency_event_2 / frequency_event_2_early
                pr_compound = ((frequency_of_compound_events / (frequency_event_1 * frequency_event_2)) /
                              (frequency_of_compound_events_early / (frequency_event_1_early * frequency_event_2_early)))
                
                # Store locations where frequency_of_compound_events_early == 0 
                # while the frequency of event 1 and 2 early is not zero
                inf_mask_compound = (frequency_of_compound_events_early == 0) & (frequency_event_1_early > 0) & (frequency_event_2_early > 0)
                
                inf_for_compound_events.append(inf_mask_compound)
                
                # Handle inf values in the PR calculations
                pr_event_1 = xr.where(np.isfinite(pr_event_1), pr_event_1, np.nan)
                pr_event_2 = xr.where(np.isfinite(pr_event_2), pr_event_2, np.nan)
                pr_compound = xr.where(np.isfinite(pr_compound), pr_compound, np.nan)
                
                # Store the PRs for individual events and compound event
                if pr_across_impact_models_event_1 is None:
                    pr_across_impact_models_event_1 = pr_event_1
                    
                else:
                    pr_across_impact_models_event_1 = xr.concat(
                        [pr_across_impact_models_event_1, pr_event_1],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
                    
                
                if pr_across_impact_models_event_2 is None:
                    pr_across_impact_models_event_2 = pr_event_2
                    
                else:
                    pr_across_impact_models_event_2 = xr.concat(
                        [pr_across_impact_models_event_2, pr_event_2],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
                
                if pr_across_impact_models_compound is None:
                    pr_across_impact_models_compound = pr_compound
                    inf_across_impact_models_compound = inf_mask_compound
                else:
                    pr_across_impact_models_compound = xr.concat(
                        [pr_across_impact_models_compound, pr_compound],
                        dim='impact_models').mean(dim='impact_models', skipna=True)
                    # Retain the inf location from previous models
                    inf_across_impact_models_compound = np.logical_or(inf_across_impact_models_compound, inf_mask_compound)
            
            
            pr_for_event_1.append(pr_across_impact_models_event_1)
            pr_for_event_2.append(pr_across_impact_models_event_2)
            pr_for_compound_events.append(pr_across_impact_models_compound)
            
            inf_for_compound_events.append(inf_across_impact_models_compound)
        
        pr_for_all_possible_combinations_and_GCM_per_scenario_event_1.append(pr_for_event_1)
        pr_for_all_possible_combinations_and_GCM_per_scenario_event_2.append(pr_for_event_2)
        pr_for_all_possible_combinations_and_GCM_per_scenario_compound.append(pr_for_compound_events)
        
        inf_for_all_possible_combinations_and_GCM_per_scenario_compound.append(inf_for_compound_events)
    
    probability_ratios_individual_event_1.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_1)
    probability_ratios_individual_event_2.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_2)
    probability_ratios_compound_events.append(pr_for_all_possible_combinations_and_GCM_per_scenario_compound)
    
    inf_locations_compound_events.append(inf_for_all_possible_combinations_and_GCM_per_scenario_compound)



# Now, `inf_locations_individual_event_1`, `inf_locations_individual_event_2`, and `inf_locations_compound_events`
# contain the locations of inf values in the same structure as the probability ratios.


# Initialize lists to store averages across all GCMs
average_pr_for_event_1 = []
average_pr_for_event_2 = []
average_pr_for_compound_events = []

average_inf_for_compound_events = []

for scenario in range(4):  # We have 4 scenarios excluding early industrial
    averaged_scenario_event_1 = []
    averaged_scenario_event_2 = []
    averaged_scenario_compound = []
    
    averaged_inf_scenario_compound = []
    
    for i in range(15):  # 15 possible combinations of the 6 extreme events
        # Gather data across GCMs for the current scenario and event combination
        gcm_data_event_1 = []
        gcm_data_event_2 = []
        gcm_data_compound = []
        
        inf_data_compound = []
        
        for gcm_idx in range(len(probability_ratios_individual_event_1)):
            data_event_1 = probability_ratios_individual_event_1[gcm_idx][scenario][i]
            
            if data_event_1 is not None:
                gcm_data_event_1.append(data_event_1)
        
        for gcm_idx in range(len(probability_ratios_individual_event_2)):
            data_event_2 = probability_ratios_individual_event_2[gcm_idx][scenario][i]
            
            if data_event_2 is not None:
                gcm_data_event_2.append(data_event_2)
        
        for gcm_idx in range(len(probability_ratios_compound_events)):
            
            data_compound = probability_ratios_compound_events[gcm_idx][scenario][i]
            inf_event_compound = inf_locations_compound_events[gcm_idx][scenario][i]
            
            if data_compound is not None:
                gcm_data_compound.append(data_compound)
                inf_data_compound.append(inf_event_compound)  # Correct location for appending inf_data_compound
        
        # Ensure there is data to average across
        if gcm_data_event_1:
            if all(data.shape == gcm_data_event_1[0].shape for data in gcm_data_event_1):
                average_pr_across_gcms_event_1 = xr.concat(gcm_data_event_1, dim="gcm").mean(dim="gcm", skipna=True)
                averaged_scenario_event_1.append(average_pr_across_gcms_event_1)
            else:
                print(f"Shape mismatch found in gcm_data_event_1 for scenario {scenario}, event combination {i}")
                averaged_scenario_event_1.append(None)
        
        if gcm_data_event_2:
            if all(data.shape == gcm_data_event_2[0].shape for data in gcm_data_event_2):
                average_pr_across_gcms_event_2 = xr.concat(gcm_data_event_2, dim="gcm").mean(dim="gcm", skipna=True)
                averaged_scenario_event_2.append(average_pr_across_gcms_event_2)
            else:
                print(f"Shape mismatch found in gcm_data_event_2 for scenario {scenario}, event combination {i}")
                averaged_scenario_event_2.append(None)
        
        if gcm_data_compound:
            if all(data.shape == gcm_data_compound[0].shape for data in gcm_data_compound):
                average_pr_across_gcms_compound = xr.concat(gcm_data_compound, dim="gcm").mean(dim="gcm", skipna=True)
                averaged_scenario_compound.append(average_pr_across_gcms_compound)
                
                # Calculate the most common inf mask for this scenario and event combination
                stacked_inf_data = np.stack(inf_data_compound, axis=0)
                #any_true_inf_compound = np.any(stacked_inf_data, axis=0)
                
                most_common_inf_compound = mode(stacked_inf_data, axis=0)[0].squeeze()
                
                # Count the number of True values across the models (axis=0)
                #agreement_count = np.sum(stacked_inf_data, axis=0)
                
                # Identify grid points where at least 3 out of 4 models agree
                #agreed_inf_compound = (agreement_count >= 3)
                                
                
                averaged_inf_scenario_compound.append(most_common_inf_compound)
            else:
                print(f"Shape mismatch found in gcm_data_compound for scenario {scenario}, event combination {i}")
                averaged_scenario_compound.append(None)
                averaged_inf_scenario_compound.append(None)
    
    average_pr_for_event_1.append(averaged_scenario_event_1)
    average_pr_for_event_2.append(averaged_scenario_event_2)
    average_pr_for_compound_events.append(averaged_scenario_compound)
    
    average_inf_for_compound_events.append(averaged_inf_scenario_compound)


# Generate names of the compound events (in the same order as the calculations above)
compound_events_names = list(itertools.combinations(extreme_event_categories, 2))
 
selected_indices = [9, 11, 5, 1]  # Indices for selected compound events
 
# Function `plot_probability_ratios` that can handle three columns
plot_probability_ratios_for_selected_indices_main = fn.plot_probability_ratios_with_hatches(
         average_pr_for_event_1, 
         average_pr_for_event_2, 
         average_pr_for_compound_events, 
         compound_events_names, 
         selected_indices,
         average_inf_for_compound_events)

second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_probability_ratios_for_second_selected_events = fn.plot_probability_ratios_second_selection_with_hatches(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    second_selected_indices,
    average_inf_for_compound_events)




#%% NESTED (FOR) LOOP FOR PLOTTING BOX PLOT COMPARISON OF OCCURRENCE PER EXTREME EVENT PAIR and warming scenarios

# All 15 combinations of compound events and GCMS data on timeseries (50-year periods) of joint occurrence of compound events for all scenarios
all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events = []
# Probability of occurrence of all the 15 pairs of compound events 
all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models = []

for compound_event in compound_events:
    extreme_event_1 = compound_event[0]
    extreme_event_2 = compound_event[1]
        
    # Renaming of extreme events for plots/graphs
    extreme_event_1_name = fn.event_name(extreme_event_1)
    extreme_event_2_name = fn.event_name(extreme_event_2)
    
    
    # All GCMS data on timeseries (50-year periods) of joint occurrence of compound events for all scenarios
    all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events = []
    
    
    for gcm in gcms:
                
        
        #Considering all the impact models (with the same driving GCM)
        extreme_event_1_impact_models = fn.impact_model(extreme_event_1, gcm)
        extreme_event_2_impact_models = fn.impact_model(extreme_event_2, gcm)
        
              
        
        # timeseries (entire time periods) of joint occurrence of compound events for all scenarios
        timeseries_of_joint_occurrence_of_compound_events = []
        
        #timeseries (50-year periods) of joint occurrence of compound events for all scenarios
        timeseries_50_years_of_joint_occurrence_of_compound_events = []
        
        # Full set of timeseries (considering a pair/two extreme events) for all impact models friven by the same GCM
        gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios   
        
        
        for scenario in scenarios_of_datasets:
            
            extreme_event_1_dataset = fn.extreme_event_occurrence(extreme_event_1, extreme_event_1_impact_models, scenario)
            extreme_event_2_dataset = fn.extreme_event_occurrence(extreme_event_2, extreme_event_2_impact_models, scenario)
            
            
            if scenario == 'historical' :
                
                # EARLY INDUSTRIAL/ HISTORICAL / 50 YEARS / FROM 1861 UNTIL 1910
                
                all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910 = [] # List with total no. of years with compound events accross the multiple impact models driven by the same GCM
                all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005 = []
                
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910 = [] # List with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM               
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005 = [] 
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005 = []
                
                
                for cross_category_impact_model_pair in itertools.product(extreme_event_1_dataset[0], extreme_event_2_dataset[0]):    # Iteration function to achieve comparison of one impact model of extreme event 1 with another impact model of extreme event 2, whereby both impact models are driven by the same GCM
                    
                    extreme_event_1_from_1861_until_1910_unmasked =  cross_category_impact_model_pair[0][0:50] # (UNMASKED) occurrence of event 1 considering one impact model 
                    extreme_event_2_from_1861_until_1910_unmasked =  cross_category_impact_model_pair[1][0:50] # (UNMASKED) occurrence of event 2 considering one impact model  

                    
                    extreme_event_1_from_1861_until_1910 = xr.where(np.isnan(mask_for_historical_data[0:50]), np.nan, extreme_event_1_from_1861_until_1910_unmasked) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_1861_until_1910 = xr.where(np.isnan(mask_for_historical_data[0:50]), np.nan, extreme_event_2_from_1861_until_1910_unmasked) # (MASKED) occurrence of events considering one impact model
                    
                    # full dataset from 1861 until 2005... to create timeseries data
                    extreme_event_1_from_1861_until_2005 = xr.where(np.isnan(mask_for_historical_data), np.nan, cross_category_impact_model_pair[0]) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_1861_until_2005 = xr.where(np.isnan(mask_for_historical_data), np.nan, cross_category_impact_model_pair[1]) # (MASKED) occurrence of events considering one impact model
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model during the period '+ time_periods_of_datasets[0] + '\n')
                    else:
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 1861 UNTIL 1910
                        occurrence_of_compound_events_from_1861_until_1910 = fn.compound_event_occurrence(extreme_event_1_from_1861_until_1910, extreme_event_2_from_1861_until_1910) #returns True for locations with occurence of compound events within same location in same year
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 1861 UNTIL 1910 (Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_1861_until_1910 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_1861_until_1910)
                        all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910.append(no_of_years_with_compound_events_from_1861_until_1910) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        

                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910) 
                        
                
 
                
                    # PRESENT DAY/ HISTORICAL / 50 YEARS / FROM 1956 UNTIL 2005
                    
                    extreme_event_1_from_1956_until_2005_unmasked =  cross_category_impact_model_pair[0][95:] # (UNMASKED) occurrence of event 1 considering ensemble of gcms
                    extreme_event_2_from_1956_until_2005_unmasked =  cross_category_impact_model_pair[1][95:] # (UNMASKED) occurrence of event 2 considering ensemble of gcms
                    
                    extreme_event_1_from_1956_until_2005 = xr.where(np.isnan(mask_for_historical_data[95:]), np.nan, extreme_event_1_from_1956_until_2005_unmasked) # (MASKED) occurrence of events considering ensemble of gcms
                    extreme_event_2_from_1956_until_2005 = xr.where(np.isnan(mask_for_historical_data[95:]), np.nan, extreme_event_2_from_1956_until_2005_unmasked) # (MASKED) occurrence of events considering ensemble of gcms
                        
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[1] + '\n')
                    else:
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 1956 UNTIL 2005
                        occurrence_of_compound_events_from_1956_until_2005 = fn.compound_event_occurrence(extreme_event_1_from_1956_until_2005, extreme_event_2_from_1956_until_2005) #returns True for locations with occurence of compound events within same location in same year
                        
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 (Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_1956_until_2005 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_1956_until_2005)
                        all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005.append(no_of_years_with_compound_events_from_1956_until_2005) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005)                     
                           
                
                
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[0] + '\n')
                else:   
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910)
                    
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[1] + '\n')
                else:
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005)
                

                
            else:
                
                # Note: cropfailure events have no data for the rcp85 scenario: thus will be igonored in the rcp 85 scenario
                
                # END OF CENTURY / 50 YEARS / 2050 UNTIL 2099
                
                all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099 = [] # List with total no. of years with compound events accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099 = [] # List with 95th quantile of length of spell with compound events accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = [] # List with maximum no. of years with consecutive compound events accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099 = [] # List with timeseries of occurrence of compound events accross the multiple impact models driven by the same GCM
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099 = []
                
                gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events_from_2050_until_2099 = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of compound events accross the multiple impact models
                
                end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
                
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099 = []
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099 = []
                
                
                for cross_category_impact_model_pair in itertools.product(extreme_event_1_dataset[1], extreme_event_2_dataset[1]):  # Iteration function to achieve comparison of one impact model of extreme event 1 with another impact model of extreme event 2, whereby both impact models are driven by the same GCM
                    
                    extreme_event_1_from_2050_until_2099_unmasked =  cross_category_impact_model_pair[0][44:] # (UNMASKED) occurrence of event 1 considering one impact model 
                    extreme_event_2_from_2050_until_2099_unmasked =  cross_category_impact_model_pair[1][44:] # (UNMASKED) occurrence of event 2 considering one impact model
                    
                    extreme_event_1_from_2050_until_2099 = xr.where(np.isnan(mask_for_projected_data[44:]), np.nan, extreme_event_1_from_2050_until_2099_unmasked) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_2050_until_2099 = xr.where(np.isnan(mask_for_projected_data[44:]), np.nan, extreme_event_2_from_2050_until_2099_unmasked) # (MASKED) occurrence of events considering one impact model
                    
                    # full dataset from 1861 until 2005... to create timeseries data
                    extreme_event_1_from_2006_until_2099 = xr.where(np.isnan(mask_for_projected_data), np.nan, cross_category_impact_model_pair[0]) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_2006_until_2099 = xr.where(np.isnan(mask_for_projected_data), np.nan, cross_category_impact_model_pair[1]) # (MASKED) occurrence of events considering one impact model
                   
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                    else: 
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 2050 UNTIL 2099
                        occurrence_of_compound_events_from_2050_until_2099 = fn.compound_event_occurrence(extreme_event_1_from_2050_until_2099, extreme_event_2_from_2050_until_2099) #returns True for locations with occurence of compound events within same location in same year
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099(Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_2050_until_2099 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_2050_until_2099)                        
                        all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099.append(no_of_years_with_compound_events_from_2050_until_2099) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        
                        # TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099)
                        
                        
                # AREA AFFECTED BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                else:
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099)
                    
                                   
        # COMPARISON OF ALL THE SCENARIOS PER PAIR OF EXTREME EVENTS
        
        # Append all 4 GCMS (50-year) timeseries of compound events
        all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append(timeseries_50_years_of_joint_occurrence_of_compound_events)
        

    
    # BOX PLOT COMPARISON OF OCCURRENCE PER EXTREME EVENT PAIR
    
    box_plot_per_compound_event = fn.boxplot_comparing_gcms(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, gcms)
     
    all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append([all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name])
            
                
# COMPARISON PLOT FOR ALL THE 15 BOX PLOTS   
           
all_box_plots = fn.all_boxplots(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, gcms)                
               
#considering all impact models and all their driving GCMs per extreme event  
new_box = fn.comparison_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)

new_box_with_median_values_per_boxplot = fn.comparison_boxplot_with_median_values_per_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)


# print total runtime of the code
end_time=datetime.now()
print('Processing duration: {}'.format(end_time - start_time))

print('*******************DONE************************ ')

