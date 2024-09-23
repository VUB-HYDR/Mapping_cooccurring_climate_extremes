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
#probability_of_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] 
#length_of_spell_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] 
#maximum_number_of_years_with_consecutive_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] 
#timeseries_of_fraction_of_area_affected_by_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] 

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

average_no_of_years_with_occurrence_of_all_extreme_events_considering_all_gcms_per_scenario = []

for extreme_event_occurrence in range(len(no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models)): # one event's data out of the six categories of extremes
    
    # Event name
    extreme_event_category = no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][5]
    
    # Average number of occurrences per grid point considering different impact models driven by the GCM 
    
    average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_per_scenario = []
    
    
    for scenario in range(len(no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence]) - 1): # -1 is to avoid the last member of the list which is the extreme event name (String)
        
        if extreme_event_category == 'Crop Failures' and scenario == 4:
            print('No data available on occurrence of crop failures for selected impact model during the period '+ time_periods_of_datasets[2] + ' under RCP 8.5 \n')
        else:

            average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms =[]
            
            for gcm in no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][scenario]:
                
                average_no_of_years_with_occurrence_of_extreme_event_considering_all_impact_models_driven_by_the_same_gcm = xr.concat(gcm[0], dim='impact_models').mean(dim='impact_models', skipna= True) # gcm[0] because it is a list with one element (that is a list of arrays)
                average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms.append(average_no_of_years_with_occurrence_of_extreme_event_considering_all_impact_models_driven_by_the_same_gcm)
        
            average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_per_scenario.append(average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms)
                
    average_no_of_years_with_occurrence_of_all_extreme_events_considering_all_gcms_per_scenario.append(average_no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_per_scenario)
    


# All GCM data for all the six extreme events per scenario in one list
all_gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario = [[[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]]] # In the order of scenarios: early-industrial, present day, RCP2.6, RCP6.0 and RCP8.5: AND WITHIN EACH OF THOSE: Data from the 4GCMs

for extreme_event in average_no_of_years_with_occurrence_of_all_extreme_events_considering_all_gcms_per_scenario:
    
    for scenario_of_dataset in range(len(extreme_event)):
        
        data_per_scenario = extreme_event[scenario_of_dataset]
        
        #gcm_data_for_an_extreme_event_per_scenario = []
        
        for gcm in range(len(data_per_scenario)): 
        
            #### TRYING TO GET THE SAME GCM DATA FOR ALL EXTREMES  SUCH THAT I CAN SUM THEM UP TO THE TOTAL AVERAGE
            #position_of_gcm =  scenario_of_dataset.index(gcm)
            
            #all_gcm_data[gcm].append(scenario_of_dataset[gcm])
            gcm_data = data_per_scenario[gcm]
            
            all_gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario[scenario_of_dataset][gcm].append(gcm_data)
    


# Calculating and plotting the propensity of an extreme event considering all the GCMs and Impact Models (Inorder to plot one figure of propensity per extreme event and scenario)

summary_of_propensity_of_an_extreme_event_considering_all_GCMs_and_scenarios = [[],[],[],[],[]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0 and RCP8.5

for scenario in range(len(all_gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario)): 
    
    all_gcm_data = all_gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario[scenario]
    
    propensity_of_extreme_events_considering_different_gcms_per_scenario = [[],[],[],[],[],[]] # Whereby the order of the extreme events follows the order in 'extreme_event_categories' at the beginning of this script
    
    # Average occurrence for only one of the extremes considering all GCMS and impact models
    average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_all_gcms = [[],[],[],[],[],[]] # Whereby the order of the extreme events follows the order in 'extreme_event_categories' at the beginning of this script
    
    # Average number of years with occurrence of all extreme events considering all GCMS and impact models
    average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario_and_all_gcms = []
    
    for gcm in range(len(all_gcm_data)):
        
        gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario = all_gcm_data[gcm]
        
        # Average occurrence for only one of the extremes
        for i in range(len(gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario)):
            average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_all_gcms[i].append(gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario[i])
            
        # Average number of occurrences for all extreme events
        average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario= xr.concat(gcm_data_for_occurrence_of_all_extreme_event_categories_per_scenario, dim='impact_models_data_for_all_extremes_driven_by_same_gcm').mean(dim='impact_models_data_for_all_extremes_driven_by_same_gcm', skipna= True)
        # Append list with average data from all GCMS
        average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario_and_all_gcms.append(average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario)
        
    average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario_considering_all_gcms = xr.concat(average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario_and_all_gcms, dim='gcms_data_for_all_extremes').mean(dim='gcms_data_for_all_extremes', skipna= True)
    
    for extreme_event in range(len(average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_all_gcms)):
        
        # Import extreme event order such that data for crop failures is recognized since it is missing RCP8.5 simulations
        extreme_event_order = extreme_event_categories
    
        extreme_event_data_name = extreme_event_order[extreme_event]
        
        extreme_event_name = fn.event_name(extreme_event_data_name)
        
        if extreme_event_data_name == 'cropfailedarea' and scenario == 4: # Reminder note: data on crop failures RCP8.5 simulations is missing under this dataset
            
            print('No data available on occurrence of crop failures under RCP8.5') 
        
        else:        
        
            average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario = average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_all_gcms[extreme_event]
            
            average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_considering_all_gcms = xr.concat(average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario, dim='gcms_data_for_extreme_event').mean(dim='gcms_data_for_extreme_event', skipna= True)
            
            # Propensity of an extreme event considering all GCMs (ONE-FIGURE)
            propensity_of_an_extreme_event_considering_all_GCMs= fn.propensity(average_number_of_years_with_occurrence_of_one_extreme_event_category_per_scenario_and_considering_all_gcms, average_number_of_years_with_occurrence_of_all_extreme_event_categories_per_scenario_considering_all_gcms)
            summary_of_propensity_of_an_extreme_event_considering_all_GCMs_and_scenarios[scenario].append(propensity_of_an_extreme_event_considering_all_GCMs)
            
            plot_propensity_of_an_extreme_event_considering_all_GCMs = fn.plot_propensity_considering_all_gcms_in_one_plot(propensity_of_an_extreme_event_considering_all_GCMs, extreme_event_name, scenario)
    
    
    
    
plot_summary_of_propensity_of_an_extreme_event_considering_all_GCMs_and_scenarios = fn.plot_propensity_considering_all_gcms_and_showing_all_extremes(summary_of_propensity_of_an_extreme_event_considering_all_GCMs_and_scenarios, extreme_event_categories)
        
        
    

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

quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_gcms = [[[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]] # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the six extreme event catergories 

for extreme_event_occurrence in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models)): # one event's data out of the six categories of extremes
    
    # Event name
    extreme_event_category = occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][5] # name of extreme event
    
    # Occurrences per grid point considering different impact models driven by the GCM 
    #occurrence_of_extreme_event_considering_all_gcms_per_scenario = []
    
    #average_length_of_spell_with_occurrence_of_extreme_event_considering_all_gcms_per_scenario = []
    
    
    for scenario in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence]) - 1): # -1 is to avoid the last member of the list which is the extreme event name (String)
        
        if extreme_event_category == 'Crop Failures' and scenario == 4:
            print('No data available on occurrence of crop failures for selected impact model during the period '+ time_periods_of_datasets[2] + ' under RCP 8.5 \n')
        else:
            
            #occurrence_of_extreme_event_considering_all_gcms = []
            #average_length_of_spell_with_occurrence_of_extreme_event_considering_all_gcms =[]
            
            for gcm in range(len(occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][scenario])):
                
                gcm_data_on_occurrence_of_extreme_event = occurrence_of_extreme_event_considering_all_gcms_and_impact_models[extreme_event_occurrence][scenario][gcm][0] #gcm[0] because it is a list with one element (that is a list of arrays)
                
                #average_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models_driven_by_the_same_gcm 
                quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models = []
                
                for impact_model_data in gcm_data_on_occurrence_of_extreme_event:
                   
                    # Length of spell of an extreme event (consecutive years with impacts) such that the minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
                    length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model = fn.length_of_spell_with_occurrence_of_extreme_event(impact_model_data)
                    # Remove zeros to ensure only length of spells is captured. Note: minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
                    length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_considering_only_occurrences = xr.where(length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model > 0, length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model, np.nan)
                    # 95th quantile of the length of spells
                    quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model = length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_considering_only_occurrences.quantile(0.95, dim = 'time', skipna = True)
                    quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models.append(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model)
                
                if extreme_event_occurrence <= 5: # To avoid the name of the extreme event in the list
                    
                    #full_dataset_occurrence_of_extreme_event_considering_all_gcms_per_scenario[gcm][scenario].insert(extreme_event_occurrence, gcm_data_on_occurrence_of_extreme_event)
                
                    full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm][scenario].append(gcm_data_on_occurrence_of_extreme_event) 
                    
                    
                        
                    
                #average_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models_driven_by_the_same_gcm 
                average_quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models = xr.concat(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models, dim = 'impact_models_data_for_all_extremes_driven_by_same_gcm').mean(dim='impact_models_data_for_all_extremes_driven_by_same_gcm', skipna = True)
                # Append to full list of length of spells for all extreme events considering all impact models and gcms
                quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_gcms[gcm][scenario].append(average_quantile_95th_of_length_of_spell_with_occurrence_of_extreme_event_considering_all_impact_models)
                              



# CO-OCCURRENCE RATIO FOR EXTREME EVENTS considering all scenarios mapped on a single plot

# Calculating co-occurrence ratio
summary_of_average_cooccurrence_ratio_considering_all_GCMs_and_scenarios = [[],[],[],[]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0

total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios = [[],[],[],[]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0
total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios = [[],[],[],[]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0

for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm] # All GCM data on occurrence of all extreme eventws==s
        
    for scenario in range(len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        
        if scenario == 4:
            
            print('No data available on occurrence of crop failures for selected impact model during the period '+ time_periods_of_datasets[2] + ' under RCP 8.5 \n')
        
        else: 
            
            cooccurrence_ratio_considering_cross_category_impact_models_for_all_extreme_events_per_scenario = []
            
            total_number_of_extreme_events_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario = []
            total_number_of_extreme_events_not_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario =[]
            
            # Iteration function to achieve comparison between the different impact models driven by the same GCM and for the different extreme events
            for cross_category_impact_models_for_all_extreme_events in itertools.product(*occurrence_of_extreme_events_considering_one_scenario): # the asterisk * unpacks all the sublits under 'occurrence_of_extreme_events_considering_one_scenario' here being impact models of occurences for each extreme event under same scenario and GCM
                
                #extreme_event_not_occurring = 0 # Recall zero (0) in a grid represents an extreme event not occurring that year within the grid cell
                #extreme_event_occurring = 1 # Recall one (1) in a grid represents an extreme event occurring that year within the grid cell
                
                # Total number of extreme events occurring in a grid cell per year
                total_number_of_extreme_events_occurring_per_year = xr.concat(cross_category_impact_models_for_all_extreme_events, dim='impact_models').sum(dim='impact_models',skipna=True)
                
                # Total number of extreme events occurring in isolation. Thus were only one extreme event per grid cell per year occurs
                extreme_events_occurring_in_isolation = xr.where(total_number_of_extreme_events_occurring_per_year == 1, 1, xr.where(np.isnan(total_number_of_extreme_events_occurring_per_year), np.nan, 0)) # Return 1 where an extreme event occurred in one grid cell in isolation per year and 0 where more than one extreme event occurred in the same year
                total_number_of_extreme_events_occurring_in_isolation = xr.concat(extreme_events_occurring_in_isolation, dim='time').sum(dim='time', skipna = True)
                #total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios[scenario].append(total_number_of_extreme_events_occurring_in_isolation)
                total_number_of_extreme_events_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario.append(total_number_of_extreme_events_occurring_in_isolation)
                
                # Total number of extreme events occurring not in isolation. Thus we consider where more than extreme event occurs in a grid cell per year
                extreme_events_not_occurring_in_isolation = xr.where(total_number_of_extreme_events_occurring_per_year > 1, total_number_of_extreme_events_occurring_per_year, xr.where(np.isnan(total_number_of_extreme_events_occurring_per_year), np.nan, 0))
                total_number_of_extreme_events_not_occurring_in_isolation = xr.concat(extreme_events_not_occurring_in_isolation, dim='time').sum(dim='time', skipna = True)                                
                #total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios[scenario].append(total_number_of_extreme_events_not_occurring_in_isolation)
                total_number_of_extreme_events_not_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario.append(total_number_of_extreme_events_not_occurring_in_isolation)
             
            
            average_total_number_of_extreme_events_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario = xr.concat(total_number_of_extreme_events_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario, dim ='impact_models').mean(dim='impact_models', skipna = True)
            total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios[scenario].append(average_total_number_of_extreme_events_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario)

            average_total_number_of_extreme_events_not_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario = xr.concat(total_number_of_extreme_events_not_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario, dim ='impact_models').mean(dim='impact_models', skipna = True)
            total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios[scenario].append(average_total_number_of_extreme_events_not_occurring_in_isolation_considering_cross_category_impact_models_under_same_scenario)
            
average_total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios = []
for scenario in total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios:
    average_total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs = xr.concat(scenario, dim='all_gcms').mean(dim='all_gcms', skipna=True)
    average_total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios.append(average_total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs)

average_total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios = []
for scenario in total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios:
    average_total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs = xr.concat(scenario, dim='all_gcms').mean(dim='all_gcms', skipna=True)
    average_total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios.append(average_total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs)

    
for i in range(4): # Calculating average co-occurrence ratio for all the scenarios
    average_cooccurrence_ratio_considering_all_GCMs_and_scenarios = fn.cooccurrence_ratio(average_total_number_of_extreme_events_not_occurring_in_isolation_considering_all_GCMs_and_scenarios[i], average_total_number_of_extreme_events_occurring_in_isolation_considering_all_GCMs_and_scenarios[i])
    summary_of_average_cooccurrence_ratio_considering_all_GCMs_and_scenarios[i].append(average_cooccurrence_ratio_considering_all_GCMs_and_scenarios)

plot_cooccurrence_ratio_considering_all_gcms_in_a_single_plot = fn.plot_cooccurrence_ratio_considering_all_gcms_in_a_single_plot(summary_of_average_cooccurrence_ratio_considering_all_GCMs_and_scenarios, mask_for_historical_data[0])           




#%%  LENGTH OF SPELLS -- AGGREGATED ACROSS MODELS -- CONSIDERING ALL IMPACT MODELS AND THEIR DRIVING GCMs
summary_of_length_of_spells_for_all_extreme_events_considering_all_GCMs_and_scenarios = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0 and RCP8.5

for gcm in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_gcms)):  # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the six extreme event catergories 
    
    gcm_data_on_quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_the_scenarios = quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_gcms[gcm] # All GCM data on length of spells
    
    # 95th quantile of length of spell with occurrence of extreme events considering all GCMS and impact models
    quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm = [[],[],[],[],[],[]] # Whereby the order of the extreme events follows the order in 'extreme_event_categories' at the beginning of this script
    
    for scenario in range(len(gcm_data_on_quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_the_scenarios)):
        
        quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        
        if scenario == 4:
            
            for i in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario)):
                quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[i].append(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario[i])
            # Get the index of 'cropfailedarea'
            cropfailedarea_index = extreme_event_categories.index('cropfailedarea')
            # Generate an empty map for cropfailedarea (CF) in RCP8.5 since we have no data
            empty_map = xr.full_like(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario[0], fill_value=np.nan)
            quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[cropfailedarea_index].append(empty_map)
            print('No data available on occurrence of crop failures for selected impact model during the period '+ time_periods_of_datasets[2] + ' under RCP 8.5 \n')
        
        else: 
            
            for i in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario)):
                quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[i].append(quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_one_scenario[i])
            
    for extreme_event in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm)):
        
        for scenario in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[extreme_event])):
            
            summary_of_length_of_spells_for_all_extreme_events_considering_all_GCMs_and_scenarios[scenario][gcm].append(quantile_95th_of_length_of_spell_with_occurrence_of_one_extreme_event_category_per_scenario_and_one_gcm[extreme_event][scenario])

# Initialize the structure to store aggregated length of spells across all GCMs and scenarios
average_length_of_spells_for_all_extreme_events = []
for scenario in summary_of_length_of_spells_for_all_extreme_events_considering_all_GCMs_and_scenarios[:5]:
    
    averaged_scenario = []
    
    for i in range(6):
        gcm_data = [gcm[i] for gcm in scenario]
        
        if gcm_data:
            average_length_of_spell_for_one_extreme_event_category_accross_gcms = xr.concat(gcm_data, dim="gcm").mean(dim="gcm", skipna= True)
            averaged_scenario.append(average_length_of_spell_for_one_extreme_event_category_accross_gcms)
        else:
            averaged_scenario.append(None)
            
    
    average_length_of_spells_for_all_extreme_events.append(averaged_scenario)
        
#Plot figure of_average length of spells for all extreme events 
plot_of_average_length_of_spells_for_all_extreme_events = fn.plot_quantile_95th_of_length_of_spell_with_occurrence_considering_all_gcms_and_showing_all_extremes(average_length_of_spells_for_all_extreme_events, extreme_event_categories)
        

        
#%% LENGTH OF SPELLS FOR COMPOUND EVENT PAIRS-- AGGREGATED ACROSS MODELS -- CONSIDERING ALL IMPACT MODELS AND THEIR DRIVING GCMs
# Calculating co-occurrence ratio

### CREATING SPACE/RAM BY FREEING MEMORY OF UNUSED VARIABLES
occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] # Where order of list is the six extreme event catergories and within these: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5 and extreme event name] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']   
no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = [[],[],[],[],[],[]] # Where order of list is the six extreme event catergories and within these: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5 and extreme event name] AND within each element the order of the list of data considering the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']   
quantile_95th_of_length_of_spell_with_occurrence_of_extreme_events_considering_all_gcms = [[[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]] # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the six extreme event catergories 
no_of_years_with_occurrence_of_extreme_event_considering_all_gcms_and_impact_models = []
occurrence_of_extreme_event_considering_all_gcms_and_impact_models = []
mask_for_projected_data = []
occurrence_of_historical_floods =[]
occurrence_of_projected_floods =[]
extreme_event_from_1861_until_1910 = []
extreme_event_from_1861_until_1910_unmasked =[]
extreme_event_from_1956_until_2005 = []
extreme_event_from_1956_until_2005_unmasked = []



# Initialize the structure to store quantiles across all GCMs and scenarios
average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_cross_category_impact_models_for_all_extreme_events_for_condidering_all_scenarios_considering_all_gcms = []


# Loop through GCMs and scenarios
for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
    
    gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]  # All GCM data on occurrence of extreme events
    
    # Initialize for each GCM across all scenarios
    average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario = []  
    
    for scenario in range(len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):
        
        occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
        
        # Initialize the container for results per scenario and GCM
        average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM = []

        # Handle the missing cropfailedarea data for RCP8.5
        if scenario == 4:  # since 'cropfailedarea' has no data for RCP8.5, we need to insert an empty map where appropriate
            print('No data available for crop failures in RCP8.5; filling with empty map.')
            empty_map = np.nan * xr.zeros_like(occurrence_of_extreme_events_considering_one_scenario[0][0].mean(dim ='time'))
            
            # Generate pairs of extreme events using the names of the categories
            event_pairs = list(itertools.combinations(extreme_event_categories, 2))
            
            for pair in event_pairs:
                event_1, event_2 = pair
                
                if 'cropfailedarea' in [event_1, event_2]:
                    # Insert an empty map for pairs involving cropfailedarea
                    average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM.append(empty_map)
                else:
                    # Calculate normally for other pairs
                    index_1 = extreme_event_categories.index(event_1)
                    index_2 = extreme_event_categories.index(event_2)
                    
                    models_event_1 = occurrence_of_extreme_events_considering_one_scenario[index_1]
                    models_event_2 = occurrence_of_extreme_events_considering_one_scenario[index_2]
                    
                    average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = None
                    
                    for model_1, model_2 in itertools.product(models_event_1, models_event_2):
                        # Calculate occurrence and length of spells
                        occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                        length_of_spell_with_occurrence_of_extreme_event = fn.length_of_spell_with_occurrence_of_extreme_event(occurrence_of_compound_events)
                        length_of_spell_with_occurrence_of_extreme_event = xr.where(length_of_spell_with_occurrence_of_extreme_event > 0, length_of_spell_with_occurrence_of_extreme_event, np.nan)
                        
                        # Calculate 95th quantile
                        quantile_95th = length_of_spell_with_occurrence_of_extreme_event.quantile(0.95, dim='time', skipna=True)
                        
                        if average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models is None:
                            average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = quantile_95th
                        else:
                            average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = xr.concat(
                                [average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models, quantile_95th],
                                dim='impact_models').mean(dim='impact_models', skipna=True)
                    
                    average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM.append(average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models)
        
        else:
            # Normal processing for other scenarios
            event_pairs = list(itertools.combinations(range(len(occurrence_of_extreme_events_considering_one_scenario)), 2))
            
            for pair in event_pairs:
                event_1, event_2 = pair
                
                models_event_1 = occurrence_of_extreme_events_considering_one_scenario[event_1]
                models_event_2 = occurrence_of_extreme_events_considering_one_scenario[event_2]
                
                average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = None
                
                for model_1, model_2 in itertools.product(models_event_1, models_event_2):
                    # Calculate occurrence and length of spells
                    occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
                    length_of_spell_with_occurrence_of_extreme_event = fn.length_of_spell_with_occurrence_of_extreme_event(occurrence_of_compound_events)
                    length_of_spell_with_occurrence_of_extreme_event = xr.where(length_of_spell_with_occurrence_of_extreme_event > 0, length_of_spell_with_occurrence_of_extreme_event, np.nan)
                    
                    # Calculate 95th quantile
                    quantile_95th = length_of_spell_with_occurrence_of_extreme_event.quantile(0.95, dim='time', skipna=True)
                    
                    if average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models is None:
                        average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = quantile_95th
                    else:
                        average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models = xr.concat(
                            [average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models, quantile_95th],
                            dim='impact_models').mean(dim='impact_models', skipna=True)
                
                average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM.append(average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_accross_impact_models)
        
        average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario.append(average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM)
    
    average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_cross_category_impact_models_for_all_extreme_events_for_condidering_all_scenarios_considering_all_gcms.append(
        average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_for_all_possible_combinations_and_GCM_per_scenario
    )


summary_of_length_of_spells_with_occurrence_of_compound_events_considering_all_GCMs_and_scenarios = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]] # scenarios in order: Early industrial, Present Day, RCP2.6, RCP6.0 and RCP8.5

for gcm in range(len(average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_cross_category_impact_models_for_all_extreme_events_for_condidering_all_scenarios_considering_all_gcms)):  # Where order of list is: the gcms: ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] and within each of these gcms: [early industrial, present day, rcp 2.6, rcp6.0, rcp 8.5] whereby also within each element is the order of the list of data considering the 15 pairs of extreme event catergories (compound events)
    
    gcm_data_on_average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_all_the_scenarios = average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_cross_category_impact_models_for_all_extreme_events_for_condidering_all_scenarios_considering_all_gcms[gcm] # All GCM data on length of spells
    
    # 95th quantile of length of spell with occurrence of extreme events considering all GCMS and impact models
    quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_per_scenario_and_one_gcm = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]] # Whereby the order of the extreme events follows the order in 'extreme_event_categories' at the beginning of this script
    
    for scenario in range(len(gcm_data_on_average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_all_the_scenarios)):
        
        quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_one_scenario = gcm_data_on_average_quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_all_the_scenarios[scenario]
             
        for i in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_one_scenario)):
            quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_per_scenario_and_one_gcm[i].append(quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_considering_one_scenario[i])
            
    for extreme_event in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_per_scenario_and_one_gcm)):
        
        for scenario in range(len(quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_per_scenario_and_one_gcm[extreme_event])):
            
            summary_of_length_of_spells_with_occurrence_of_compound_events_considering_all_GCMs_and_scenarios[scenario][gcm].append(quantile_95th_of_length_of_spell_with_occurrence_of_compound_events_per_scenario_and_one_gcm[extreme_event][scenario])

# Initialize the structure to store aggregated length of spells across all GCMs and scenarios
average_length_of_spells_for_all_compound_extreme_events = []
for scenario in summary_of_length_of_spells_with_occurrence_of_compound_events_considering_all_GCMs_and_scenarios[:5]:
    
    averaged_scenario = []
    
    for i in range(15):
        gcm_data = [gcm[i] for gcm in scenario]
        
        if gcm_data:
            average_length_of_spell_for_one_pair_of_cooccuring_extreme_events_accross_gcms = xr.concat(gcm_data, dim="gcm").mean(dim="gcm", skipna= True)
# =============================================================================
#             # check
#             if average_length_of_spell_for_one_pair_of_cooccuring_extreme_events_accross_gcms.shape == (50,300,720):
#                 average_length_of_spell_for_one_pair_of_cooccuring_extreme_events_accross_gcms = average_length_of_spell_for_one_pair_of_cooccuring_extreme_events_accross_gcms.mean(dim ='time')        
# =============================================================================
            
            averaged_scenario.append(average_length_of_spell_for_one_pair_of_cooccuring_extreme_events_accross_gcms)
            
        else:
            averaged_scenario.append(None)
            
    
    average_length_of_spells_for_all_compound_extreme_events.append(averaged_scenario)
 

# Generate names of the compound events (in the same order as the calculations above)
compound_events_names = list(itertools.combinations(extreme_event_categories, 2))

#Plot figure of_average length of spells for all compound extreme events 
plot_of_average_length_of_spells_for_all_extreme_events = fn.plot_quantile_95th_of_length_of_spell_with_compound_event_occurrence_considering_all_gcms_and_showing_all_extremes(average_length_of_spells_for_all_compound_extreme_events, compound_events_names)

#Plot figure of_average length of spells for only selected compound extreme events
selected_indices = [9, 11, 5, 1] # for the heatwaves and wildfires, heatwaves and crop failures, droughts and heatwaves, and river floods and heatwaves pairs
plot_of_average_length_of_spells_for_selected_events = fn.plot_quantile_95th_of_length_of_spell_with_selected_compound_event_occurrence_considering_all_gcms_and_showing_all_extremes(average_length_of_spells_for_all_compound_extreme_events, compound_events_names, selected_indices)

second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_of_average_length_of_spells_for_second_selected_events = fn.plot_quantile_95th_of_length_of_spell_with_selected_compound_event_occurrence_considering_all_gcms_and_showing_all_extremes(average_length_of_spells_for_all_compound_extreme_events, compound_events_names, second_selected_indices)


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
# =============================================================================
#                 if data.shape[0] == 50:  # If the first dimension is 50, likely a time dimension
#                     data = data.mean(dim="time", skipna=True)  # Average over the time dimension
# =============================================================================
                gcm_data_event_1.append(data)
        
        for gcm in probability_ratios_individual_event_2:
            data = gcm[scenario][i]
            if data is not None:
# =============================================================================
#                 if data.shape[0] == 50:  # If the first dimension is 50, likely a time dimension
#                     data = data.mean(dim="time", skipna=True)
# =============================================================================
                gcm_data_event_2.append(data)
        
        for gcm in probability_ratios_compound_events:
            data = gcm[scenario][i]
            if data is not None:
# =============================================================================
#                 if data.shape[0] == 50:  # If the first dimension is 50, likely a time dimension
#                     data = data.mean(dim="time", skipna=True)
# =============================================================================
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

selected_indices =[5]
# Function `plot_probability_ratios` that can handle three columns
plot_probability_ratios_for_selected_indices_main = fn.plot_probability_ratios(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    selected_indices
)



second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
plot_probability_ratios_for_second_selected_events = fn.plot_probability_ratios_second_selection(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    second_selected_indices
)











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
plot_probability_ratios_for_second_selected_events = fn.plot_probability_ratios_second_selection(
    average_pr_for_event_1, 
    average_pr_for_event_2, 
    average_pr_for_compound_events, 
    compound_events_names, 
    second_selected_indices,
    average_inf_for_compound_events)









# =============================================================================
# 
# # Initialize lists to store PRs and inf masks for individual and compound events
# probability_ratios_individual_event_1 = []
# probability_ratios_individual_event_2 = []
# probability_ratios_compound_events = []
# 
# inf_locations_individual_event_1 = []
# inf_locations_individual_event_2 = []
# inf_locations_compound_events = []
# 
# for gcm in range(len(full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario)):
#     
#     gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios = full_dataset_occurrence_of_extreme_events_considering_all_gcms_per_scenario[gcm]  # All GCM data on occurrence of extreme events
#     
#     pr_for_all_possible_combinations_and_GCM_per_scenario_event_1 = []
#     pr_for_all_possible_combinations_and_GCM_per_scenario_event_2 = []
#     pr_for_all_possible_combinations_and_GCM_per_scenario_compound = []
#     
#     inf_for_all_possible_combinations_and_GCM_per_scenario_event_1 = []
#     inf_for_all_possible_combinations_and_GCM_per_scenario_event_2 = []
#     inf_for_all_possible_combinations_and_GCM_per_scenario_compound = []
#     
#     for scenario in range(1, len(gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios)):  # Start from scenario 1 (skip early industrial)
#         
#         occurrence_of_extreme_events_considering_one_scenario = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[scenario]
#         occurrence_of_extreme_events_in_early_industrial = gcm_data_on_occurrence_of_extreme_events_considering_all_the_scenarios[0]  # Early industrial
#         
#         pr_for_event_1 = []
#         pr_for_event_2 = []
#         pr_for_compound_events = []
#         
#         inf_for_event_1 = []
#         inf_for_event_2 = []
#         inf_for_compound_events = []
#         
#         # Generate pairs of extreme events using the indices of the categories
#         event_pairs = list(itertools.combinations(range(len(extreme_event_categories)), 2))
#         
#         for pair in event_pairs:
#             index_1, index_2 = pair  # These are now indices, not event names
#             
#             event_1_name = extreme_event_categories[index_1]
#             event_2_name = extreme_event_categories[index_2]
#             
#             if scenario == 4 and ('cropfailedarea' in [event_1_name, event_2_name]):
#                 print("Empty map created for cropfailedarea in rcp8.5 scenario")
#                 # Insert an empty map for pairs involving cropfailedarea in RCP8.5
#                 empty_map = xr.full_like(occurrence_of_extreme_events_considering_one_scenario[0][0][0], fill_value=np.nan)
#                 pr_for_event_1.append(empty_map)
#                 pr_for_event_2.append(empty_map)
#                 pr_for_compound_events.append(empty_map)
#                 continue
#             
#             models_event_1 = occurrence_of_extreme_events_considering_one_scenario[index_1]
#             models_event_2 = occurrence_of_extreme_events_considering_one_scenario[index_2]
#             
#             models_event_1_early = occurrence_of_extreme_events_in_early_industrial[index_1]
#             models_event_2_early = occurrence_of_extreme_events_in_early_industrial[index_2]
#             
#             pr_across_impact_models_event_1 = None
#             pr_across_impact_models_event_2 = None
#             pr_across_impact_models_compound = None
#             
#             inf_mask_across_impact_models_event_1 = []
#             inf_mask_across_impact_models_event_2 = []
#             inf_mask_across_impact_models_compound = []
#             
#             # Pairing model_1 with model_1_early, and model_2 with model_2_early
#             paired_models_event_1 = zip(models_event_1, models_event_1_early)
#             paired_models_event_2 = zip(models_event_2, models_event_2_early)
#             
#             for (model_1, model_1_early), (model_2, model_2_early) in itertools.product(paired_models_event_1, paired_models_event_2):
#                 # Calculate the occurrence of individual and compound events
#                 occurrence_of_compound_events = fn.compound_event_occurrence(model_1, model_2)
#                 occurrence_of_compound_events_early = fn.compound_event_occurrence(model_1_early, model_2_early)
#                 
#                 frequency_event_1 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1) / 50
#                 frequency_event_1_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_1_early) / 50
#                 
#                 frequency_event_2 = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2) / 50
#                 frequency_event_2_early = fn.total_no_of_years_with_occurrence_of_extreme_event(model_2_early) / 50
#                 
#                 frequency_of_compound_events = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events) / 50
#                 frequency_of_compound_events_early = fn.total_no_of_years_with_occurrence_of_extreme_event(occurrence_of_compound_events_early) / 50
#                 
#                 # Calculate the PRs
#                 pr_event_1 = frequency_event_1 / frequency_event_1_early
#                 pr_event_2 = frequency_event_2 / frequency_event_2_early
#                 pr_compound = ((frequency_of_compound_events / (frequency_event_1 * frequency_event_2)) /
#                               (frequency_of_compound_events_early / (frequency_event_1_early * frequency_event_2_early)))
#                 
#                 # Store inf locations based on frequency == 0, using True/False masks
#                 inf_mask_event_1 = (frequency_event_1_early == 0)
#                 inf_mask_event_2 = (frequency_event_2_early == 0)
#                 inf_mask_compound = (frequency_of_compound_events_early == 0)
#                 
#                 inf_mask_across_impact_models_event_1.append(inf_mask_event_1)
#                 inf_mask_across_impact_models_event_2.append(inf_mask_event_2)
#                 inf_mask_across_impact_models_compound.append(inf_mask_compound)
#                 
#                 # Handle inf values in the PR calculations
#                 pr_event_1 = xr.where(np.isfinite(pr_event_1), pr_event_1, np.nan)
#                 pr_event_2 = xr.where(np.isfinite(pr_event_2), pr_event_2, np.nan)
#                 pr_compound = xr.where(np.isfinite(pr_compound), pr_compound, np.nan)
#                 
#                 # Store PRs
#                 pr_for_event_1.append(pr_event_1)
#                 pr_for_event_2.append(pr_event_2)
#                 pr_for_compound_events.append(pr_compound)
#             
#             # After all models have been processed, calculate the most common occurrence of True/False per grid cell
#             most_common_inf_event_1, _ = mode(np.array(inf_mask_across_impact_models_event_1), axis=0)
#             most_common_inf_event_2, _ = mode(np.array(inf_mask_across_impact_models_event_2), axis=0)
#             most_common_inf_compound, _ = mode(np.array(inf_mask_across_impact_models_compound), axis=0)
#             
#             most_common_inf_event_1 = most_common_inf_event_1.squeeze()
#             most_common_inf_event_2 = most_common_inf_event_2.squeeze()
#             most_common_inf_compound = most_common_inf_compound.squeeze()
#             
#             inf_for_event_1.append(most_common_inf_event_1)
#             inf_for_event_2.append(most_common_inf_event_2)
#             inf_for_compound_events.append(most_common_inf_compound)
#         
#         pr_for_all_possible_combinations_and_GCM_per_scenario_event_1.append(pr_for_event_1)
#         pr_for_all_possible_combinations_and_GCM_per_scenario_event_2.append(pr_for_event_2)
#         pr_for_all_possible_combinations_and_GCM_per_scenario_compound.append(pr_for_compound_events)
#         
#         inf_for_all_possible_combinations_and_GCM_per_scenario_event_1.append(inf_for_event_1)
#         inf_for_all_possible_combinations_and_GCM_per_scenario_event_2.append(inf_for_event_2)
#         inf_for_all_possible_combinations_and_GCM_per_scenario_compound.append(inf_for_compound_events)
#     
#     probability_ratios_individual_event_1.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_1)
#     probability_ratios_individual_event_2.append(pr_for_all_possible_combinations_and_GCM_per_scenario_event_2)
#     probability_ratios_compound_events.append(pr_for_all_possible_combinations_and_GCM_per_scenario_compound)
# 
#     inf_locations_individual_event_1.append(inf_for_all_possible_combinations_and_GCM_per_scenario_event_1)
#     inf_locations_individual_event_2.append(inf_for_all_possible_combinations_and_GCM_per_scenario_event_2)
#     inf_locations_compound_events.append(inf_for_all_possible_combinations_and_GCM_per_scenario_compound)
# 
# # Proceed with the calculation of averages and further processing
# 
# 
# 
# 
# # Calculate the averages for each scenario across all GCMs
# average_pr_for_event_1 = []
# average_pr_for_event_2 = []
# average_pr_for_compound_events = []
# 
# average_inf_for_event_1 = []
# average_inf_for_event_2 = []
# average_inf_for_compound_events = []
# 
# for scenario in range(4):  # We have 4 scenarios excluding early industrial
#     averaged_scenario_event_1 = []
#     averaged_scenario_event_2 = []
#     averaged_scenario_compound = []
#     
#     averaged_inf_scenario_event_1 = []
#     averaged_inf_scenario_event_2 = []
#     averaged_inf_scenario_compound = []
#     
#     for i in range(15):  # 15 possible combinations of the 6 extreme events
#         # Gather data across GCMs for the current scenario and event combination
#         gcm_data_event_1 = []
#         gcm_data_event_2 = []
#         gcm_data_compound = []
#         
#         inf_data_event_1 = []
#         inf_data_event_2 = []
#         inf_data_compound = []
#         
#         for gcm_idx, gcm in enumerate(probability_ratios_individual_event_1):
#             data_event_1 = gcm[scenario][i]
#             inf_event_1 = inf_locations_individual_event_1[gcm_idx][scenario][i]
#             
#             if data_event_1 is not None:
#                 gcm_data_event_1.append(data_event_1)
#                 inf_data_event_1.append(inf_event_1)
#         
#         for gcm_idx, gcm in enumerate(probability_ratios_individual_event_2):
#             data_event_2 = gcm[scenario][i]
#             inf_event_2 = inf_locations_individual_event_2[gcm_idx][scenario][i]
#             
#             if data_event_2 is not None:
#                 gcm_data_event_2.append(data_event_2)
#                 inf_data_event_2.append(inf_event_2)
#         
#         for gcm_idx, gcm in enumerate(probability_ratios_compound_events):
#             data_compound = gcm[scenario][i]
#             inf_compound = inf_locations_compound_events[gcm_idx][scenario][i]
#             
#             if data_compound is not None:
#                 gcm_data_compound.append(data_compound)
#                 inf_data_compound.append(inf_compound)
#         
#         # Ensure there is data to average across
#         if gcm_data_event_1:
#             if all(data.shape == gcm_data_event_1[0].shape for data in gcm_data_event_1):
#                 average_pr_across_gcms_event_1 = xr.concat(gcm_data_event_1, dim="gcm").mean(dim="gcm", skipna=True)
#                 average_inf_across_gcms_event_1 = np.logical_or.reduce(inf_data_event_1)
#             else:
#                 print(f"Shape mismatch found in gcm_data_event_1 for scenario {scenario}, event combination {i}")
#                 average_pr_across_gcms_event_1 = None
#                 average_inf_across_gcms_event_1 = None
#             averaged_scenario_event_1.append(average_pr_across_gcms_event_1)
#             averaged_inf_scenario_event_1.append(average_inf_across_gcms_event_1)
#         else:
#             averaged_scenario_event_1.append(None)
#             averaged_inf_scenario_event_1.append(None)
#         
#         if gcm_data_event_2:
#             if all(data.shape == gcm_data_event_2[0].shape for data in gcm_data_event_2):
#                 average_pr_across_gcms_event_2 = xr.concat(gcm_data_event_2, dim="gcm").mean(dim="gcm", skipna=True)
#                 average_inf_across_gcms_event_2 = np.logical_or.reduce(inf_data_event_2)
#             else:
#                 print(f"Shape mismatch found in gcm_data_event_2 for scenario {scenario}, event combination {i}")
#                 average_pr_across_gcms_event_2 = None
#                 average_inf_across_gcms_event_2 = None
#             averaged_scenario_event_2.append(average_pr_across_gcms_event_2)
#             averaged_inf_scenario_event_2.append(average_inf_across_gcms_event_2)
#         else:
#             averaged_scenario_event_2.append(None)
#             averaged_inf_scenario_event_2.append(None)
#         
#         if gcm_data_compound:
#             if all(data.shape == gcm_data_compound[0].shape for data in gcm_data_compound):
#                 average_pr_across_gcms_compound = xr.concat(gcm_data_compound, dim="gcm").mean(dim="gcm", skipna=True)
#                 average_inf_across_gcms_compound = np.logical_or.reduce(inf_data_compound)
#             else:
#                 print(f"Shape mismatch found in gcm_data_compound for scenario {scenario}, event combination {i}")
#                 average_pr_across_gcms_compound = None
#                 average_inf_across_gcms_compound = None
#             averaged_scenario_compound.append(average_pr_across_gcms_compound)
#             averaged_inf_scenario_compound.append(average_inf_across_gcms_compound)
#         else:
#             averaged_scenario_compound.append(None)
#             averaged_inf_scenario_compound.append(None)
#     
#     average_pr_for_event_1.append(averaged_scenario_event_1)
#     average_pr_for_event_2.append(averaged_scenario_event_2)
#     average_pr_for_compound_events.append(averaged_scenario_compound)
#     
#     average_inf_for_event_1.append(averaged_inf_scenario_event_1)
#     average_inf_for_event_2.append(averaged_inf_scenario_event_2)
#     average_inf_for_compound_events.append(averaged_inf_scenario_compound)
# 
# 
# 
# 
# 
# 
# # Generate names of the compound events (in the same order as the calculations above)
# compound_events_names = list(itertools.combinations(extreme_event_categories, 2))
# 
# selected_indices = [9, 11, 5, 1]  # Indices for selected compound events
# 
# # Function `plot_probability_ratios` that can handle three columns
# plot_probability_ratios_for_selected_indices_main = fn.plot_probability_ratios_with_hatches(
#     average_pr_for_event_1, 
#     average_pr_for_event_2, 
#     average_pr_for_compound_events, 
#     compound_events_names, 
#     selected_indices,
#     average_inf_for_event_1, 
#     average_inf_for_event_2, 
#     average_inf_for_compound_events
# )
# 
# 
# 
# second_selected_indices = [2, 6, 13]# river floods and wildfires, droughts and wildfires and crop failures and wildfires pairs
# plot_probability_ratios_for_second_selected_events = fn.plot_probability_ratios_second_selection(
#     average_pr_for_event_1, 
#     average_pr_for_event_2, 
#     average_pr_for_compound_events, 
#     compound_events_names, 
#     second_selected_indices
# )
# 
# 
# 
# 
# =============================================================================



#%% NESTED (FOR) LOOP FOR PLOTTING COMPOUND EXTREME EVENT OCCURRENCE, PROBABILITY OF JOINT OCCURRENCE AND MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS considering different combinations and rcp scenarios

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
    
    # ============================================================================
# =============================================================================
#     # ============================================================================
#     # Average number of years with compound events    
#     average_no_of_years_with_compound_events_per_scenario_and_gcm = [[],[],[],[],[]] #  Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
#         
#     # Length of spell with compound events (consecutive years with the compound events) 
#     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm =  [[],[],[],[],[]] #  Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5    
#     # =============================================================================
#     #     ******************
#     # =============================================================================        
#     
#     
#     ## Extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMs    
#     
#     average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
#     average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
#     average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
# 
# =============================================================================
    
# =============================================================================
#     # Maximum number of years with consecutive join occurrence of an extreme event pair considering all impact models and all their respective driving GCMs
#     average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
# 
# 
#     # Full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios and all GCMS
#     full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
#     
#     
#     #
#     gcms_timeseries_of_joint_occurrence_of_compound_events = []
#     
#     
#     # List of bivariate distributions for constructing: One ellipse per GCM: 
#     all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
#     
#     
#     
# =============================================================================
    
    
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
                
# =============================================================================
#                 all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1861_until_1910 = [] # List with 95th quantile of length of spell with compound events accross the multiple impact models driven by the same GCM
#                 all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1956_until_2005 = []
#                 
#                 
#                 all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = [] # List with maximum no. of years with consecutive compound events accross the multiple impact models driven by the same GCM
#                 all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = []
#                 
# =============================================================================
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910 = [] # List with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM               
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005 = [] 
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005 = []
                
                #gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios
                
# =============================================================================
#                 all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910 = []
#                 all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910 = []
#                 
#                 all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005 = []
#                 all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005 = []
#                 
# =============================================================================
# =============================================================================
#                 
#                 early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
#                 present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
#                 
# =============================================================================
                
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
                        
# =============================================================================
#                         # LENGTH OF SPELL WITH COMPOUND EVENTS (CONSECUTIVE YEARS WITH COMPOUND EVENTS) such that the minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1861_until_1910 = fn.length_of_spell_with_occurrence_of_extreme_event(occurrence_of_compound_events_from_1861_until_1910)
#                         # Remove zeros to ensure only length of spells is captured. Note: minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_1861_until_1910 = xr.where(length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1861_until_1910 > 0, length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1861_until_1910, np.nan)
#                         # 95th quantile of the length of spells
#                         quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1861_until_1910 = length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_1861_until_1910.quantile(0.95, dim = 'time', skipna = True)                      
#                         all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1861_until_1910.append(quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1861_until_1910)
#                         
#                         # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
#                         maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_1861_until_1910)
#                         all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910.append(maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910)
#                         
# =============================================================================
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910) 
                        
# =============================================================================
#                         #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT ACROSS THE FULL TIME SCALE IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
#                         occurrence_of_compound_events_from_1861_until_2005 = fn.compound_event_occurrence(extreme_event_1_from_1861_until_2005, extreme_event_2_from_1861_until_2005) #returns True for locations with occurence of compound events within same location in same year
#                         timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_2005 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1861_until_2005, entire_globe_grid_cell_areas_in_xarray)
#                         
#                         all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_2005) # Appended to the list above with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM & #to_array() changes it from a Dataset to a Dataarray
#                         
#                         # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
#                         timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)                                               
#                         timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)
#                         
#                         # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events ****** NOTE: This variable hasnt been used yet, to be used for bivariate distribution later
#                         timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
#                         early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
#                         
#                         # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
#                         comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
#                         
#                         # probability of occurrence of individual extreme events 1 and 2 (past timeperiod --> reference period for probability ratios)
#                         probability_of_occurrence_of_extreme_event_1_from_1861_until_1910 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_1861_until_1910)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910.append(probability_of_occurrence_of_extreme_event_1_from_1861_until_1910)
#                         
#                         probability_of_occurrence_of_extreme_event_2_from_1861_until_1910 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_1861_until_1910)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910.append(probability_of_occurrence_of_extreme_event_2_from_1861_until_1910)
#                         
# =============================================================================
                
 
                
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
                        
# =============================================================================
#                         # LENGTH OF SPELL WITH COMPOUND EVENTS (CONSECUTIVE YEARS WITH COMPOUND EVENTS) such that the minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1956_until_2005 = fn.length_of_spell_with_occurrence_of_extreme_event(occurrence_of_compound_events_from_1956_until_2005)
#                         # Remove zeros to ensure only length of spells is captured. Note: minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_1956_until_2005 = xr.where(length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1956_until_2005 > 0, length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1956_until_2005, np.nan)
#                         # 95th quantile of the length of spells
#                         quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1956_until_2005 = length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_1956_until_2005.quantile(0.95, dim = 'time', skipna = True)                      
#                         all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1956_until_2005.append(quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_1956_until_2005)
#                                                 
#                         # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
#                         maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_1956_until_2005)
#                         all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005.append(maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005)
#                         
# =============================================================================
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005)                     
                        
# =============================================================================
#                         # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
#                         timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
#                         timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
#                         
#                         # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events
#                         timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
#                         present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
# # =============================================================================
# =============================================================================
#                         
#                         # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
#                         comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
# 
#                         # probability of occurrence of individual extreme events 1 and 2 (during present day)
#                         probability_of_occurrence_of_extreme_event_1_from_1956_until_2005 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_1956_until_2005)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005.append(probability_of_occurrence_of_extreme_event_1_from_1956_until_2005)
#                         
#                         probability_of_occurrence_of_extreme_event_2_from_1956_until_2005 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_1956_until_2005)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005.append(probability_of_occurrence_of_extreme_event_2_from_1956_until_2005)
#                         
# =============================================================================
            
                
                
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[0] + '\n')
                else:   
                    
                    # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
# =============================================================================
#                     average_no_of_years_with_compound_events_from_1861_until_1910 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_no_of_years_with_compound_events_from_1861_until_1910 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
#                     
#                     average_no_of_years_with_compound_events_per_scenario_and_gcm[0].append(average_no_of_years_with_compound_events_from_1861_until_1910) # Append to list containing all data on occurrence per scenario and per GCM. # the index [0] represents the location for data during the Early-Industrial period
#                     
# =============================================================================
# =============================================================================
#                     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1861_until_1910_per_gcm = xr.concat(all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1861_until_1910, dim = 'impact_models').mean(dim = 'impact_models', skipna = True)   
#                     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm[0].append(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1861_until_1910_per_gcm) # Append to list containing all data on occurrence per scenario and per GCM. # the index [0] represents the location for data during the Early-Industrial period
#                     
# =============================================================================
# # =============================================================================
#                     # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 1861 UNTIL 1910 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
#                     average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
#                     average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
#                                                     
# # =============================================================================
# =============================================================================
#                     # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 1861 UNTIL 1910 (PLOTTED)
#                     average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
#                     # Append to list containing all results from all impact models and their driving GCMs
#                     average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[0].append(average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910)
#                     
# =============================================================================
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910)
                    
# =============================================================================
#                     # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE ENTIRE TIME SCALE IN SCENARIO (**list for all the impact models)
#                     timeseries_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005)
#                 
# =============================================================================
# =============================================================================
#                 
#                 # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING EARLY INDUSTRIAL PERIOD)
#                 average_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910, dim = 'models').mean(dim = 'models', skipna = True)
#                 average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                 
#                 average_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910, dim = 'models').mean(dim = 'models', skipna = True)
#                 average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                 
# =============================================================================
                
                
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[1] + '\n')
                else:
                    
# =============================================================================
#                     # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
#                     average_no_of_years_with_compound_events_from_1956_until_2005 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_no_of_years_with_compound_events_from_1956_until_2005 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
#                     
#                     average_no_of_years_with_compound_events_per_scenario_and_gcm[1].append(average_no_of_years_with_compound_events_from_1956_until_2005)  # Append to list containing all data on occurrence per scenario and per GCM. # the index [1] represents the location for data during the present-day period
#                     
# =============================================================================
# =============================================================================
#                     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1956_until_2005_per_gcm = xr.concat(all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1956_until_2005, dim = 'impact_models').mean(dim = 'impact_models', skipna = True)   
#                     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm[1].append(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_1956_until_2005_per_gcm) # Append to list containing all data on occurrence per scenario and per GCM. # the index [1] represents the location for data during the present-day
#                     
# =============================================================================
# =============================================================================
#                     # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
#                     average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
#                     average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
#                                                     
# =============================================================================
                    # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 1956 UNTIL 2005 (PLOTTED)
# =============================================================================
#                     average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
#                     # Append to list containing all results from all impact models and their driving GCMs
#                     average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[1].append(average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005)
#                     
# =============================================================================
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005)
                    
# =============================================================================
#                 # AVERAGE PROBABILITY RATIO (PR) OF AVERAGE OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 COMPARED TO EVENTS FROM 1861 UNTIL 1910 (PLOTTED) ** average across all available impact models driven by the same GCM 
#                 average_probability_ratio_of_occurrence_of_the_compound_events_from_1956_until_2005 = fn.plot_probability_ratio_of_occurrence_of_compound_events(average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005, average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
#                     
# =============================================================================
                
# =============================================================================
#                 # Append list: Full set of timeseries (considering a pair/two extreme events) for all impact models driven by the same GCM 
#                 # Early industrial time
#                 gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)    
#                 # Present age
#                 gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
#                 
# =============================================================================
                
# =============================================================================
#                 # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING PRESENT DAY)
#                 average_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005, dim = 'models').mean(dim = 'models', skipna = True)
#                 average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                 
#                 average_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005, dim = 'models').mean(dim = 'models', skipna = True)
#                 average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                 
#                 
# =============================================================================
                

                
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
                        
# =============================================================================
#                         # LENGTH OF SPELL WITH COMPOUND EVENTS (CONSECUTIVE YEARS WITH COMPOUND EVENTS) such that the minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_2050_until_2099 = fn.length_of_spell_with_occurrence_of_extreme_event(occurrence_of_compound_events_from_2050_until_2099)
#                         # Remove zeros to ensure only length of spells is captured. Note: minimum length of a spells is 1, i.e. when you have a year with impact but no impacts the year before and after)
#                         length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_2050_until_2099 = xr.where(length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_2050_until_2099 > 0, length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_2050_until_2099, np.nan)
#                         # 95th quantile of the length of spells
#                         quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_2050_until_2099 = length_of_spell_with_occurrence_of_extreme_event_considering_one_impact_model_per_extreme_event_considering_only_occurrences_from_2050_until_2099.quantile(0.95, dim = 'time', skipna = True)                      
#                         all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099.append(quantile_95th_of_length_of_spell_with_occurrence_of_compound_event_considering_one_impact_model_per_extreme_event_from_2050_until_2099)
#                            
#                         
#                         # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
#                         maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_2050_until_2099)
#                         all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099.append(maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
#                         
# =============================================================================
                        # TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099)
                        
# =============================================================================
#                         #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT ACROSS THE FULL TIME SCALE IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
#                         occurrence_of_compound_events_from_2006_until_2099 = fn.compound_event_occurrence(extreme_event_1_from_2006_until_2099, extreme_event_2_from_2006_until_2099) #returns True for locations with occurence of compound events within same location in same year
#                         timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2006_until_2099 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_2006_until_2099, entire_globe_grid_cell_areas_in_xarray)
#                         
#                         all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2006_until_2099) # Appended to the list above with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM
#                         
#                         # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
#                         timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
#                         timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
#                         
#                         # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events
#                         timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
#                         end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
#                         
#                         # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
#                         comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)     
#                         
#                         # probability of occurrence of individual extreme events 1 and 2
#                         probability_of_occurrence_of_extreme_event_1_from_2050_until_2099 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_2050_until_2099)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099.append(probability_of_occurrence_of_extreme_event_1_from_2050_until_2099)
#                                                 
#                         probability_of_occurrence_of_extreme_event_2_from_2050_until_2099 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_2050_until_2099)
#                         all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099.append(probability_of_occurrence_of_extreme_event_2_from_2050_until_2099)
#                         
# =============================================================================
                        
                # AREA AFFECTED BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                else:
                    
                    
# =============================================================================
#                     # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
#                     average_no_of_years_with_compound_events_from_2050_until_2099 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_no_of_years_with_compound_events_from_2050_until_2099 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
#                     
# =============================================================================
# =============================================================================
#                     # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 2050 UNTIL 2099 (PLOTTED)
#                     average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099, dim='time').mean(dim='time', skipna= True)
#                     plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
#                     
#                     # 95th quantile of lenght of spell of compound events
#                     average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_about_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099, dim = 'impact_models').mean(dim = 'impact_models', skipna = True)   
#                     
# =============================================================================
# =============================================================================
#                     # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
#                     average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
#                     if scenario == 'rcp26' :
#                         
#                         average_no_of_years_with_compound_events_per_scenario_and_gcm[2].append(average_no_of_years_with_compound_events_from_2050_until_2099) # Append to list containing all data on occurrence per scenario and per GCM. # the index [2] represents the location for data during the end of century under RCP2.6
#                         
#                         average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)    
# # =============================================================================
# =============================================================================
#                         average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[2].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
#                         
#                         average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm[2].append(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099_per_gcm) # Append to list containing all data on occurrence per scenario and per GCM. # the index [2] represents the location for data during the RCP2.6
#                         
# =============================================================================
# =============================================================================
#                         
#                     if scenario == 'rcp60' :
#                         
#                         average_no_of_years_with_compound_events_per_scenario_and_gcm[3].append(average_no_of_years_with_compound_events_from_2050_until_2099) # Append to list containing all data on occurrence per scenario and per GCM. # the index [3] represents the location for data during the end of century under RCP6.0
#                         
#                         average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
# =============================================================================
# =============================================================================
#                         average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[3].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
#                         
#                         average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm[3].append(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099_per_gcm) # Append to list containing all data on occurrence per scenario and per GCM. # the index [3] represents the location for data during the RCP6.0
#                         
# =============================================================================
# =============================================================================
#                     if scenario == 'rcp85' :
#                         
#                         average_no_of_years_with_compound_events_per_scenario_and_gcm[4].append(average_no_of_years_with_compound_events_from_2050_until_2099) # Append to list containing all data on occurrence per scenario and per GCM. # the index [4] represents the location for data during the end of century under RCP8.5
#                         
#                         average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
# # =============================================================================
# =============================================================================
#                         average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[4].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
#                         
#                         average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm[4].append(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_from_2050_until_2099_per_gcm) # Append to list containing all data on occurrence per scenario and per GCM. # the index [4] represents the location for data during the RCP8.5
#                     
# =============================================================================
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099)
                    
# =============================================================================
#                     # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE ENTIRE TIME SCALE IN SCENARIO (**list for all the impact models)
#                     timeseries_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099)
#                 
# =============================================================================
                    
# =============================================================================
#                 # AVERAGE PROBABILITY RATIO (PR) OF AVERAGE OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099 COMPARED TO EVENTS FROM 1861 UNTIL 1910 (PLOTTED) ** average across all available impact models driven by the same GCM 
#                 average_probability_ratio_of_occurrence_of_the_compound_events_from_2050_until_2099 = fn.plot_probability_ratio_of_occurrence_of_compound_events(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099, average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
#                 
# =============================================================================
                
# =============================================================================
#                 gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
#                             
# =============================================================================
                
# =============================================================================
#                 # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING THE END OF THE CENTURY)
#                 if len(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099) == 0 or len(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099) == 0: # checking for an empty array representing no data
#                     print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
#                 else: 
#                 
#                     if scenario == 'rcp26': 
#                     
#                         average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                         average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                     if scenario == 'rcp60' :
#                         
#                         average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                         average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                         
#                     if scenario == 'rcp85' :
#                         
#                         average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                         average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
#                         average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
#                         
#                         
# =============================================================================
                                   
        # COMPARISON OF ALL THE SCENARIOS PER PAIR OF EXTREME EVENTS
        
# =============================================================================
#         # Plot the timeseries showing the fraction of the total pixels affected by the joint occurrence of the compound events (set of two: extreme event 1 and 2) considering all impact models driven by the same GCM    
#         plot_of_timeseries_of_joint_occurrence_of_compound_events = fn.plot_timeseries_fraction_of_area_affected_by_compound_events(timeseries_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, gcm)
#         
# =============================================================================
        # Append all 4 GCMS (50-year) timeseries of compound events
        all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append(timeseries_50_years_of_joint_occurrence_of_compound_events)
        
# =============================================================================
#         # Append all 4 GCM SETS timeseries of occurrence of compound events (unique pair of extreme events). # To be used later for bivariate distribution
#         gcms_timeseries_of_joint_occurrence_of_compound_events.append([timeseries_of_joint_occurrence_of_compound_events, gcm])
#         
#         
#         
#         # BIVARIATE DISTRUBUTION
#         # Plot the pearson correlation coefficient considering fraction of total pixels affected by extreme event 1 and 2 in the same year
#         plot_of_pearson_correlation_coefficient = fn.plot_correlation_with_spearmans_rank_correlation_coefficient(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name, gcm)
#         
#         
#         # append list of bivariate distributions for all gcms to one list
#         all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
#         
# 
# =============================================================================
        
        
        # ANY OTHER INDICES
        # ****** you can add any other desired indices here
          
        
    # AVERAGE NUMBER OF YEARS WITH COMPOUND EVENTS PER SCENARIO AND DRIVING GCM
    
    #plot_average_no_of_years_with_compound_events_per_scenario_and_gcm = fnnnnnnnnnnnnnaverage_no_of_years_with_compound_events_per_scenario_and_gcm
    
# =============================================================================
#     
#     ## AVERAGE PROBABILITY RATIO
#     
#     # Probability Ratio for change in single extreme events (indiviually)   
#     
#     reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
#     
#     reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
#     
#     reference_period_average_probability_of_occurrence_of_compound_events = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
#     
#     for time_scenario in range(len(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models)):
#         
#         scenario_considered = ['early-industrial', 'present day', 'rcp2.6', 'rcp6.0', 'rcp8.5']
#         
#         # Average probability of occurrence of compound events 
#         # early industrial period
#         plot_of_average_probability_of_occurrence_of_compund_events_from_1861_until_1910_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], scenario_considered[0])
#         
#         
#         # Probability ratio
#         if time_scenario != 0: # to avoid the early industrial period which is the reference period for the Probability Ratios for the other periods
#             
#             
#             if len(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[time_scenario]) == 0 or len(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[time_scenario]) == 0 or len(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[time_scenario]) == 0 : # checking for an empty array representing no data
#                 print('No data available on occurrence of compound events for selected impact model and scenario during the period ')
#             else: 
#                 
#                 # Average probability of occurrence of compound events 
#                 # present day
#                 average_probability_of_occurrence_of_compound_events_from_1956_until_2005 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[1], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_probability_of_occurrence_of_compund_events_from_1956_until_2005_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
#                 # rcp26
#                 average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp26 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[2], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp26_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp26, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
#                 # rcp 60
#                 average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp60 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[3], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp60_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp60, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
#                 
#                 if len(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4]) != 0:
#                     #rcp 85
#                     average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp85 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4], dim = 'models').mean(dim = 'models', skipna = True)
#                     plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp85_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp85, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
#                    
#                 
#                 # PROBABILITY RATIO 
#                 
#                 # Due to Change in extreme event 1 only
#                 average_probability_of_occurrence_of_extreme_event_1_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 if time_scenario == 1:
#                     # for time period 1956_until_2005
#                     probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
#                 if time_scenario == 2:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
#                 if time_scenario == 3:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
#                 if time_scenario == 4:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
#                 
#             
#                 
#                 # Due to Change in extreme event 2 only
#                 average_probability_of_occurrence_of_extreme_event_2_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 if time_scenario == 1:
#                     # for time period 1956_until_2005
#                     probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[1], scenario_considered[1])
#                 if time_scenario == 2:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[2])
#                 if time_scenario == 3:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[3])
#                 if time_scenario == 4:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[4])
#                 
#                 
#                 # Due to Change in dependence
#                 average_probability_of_occurrence_of_compound_events = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 if time_scenario == 1:
#                     # for time period 1956_until_2005
#                     probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
#                 if time_scenario == 2:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
#                 if time_scenario == 3:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
#                 if time_scenario == 4:
#                     # for time period 2050_until_2099
#                     probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
#                                     
#                 
#             
# =============================================================================
    
# =============================================================================
#     # AVERAGE MAXIMUM NUMBER OF YEARS WITH JOINT OCCURRENCE OF EXTREME EVENTS
#     for time_scenario in range(len(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models)):
#         
#         scenario_considered = ['early-industrial', 'present day', 'rcp2.6', 'rcp6.0', 'rcp8.5']
#         
#         if len(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario]) == 0 : # checking for an empty array representing no data
#             print('No data available on occurrence of compound events for selected impact model and scenario during the period')
#         
#         else:
#             
#             if time_scenario == 0:
#                 # for time period 1861_until_1910
#                 average_max_no_of_consecutive_years_with_compound_events_from_1861_until_1910 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], scenario_considered[time_scenario])
#             
#             if time_scenario == 1:
#                 # for time period 1956_until_2005
#                 average_max_no_of_consecutive_years_with_compound_events_from_1956_until_2005 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[time_scenario])
#             
#             if time_scenario == 2:
#                 # for time period 2050_until_2099
#                 average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp26 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp26, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
#             
#             if time_scenario == 3:
#                 # for time period 2050_until_2099
#                 average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp60 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp60, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
#             
#             if time_scenario == 4:
#                 # for time period 2050_until_2099
#                 average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp85 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
#                 plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp85, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
#             
#             
# =============================================================================
# =============================================================================
#     # AVERAGE NUMBER OF YEARS WITH COMPOUND EVENTS
#     plot_total_no_of_years_with_compound_event_occurrence = fn.plot_total_no_of_years_with_compound_event_occurrence_considering_all_gcms(average_no_of_years_with_compound_events_per_scenario_and_gcm, extreme_event_1_name, extreme_event_2_name, gcms)
#     
#     # AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS
#     plot_average_probability_of_occurrence_of_the_compound_events = fn.plot_probability_of_occurrence_of_compound_events_considering_all_gcms(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models, extreme_event_1_name, extreme_event_2_name, gcms)
#     all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models.append(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models)
#     
#     # 95TH PERCENTILE OF LENGTH OF SPELL OF COMPOUND EVENTS
#     plot_average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_events = fn.plot_average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_events_considering_all_gcms(average_95th_quantile_of_length_of_spell_with_occurrence_of_compound_event_per_scenario_and_gcm, extreme_event_1_name, extreme_event_2_name, gcms)      
#     
#     
#     # BIVARIATE DISTRIBUTION CONSIDERING ALL GCMs
#     
#     bivariate_plot_considering_all_impact_models_per_gcm = fn.plot_correlation_with_spearmans_rank__correlation_coefficient_considering_scatter_points_from_all_impact_models(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name, gcms)
#     
#     #considering all impact models and all their driving GCMs per extreme event    
#     combined_bivariate_plot_considering_all_impact_models_and_all_their_driving_gcms = fn.plot_correlation_with_spearmans_rank_correlation_coefficient_considering_scatter_points_from_all_impact_models_and_all_gcms(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name)
#     
# =============================================================================
    
    # BOX PLOT COMPARISON OF OCCURRENCE PER EXTREME EVENT PAIR
    
    box_plot_per_compound_event = fn.boxplot_comparing_gcms(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, gcms)
     
    all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append([all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name])
            
                
# COMPARISON PLOT FOR ALL THE 15 BOX PLOTS   
           
all_box_plots = fn.all_boxplots(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, gcms)                
               
#considering all impact models and all their driving GCMs per extreme event  
new_box = fn.comparison_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)

new_box_with_median_values_per_boxplot = fn.comparison_boxplot_with_median_values_per_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)





#%% PLOT MOST DOMINANT COMPOUND EVENT PER SCENARIO 

averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models = []

for compound_event in all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models:
        
    averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models_per_scenario = []
    for gcm in compound_event:
        
        if len(gcm) == 0 : # checking for an empty array representing no data
            print('No data available on occurrence of compound events for selected impact model and scenario during the period ')
        else: 
            
            average_probability_across_gcms = xr.concat(gcm, dim='gcm_data').mean(dim='gcm_data', skipna= True)
            averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models_per_scenario.append(average_probability_across_gcms)
        
    averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models.append(averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models_per_scenario)

scenarios_of_compound_events = [[],[],[],[],[]]
for compound_event in averaged_all_compound_event_combinations_and_gcms_average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models:
    for scenario in range(len(compound_event)):
        scenario_of_compound_event = compound_event[scenario]
        scenarios_of_compound_events[scenario].append(scenario_of_compound_event)


dominant_compound_event_per_scenario_considering_all_gcms = fn.plot_dominant_compound_event_per_scenario_considering_all_gcms(scenarios_of_compound_events, compound_events)



# =============================================================================
# stacked_arrays_c= xr.concat(scenarios_of_compound_events[2], dim='array')
# all_zeros_mask = (stacked_arrays_c == 0).all(dim='array')
# stacked_arrays_masked = xr.where(all_zeros_mask, np.nan, stacked_arrays_c)
# 
# # Find the index of the array with the maximum value at each grid point
# filled_data_array_c = stacked_arrays_c.fillna(-9999) 
# max_array_indices_unmasked_c = filled_data_array_c.argmax(dim='array')
# max_array_indices_masked_c = xr.where((filled_data_array_c[0] == -9999), np.nan, max_array_indices_unmasked_c)
# 
# 
# stacked_arrays_d= xr.concat(scenarios_of_compound_events[0], dim='array')
# 
# # Check if all values across different arrays at each grid point are equal to zero
# all_zeros_mask = (stacked_arrays_d == 0).all(dim='array')
# #stacked_arrays_masked = xr.where(all_zeros_mask, np.nan, stacked_arrays)
# 
# # Find the index of the array with the maximum value at each grid point
# filled_data_array_d = stacked_arrays_d.fillna(-9999) # replace nan with -9999 because the xarray arg.max doesnt work with "All-NaN slice encountered"
# max_array_indices_unmasked_d = filled_data_array_d.argmax(dim='array', skipna=True) # determine index of array with max value : https://docs.xarray.dev/en/stable/generated/xarray.DataArray.argmax.html
# max_array_indices_unmasked_for_non_occurrences_d = xr.where(all_zeros_mask, np.nan, max_array_indices_unmasked_d)
# max_array_indices_masked_d = xr.where((filled_data_array_d[0] == -9999), np.nan, max_array_indices_unmasked_for_non_occurrences_d) # return Nan values in the previous positions
# 
# plt.imshow(max_array_indices_masked_d)
# 
# plt.imshow(all_zeros_mask)
# # =============================================================================
# # 
# # 
# # # Find the index of the array with the maximum value at each grid point
# # filled_data_array_d = stacked_arrays_d.fillna(-9999) 
# # max_array_indices_unmasked_d = filled_data_array_d.argmax(dim='array')
# # max_array_indices_masked_d = xr.where((filled_data_array_d[0] == -9999), np.nan, max_array_indices_unmasked_d)
# # =============================================================================
# 
# # =============================================================================
# # 
# # is_value_present = 4 in max_array_indices_masked_d.values
# # if 4 in is_value_present.value:
# #     print(f"The value {value_to_check} exists in the DataArray.")
# # =============================================================================
# 
# 
# import cartopy.crs as ccrs
# # Setting the projection of the map to cylindrical / Mercator
# fig, axs = plt.subplots(2,2, figsize=(9, 6), subplot_kw = {'projection': ccrs.PlateCarree()})  # , constrained_layout=True
# 
# # since axs is a 2 dimensional array of geozaxes, we have to flatten it into 1D; as explained on a similar example on this page: https://kpegion.github.io/Pangeo-at-AOES/examples/multi-panel-cartopy.html
# axs=axs.flatten()
# 
# for i in range(4):
#     # Concatenate the arrays along a new dimension to create a new xarray dataset
#     stacked_arrays = xr.concat(scenarios_of_compound_events[i], dim='array')
#     
#     # Find the index of the array with the maximum value at each grid point
#     filled_data_array = stacked_arrays.fillna(-9999) 
#     max_array_indices_unmasked = filled_data_array.argmax(dim='array')
#     max_array_indices_masked = xr.where((filled_data_array[0] == -9999), np.nan, max_array_indices_unmasked)
#     
#     # Define the number of unique colors
#     num_colors = 15
#     
#     # Choose a discrete colormap
#     cmap = plt.cm.get_cmap('tab20', num_colors)
#     
#     axs[i].imshow(max_array_indices_masked, extent= [-180, 180, -60, 90])
# =============================================================================

# print total runtime of the code
end_time=datetime.now()
print('Processing duration: {}'.format(end_time - start_time))

print('*******************DONE************************ ')

