# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:34:17 2025

   Copyright 2025 Sylvan Energy Analytics

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""


import pandas as pd
import numpy as np
import os

project = 'Wind_MinLTFrights'
Utility_POD = 'BPAT.PGE'

# create output directories
if os.path.exists('outputs') == False:
    os.mkdir('outputs')
    
# read input data
project_information = pd.read_csv(os.path.join('inputs','project_info.csv'),index_col='project')
ltf_rights = pd.read_csv(os.path.join('inputs','Utility_LTF_rights.csv'))
ptdfs = pd.read_csv(os.path.join('inputs','PTDFs.csv'))
# The list of flowgates
path_list = list(np.unique(ptdfs['Path']))

# The list of flowgates with historical flow data
path_list_with_data = []
flow_data = []
path_allocation_factors = []
for path in path_list:
    # If the path has historical flow data
    if os.path.exists(os.path.join('inputs','historical_flows',path+'.csv')):
        
        path_list_with_data.append(path)
        # Read historical flow data
        flow_data_tmp = pd.read_csv(os.path.join('inputs','historical_flows',path+'.csv'))
        
        # calculate the maximum flow as the minimum of the TTC and SOL
        flow_data_tmp['TTC'].fillna(999999,inplace=True)
        flow_data_tmp['SOL'].fillna(999999,inplace=True)
        # Instead of doing a interpolation, Elaine filled the missing actual flow with 999999, which will definitely cause a congestion.
        flow_data_tmp['actual flow'].fillna(999999,inplace=True)
        flow_data_tmp['min(TTC,SOL)'] = np.minimum(flow_data_tmp['TTC'],flow_data_tmp['SOL'])
        
        # calculate average TTC
        TTC_average = np.mean(flow_data_tmp['min(TTC,SOL)'])
        
        # calculate utility path allocation factor (capping between 0 and 1)
        # path allocation factor here means how much of a path's capacity is effectively reserved for the utility's use
        path_allocation_factor_tmp = 0
        # sum(PTDF * LTF rights / TTC)
        for _, row in ltf_rights.iterrows():
            # Find the flowgate, POR, POD
            path_allocation_factor_tmp += ptdfs[(ptdfs['Path'] == path) & (ptdfs['POR']==row['POR']) & (ptdfs['POD']==row['POD'])]['PTDF'].iloc[0]*row['LTF rights (MW)']/TTC_average
        path_allocation_factors.append(max(min(path_allocation_factor_tmp,1),0))
        
        # calculate headroom
        flow_data_tmp['headroom'] = flow_data_tmp['min(TTC,SOL)'] - flow_data_tmp['actual flow']
        
        # exclude timepoints for which TTC and SOL or actual flow data is unavailable
        # Elaine's approach is differnt from mine. She didn't do an interpolation here
        flow_data_tmp = flow_data_tmp[(flow_data_tmp['min(TTC,SOL)'] != 999999) & (flow_data_tmp['min(TTC,SOL)'] != 0) & (flow_data_tmp['actual flow'] != 999999) & (flow_data_tmp['actual flow'] != 0)]
        flow_data.append(flow_data_tmp)
    
# Estimate impacts of delivering output to POD on flows across each path based on PTDFs
project_info = project_information.loc[project]
project_hourly_data = pd.read_csv(os.path.join('inputs','project_hourly_data',project_info['Hourly data']))
# "deliverable output" here means the output that is not at risk
project_hourly_data['deliverable output (MW)'] = np.minimum(project_hourly_data['total output (MW)'],project_hourly_data['available LTF tx (MW)'])
project_hourly_data['output at risk (MW)'] = project_hourly_data['total output (MW)'] - project_hourly_data['deliverable output (MW)']

for path in path_list_with_data:
    project_hourly_data[path+'_flow_impact'] = project_hourly_data['output at risk (MW)']*np.array(ptdfs[(ptdfs['POR']==project_info['POI or POD']) & (ptdfs['POD']==Utility_POD) & (ptdfs['Path']==path)]['PTDF'])

# initialize columns to record curtailment probabilit and expected curtailment
project_hourly_data['curtailment probability'] = project_hourly_data['output at risk (MW)']*0
project_hourly_data['expected curtailment (MW)'] = project_hourly_data['output at risk (MW)']*0

# calculate probability of curtailment and average curtailment in each hour with project hourly data based on month-hour headroom distributions
curtailment_prob_tmp = np.zeros(np.shape(project_hourly_data)[0])
curtailment_exp_tmp = np.zeros(np.shape(project_hourly_data)[0])
for HE in range(24):
    for month in range(12):
        total_congestion_probability = 0
        total_average_curtailment = 0
        
        # pull path flow impacts for month-hour bin
        project_hourly_subset = project_hourly_data[(project_hourly_data['month']==month+1) & (project_hourly_data['HE']==HE+1)]
        
        for p_ind in range(len(path_list_with_data)):

            path = path_list_with_data[p_ind]
            
            # pull estimated path flow impacts for month-hour bin
            flow_impact = [project_hourly_subset[path+'_flow_impact']]

            # estimate utility headroom for month-hour bin
            flow_data_subset = flow_data[p_ind][(flow_data[p_ind]['month']==month+1) & (flow_data[p_ind]['HE']==HE+1)]
            utility_headroom = np.transpose([path_allocation_factors[p_ind]*np.maximum(0,flow_data_subset['min(TTC,SOL)'] - flow_data_subset['actual flow'])])
            
            # estimate the curtailed flow for each combination of project hourly output data and historical headroom data
            flow_impact_matrix = np.outer(flow_impact,np.ones(np.shape(utility_headroom)))
            utility_headroom_matrix = np.outer(np.ones(np.shape(flow_impact)),utility_headroom)
            actual_flow_matrix = np.minimum(flow_impact_matrix,utility_headroom_matrix)
            curtailed_flow = flow_impact_matrix - actual_flow_matrix
            
            # calculate the project curtailment required to achieve the curtailed flow based on the corresponding PTDF
            curtailed_output = curtailed_flow/(np.array(ptdfs[(ptdfs['POR']==project_info['POI or POD']) & (ptdfs['POD']==Utility_POD) & (ptdfs['Path']==path)]['PTDF'])[0])
            
            # update the likelihood of curtailment and average curtailment for each hour with hourly project data
            # this approach assumes curtailments due to constraints on different paths are non-overlapping (i.e. could overestimate curtailment)
            curtailment_prob_tmp[project_hourly_subset.index] += np.mean(curtailed_output > 0,axis=1)
            curtailment_exp_tmp[project_hourly_subset.index] += np.mean(curtailed_output,axis=1)

            
project_hourly_data['curtailment probability'] = curtailment_prob_tmp
project_hourly_data['expected curtailment (MW)'] = curtailment_exp_tmp
project_hourly_data['derated output (MW)'] = np.maximum(project_hourly_data['total output (MW)'] -project_hourly_data['expected curtailment (MW)'],0)

project_hourly_data.to_csv(os.path.join('outputs',project+'_results.csv'),index=False)

    

    