import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import hospital_plots as hp
import forecasts as frc
import os

np.random.seed(11)
######################################################
#this file runs hospital pandemic capacity simulations
#takes inpatient admissions, inpatient classification,
#hospital census & capacity, icu census & capacity 
#timeseries data to evaluate rhospital spare capacity 
#and trigger decisions to cancel certain types of 
#procedures to create more spare capacity
######################################################
################
#LOAD INPUT DATA
################
#models run across regions, defined as groups of hospitals
print('loading input data.....')
simulation_region = 'Triangle'

##hospitals comprising each 'region'
facility_list = {}
facility_list['Greenville'] = ['Vidant Medical Center',]
facility_list['Charlotte'] = ['Atrium Health - Pineville', 'Atrium Health - Union', 'Novant Health - Presbyterian', 'Novant Health - Huntersville', 'Novant Health - Matthews', 'Atrium Health - University', 'Atrium Health - Main']
facility_list['UNC'] = ['UNC Medical Center', 'UNC Rex Hospital', 'UNC Chatham Hospital', 'UNC- Caldwell Memorial Hospital', 'Wayne UNC Health Care', 'UNC Rockingham Hospital', 'UNC Lenoir Hospital', 'Johnston UNC Health Care']
facility_list['Triangle'] = ['UNC Rex Hospital', 'UNC Medical Center', 'UNC Chatham Hospital', 'Duke Raleigh Hospital', 'Duke Regional Hospital', 'Duke University Hospital', 'Wake Medical Center - Cary', 'Wake Medical Center', 'Wake Medical Center Cary', 'Durham VA', 'Central Carolina Hospital'] 

#link hospital systems to wastewater systems
wwtp_systems = {}
wwtp_systems['Charlotte'] = ['Charlotte Mecklenburg Utilities',]
wwtp_systems['Triangle'] = ['Orange County Water and Sewer Authority', 'Town of Pittsboro', 'City of Raleigh', 'City of Durham', 'Durham County', 'Town of Cary', 'Town of Morrisville', 'Town of Wake Forest', 'Town of Apex']

#set simulation timeframe
start_value = datetime.strptime('03-01-2020', '%m-%d-%Y')
end_value = datetime.strptime('01-31-2021', '%m-%d-%Y')
calibration_start_datetime = datetime.strptime('04-06-2020', '%d-%m-%Y')
calibration_end_datetime = datetime.strptime('20-12-2020', '%d-%m-%Y')
period_datetime_dict, period_index_dict = frc.make_period_datetimes(start_value)
start_simulation = calibration_end_datetime.timetuple().tm_yday - start_value.timetuple().tm_yday + (calibration_end_datetime.year - start_value.year) * 365
start_simulation_datetime = datetime.strptime('16-01-2021', '%d-%m-%Y')
start_simulation = start_simulation_datetime.timetuple().tm_yday - start_value.timetuple().tm_yday + (start_simulation_datetime.year - start_value.year) * 365
total_synthetic_length = start_simulation + 135
#load admissions/census data at the level of individual hospitals and aggregate to overall 'regions'
#two aggregated regions, (1) UNC (subset of hospitals where we have detailed inpatient data w/MS-DRG codes)
# and (2) the Triangle - all hospitals in the population region covered by the SEIR model 
#returns dictionaries with aggregated regional timeseries (3/1/2020 - 1/31/21) for variables including: 
#hospital census - total, covid +, covid + icu, icu, and ventilators
#admissions - total
#capacity - total, icu, ventilators, 'surge'
print('     ......hospital census')
regional_census_unc = frc.load_facilities(facility_list['UNC'], start_value, end_value, regional_name = 'UNC', show_plot = False)#UNC hospitals are the subset of hospitals for which we have detailed admissions data
regional_census = frc.load_facilities(facility_list[simulation_region], start_value, end_value, regional_name = 'triangle', show_plot = False)
#what are the system capacity ratios between UNC (detailed admission data) and the Triangle (analysis region)
#use these ratios to scale up admissions data from UNC (assume a constant admission/capacity ratio between the regions)
#capacity ratio between the study area and the unc-specific hospital system
reg_cap_ratio = regional_census['total_capacity'][-1]/regional_census_unc['total_capacity'][-1]
icu_cap_ratio = regional_census['total_icu_capacity'][-1]/regional_census_unc['total_icu_capacity'][-1]
#load patient-level hospital data from UNC system
#inpatient_data - detailed admissions of different inpatient types (MS-DRG), aggregated to daily values across the system
inpatient_data = pd.read_csv('aggregated_inpatient_values.csv')
ms_drgs = pd.read_csv('ms_drgs.csv')
#daily admissions, daily_msdrg - 56 x 365, for each MDC groups 1-26 (plus COVID & PRE) average number of patients & msdrg points being admitted via emergency room or scheduled procedures in each day of the year, 2018-19
#daily_admissions_ratio, daily_msdrg_ratio - ratio of the number of patients and msdrg points admitted in Jan & Feb, 2020, compared to the average Jan & Feb 2018-19
#tot_admissions, tot_msdrg - dictionary with timeseries of total daily admissions/msdrg points as a % of that day's 2018-2019 average, for each MDC groups 1-26 (plus COVID & PRE), admitted either through the ER or scheduled procedure
#each dictionary has 3 categories - 'baseline', 'series', and 'complete'. Complete is all days 1/1/20-1/31/21, baseline is 1/1/20 - 3/1/20, series is 3/1/20 - 1/31/21
procedure_type_list = ['MED', 'SURG']
admission_type_list = ['EI', 'IP']
mdc_list = []
for mdc_num in range(1, 26):
  mdc_list.append(str(mdc_num).zfill(2))
mdc_list.append('PRE')

#for regional admissions and total msdrg, calculate average seasonal values from 2018-2019, ratio of pre-covid 2020 to same dates i 2018 & 2019, AND
#timeseries from 2018 - 2020 of admissions, msdrg, and msdrg per admission, and parameters for admission curves when elective procedures are cancelled, when hospital admissions decline voluntarily, and when admissions rebound
print('     ......patient distributions')
daily_admissions, daily_msdrg, daily_admission_ratio, daily_msdrg_ratio, tot_admissions, tot_msdrg, msdrg_per_admit, function_parameters = frc.calibrate_msdrg_changes(inpatient_data, procedure_type_list, admission_type_list, mdc_list, period_index_dict, start_value, reg_cap_ratio)
#make baseline admission plots
#set average pre-covid admissions for each group of med/surg inpatient/emergency category
pre_covid_baselines = {}  
pre_covid_baselines['COVID_MSDRG'] = np.mean(tot_msdrg['baseline']['COVID'])  
for pro_type in procedure_type_list:
  for admit_type in admission_type_list:
    typel = pro_type + '_' + admit_type
    pre_covid_baselines[typel + '_MSDRG'] = np.mean(tot_msdrg['baseline'][typel])
    pre_covid_baselines[typel + '_ADMIT'] = np.mean(tot_msdrg['baseline'][typel])

#Get survival probabilities for each oxygen-need category (survival is not live/die, but remaining in the hospital using a certain piece of equipment)
#survival probabilities are expressed as % of the admitted cohort that is using that type of equipment as a function of the number of days after they were admitted
#multiple cohorts enable multiple survival 'scenarios' for each hospital equipment type
#scenario_list - dictionary of lists of cohort/scenario 'names'
#machine_survival_probabilities - dictionary with a key for each hospital equipment piece, that contains a surival prob. as a function of time for each 'scenario'
prob_type_list = ['_low_probs_unc', '_low_probs', '_high_probs']
machine_type_list = ['room_air', 'o2', 'icu', 'vents']
print('     ......icu requirements')
scenario_lists, machine_survival_probabilities = frc.get_admissions_standards(prob_type_list, machine_type_list)
#this estimates historical patients using a ventilator in the unc system from the total admissions + the oxygen usage estimations b/c numbers are not good from hospital reporting alone
unc_beds, unc_icu, unc_vents = frc.calculate_oxygen_usage(regional_census_unc['admissions'], 20, machine_survival_probabilities, scenario_lists, machine_type_list, '_low_probs_unc')
regional_census_unc['vents_covid'] = unc_vents

#Get coefficients to translate change in hospital admissions to change in icu census
#each MDC + EI/IP admission type has a regression coefficient
icu_coefs = frc.calibrate_icu_census(regional_census, tot_admissions['series'], start_value, calibration_start_datetime, calibration_end_datetime, admission_type_list, mdc_list)  

#Make synthetic SEIHR series for future covid transmission under a 'high' and 'low' scenario
#set total population of the study region
total_pop = 1582112.0

#low transmission scenario assumes transmission begins to slow immediately
days_extended = 10
days_cutoff = 10
#low tranmission scenario represents a decline in new infectious at the start of February
print('project covid hospital demands')
seir_ensemble_low, date_index = frc.make_synthetic_series(regional_census['admissions'], days_extended, days_cutoff, calibration_start_datetime, start_value, end_value, total_pop, start_simulation, total_synthetic_length)
#calculate timeseries of the oxygen type usage from the synthetic (low) admissions series
regional_beds_low, regional_icu_low, regional_vents_low = frc.calculate_oxygen_usage(seir_ensemble_low['h']['3'], 90, machine_survival_probabilities, scenario_lists, machine_type_list, '_low_probs')
#high transmission scenario continues growth for another 30 days
days_extended = 30
days_cutoff = 10
#high transmission scenario represents faster growth in January and slowdown at the end of February
seir_ensemble_high, date_index = frc.make_synthetic_series(regional_census['admissions'], days_extended, days_cutoff, calibration_start_datetime, start_value, end_value, total_pop, start_simulation, total_synthetic_length)
#calculate timeseries of the oxygen type usage from the synthetic (high) admissions series
regional_beds_high, regional_icu_high, regional_vents_high = frc.calculate_oxygen_usage(seir_ensemble_high['h']['3'], 90, machine_survival_probabilities, scenario_lists, machine_type_list, '_low_probs')
#make plots of the seir model inputs/outputs
print('simulate hospital actions')
observed_timeseries_values, synthetic_timeseries_values_low = frc.simulate_icu_usage(regional_census, tot_admissions['series'], tot_msdrg['series'], daily_admissions, daily_admission_ratio, daily_msdrg, daily_msdrg_ratio, seir_ensemble_low['h']['3'], regional_icu_low, icu_coefs, function_parameters, period_index_dict, admission_type_list, mdc_list, start_value, calibration_start_datetime, start_simulation, total_synthetic_length, regional_census['total_icu_capacity'][-1])
observed_timeseries_values, synthetic_timeseries_values_high = frc.simulate_icu_usage(regional_census, tot_admissions['series'], tot_msdrg['series'], daily_admissions, daily_admission_ratio, daily_msdrg, daily_msdrg_ratio, seir_ensemble_high['h']['3'], regional_icu_high, icu_coefs, function_parameters, period_index_dict, admission_type_list, mdc_list, start_value, calibration_start_datetime, start_simulation, total_synthetic_length, regional_census['total_icu_capacity'][-1])

print('make plots')
if not os.path.isdir('hospital_admissions'):
  os.mkdir('hospital_admissions')
hp.plot_admission_baselines(tot_admissions['complete'], 'general', 'admission_lg')
hp.plot_admission_baselines(tot_msdrg['complete'], 'general', 'msdrg_lg')
hp.plot_msdrg_distributions(ms_drgs)

if not os.path.isdir('manuscript_figures'):
  os.mkdir('manuscript_figures')
hp.plot_seir_sens(date_index, seir_ensemble_low, start_simulation - 10)
hp.plot_seir_timeseries(date_index, seir_ensemble_low, seir_ensemble_high, start_simulation)

hp.plot_observed_admissions_figures(regional_census, tot_admissions['series'], procedure_type_list, admission_type_list, daily_admissions, daily_msdrg, observed_timeseries_values, synthetic_timeseries_values_low['msdrg_baseline'], regional_icu_high, date_index, start_value)
for ani_plot in range(0, 4):
  hp.plot_admissions_figures(regional_census, observed_timeseries_values, synthetic_timeseries_values_low, synthetic_timeseries_values_high, seir_ensemble_low['h']['3'], seir_ensemble_high['h']['3'], regional_icu_low, regional_icu_high, start_value, start_simulation, date_index, ani_plot)


