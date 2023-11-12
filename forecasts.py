import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit, lsq_linear
import matplotlib.pyplot as plt
import os

def make_period_datetimes(simulation_start_date):
  datetime_dict = {}
  #this sets the start date and end date for each segment (1-3)
  for x in range(1, 4):
    datetime_dict[str(x)] = {}
    if x == 1:
      datetime_dict[str(x)]['start'] = [datetime.strptime('03-01-2020', '%m-%d-%Y'), datetime.strptime('03-13-2020', '%m-%d-%Y')]
      datetime_dict[str(x)]['end'] = [datetime.strptime('04-10-2020', '%m-%d-%Y'), datetime.strptime('04-30-2020', '%m-%d-%Y')]
    elif x == 2:
      datetime_dict[str(x)]['start'] = [datetime.strptime('04-10-2020', '%m-%d-%Y'), datetime.strptime('04-30-2020', '%m-%d-%Y')]
      datetime_dict[str(x)]['end'] = [datetime.strptime('09-1-2020', '%m-%d-%Y'), datetime.strptime('09-10-2020', '%m-%d-%Y')]
    elif x == 3:
      datetime_dict[str(x)]['start'] = [datetime.strptime('09-1-2020', '%m-%d-%Y'), datetime.strptime('09-10-2020', '%m-%d-%Y')]
      datetime_dict[str(x)]['end'] = [datetime.strptime('12-20-2020', '%m-%d-%Y'), datetime.strptime('01-04-2021', '%m-%d-%Y')]

  period_range = {}
  for x in range(1, 4):
    period_range[str(x)] = {}
    for index_use in ['start', 'end']:  
      period_range[str(x)][index_use] = np.zeros(2)
      for this_range_spot in range(0, 2):
        period_range[str(x)][index_use][this_range_spot] = int(datetime_dict[str(x)][index_use][this_range_spot].timetuple().tm_yday - simulation_start_date.timetuple().tm_yday + (datetime_dict[str(x)][index_use][this_range_spot].year - simulation_start_date.year) * 366)
      period_range[str(x)][index_use] = period_range[str(x)][index_use].astype(int)
  return datetime_dict, period_range

def load_facilities(facility_list, start_value, end_value, regional_name = 'none', show_plot = False):

  #load statewide hospital admissions data (monthly files)
  ender_list = ['1016', '1102', '1201', '0104', '0201']
  hospital_admissions = pd.DataFrame()
  counter = 0
  #join hospital input files
  for x in ender_list:
    hospital_admissions_int = pd.read_csv('state_hospital_data/UNC_Data_Request_' + x + '.csv', encoding = 'latin')
    if counter > 1:
      hospital_admissions_int['hospital'] = hospital_admissions_int['hospital_name']
    counter += 1
    
    hospital_admissions = pd.concat([hospital_admissions, hospital_admissions_int])

  #reset indices, set date string to datetime
  hospital_admissions = hospital_admissions.reset_index()
  hospital_admissions['Date'] = pd.to_datetime(hospital_admissions['Date'])
  
  #sort admissions by date
  sorted_all_admissions = hospital_admissions.sort_values(by = 'Date')
  sorted_all_admissions = sorted_all_admissions.reset_index()
    
  daily_hospital_census = {}
  admission_types = ['admissions', 'covid_census', 'total_census', 'total_capacity', 'covid_icu_census', 'total_icu_census', 'total_icu_capacity', 'total_vent_census', 'total_vent_capacity', 'total_surge']
  for x in admission_types:
    daily_hospital_census[x] = {}
  admissions_date = {}

  #loop through hospital list
  for facility in facility_list:
    #find hospital-specific admissions
    ind_admissions = hospital_admissions[hospital_admissions['hospital'] == facility]
    ind_admissions.loc[:,'Date'] = pd.to_datetime(ind_admissions.loc[:,'Date'])
    if len(ind_admissions['Date']) > 0:
      #find bounds of admissions timeseries
      sorted_admissions = ind_admissions.sort_values(by = 'Date')
      sorted_admissions = sorted_admissions.reset_index()
      start_date = sorted_all_admissions['Date'][0]
      end_date = sorted_all_admissions['Date'][len(sorted_all_admissions['Date']) - 1]
      total_length = end_date - start_date
      #initialize census timeseries
      for admission_type in admission_types:
        daily_hospital_census[admission_type][facility] = np.zeros(total_length.days + 1)    
      
      #set admission datetime index
      admissions_date[facility] = []
      for n in range(0, total_length.days + 1):
        daily_dates = start_date + timedelta(n)
        admissions_date[facility].append(daily_dates)
        
      #loop through admission series and set census dictionary timeseries'
      for index, row in sorted_admissions.iterrows():
        index_val = row['Date'] - start_date
        if pd.isnull(row['Number of new patients admitted to an inpatient bed who had confirmed COVID-19 at the time of admission in the past 24 hours']):
          if pd.isnull(row['Number of new patients admitted to an inpatient bed who had suspected COVID-19 at the time of admission in the past 24 hours']):
            daily_hospital_census['admissions'][facility][index_val.days] += 0.0
        else:
          #use suspected COVID cases if that data is available, otherwise use confirmed COVID cases
          if pd.isnull(row['Number of new patients admitted to an inpatient bed who had suspected COVID-19 at the time of admission in the past 24 hours']):
            daily_hospital_census['admissions'][facility][index_val.days] = (row['Number of new patients admitted to an inpatient bed who had confirmed COVID-19 at the time of admission in the past 24 hours'])
          else:
            daily_hospital_census['admissions'][facility][index_val.days] = (row['Number of new patients admitted to an inpatient bed who had confirmed COVID-19 at the time of admission in the past 24 hours']) #+ (row['Number of new patients admitted to an inpatient bed who had suspected COVID-19 at the time of admission in the past 24 hours'])
        #Total COVID patients in the hospital
        if pd.notna(row['Number of COVID-19 Positive Patients In Hospital']):
          daily_hospital_census['covid_census'][facility][index_val.days] = row['Number of COVID-19 Positive Patients In Hospital']
        #Total patients in the hospital
        if pd.notna(row['Inpatient Census (all bed types)']):
          daily_hospital_census['total_census'][facility][index_val.days] = row['Inpatient Census (all bed types)']

        #Total hospital capacity
        if pd.notna(row['Staffed Inpatient Capacity (all bed types)']):
          daily_hospital_census['total_capacity'][facility][index_val.days] = row['Staffed Inpatient Capacity (all bed types)']

        #Total COVID patients in the ICU
        if pd.notna(row['Number of COVID-19 Positive Adults in ICU']):
          daily_hospital_census['covid_icu_census'][facility][index_val.days] = row['Number of COVID-19 Positive Adults in ICU']
          
        #Total patients in the ICU
        if pd.notna(row['Adult ICU Census']):
          if facility == 'Duke University Hospital' and index_val.days > 216 and index_val.days < 264:
            daily_hospital_census['total_icu_census'][facility][index_val.days] = row['Adult ICU Staffed Bed Capacity']
          else:
            daily_hospital_census['total_icu_census'][facility][index_val.days] = row['Adult ICU Census']
            
        #Total ICU capacity
        if pd.notna(row['Adult ICU Staffed Bed Capacity']):
          daily_hospital_census['total_icu_capacity'][facility][index_val.days] = row['Adult ICU Staffed Bed Capacity']

        #Total census of Ventilator patients
        if pd.notna(row['Number of Ventilators In Use']):
          daily_hospital_census['total_vent_census'][facility][index_val.days] = row['Number of Ventilators In Use']
          
        #Total number of ventilators
        if pd.notna(row['Number of Ventilators in Hospital']):
          daily_hospital_census['total_vent_capacity'][facility][index_val.days] = row['Number of Ventilators in Hospital']
          
        #Total surge capacity
        if pd.notna(row['Number of Additional Surge Beds Being Planned']):
          daily_hospital_census['total_surge'][facility][index_val.days] = row['Number of Additional Surge Beds Being Planned']
          
  #clean hospital admission/census data    
  for facility in facility_list:  
    index_list = []
    if facility == 'Western Wake Medical Center' and admissions_date[facility][yy] < datetime.strptime('07-24-2020', '%m-%d-%Y'):
      dontuse = 1
    else: 
      for yy in range(0, len(daily_hospital_census['admissions'][facility])):
        if admissions_date[facility][yy] >= start_value:
        
          ####For census/capacity observations, we need to smooth out bumpy data (i.e, census vals that go from 20 -> 0 -> 20
          for admission_type in admission_types:
            if admission_type != 'admissions':
              #look for large reductions in capacity
              if daily_hospital_census[admission_type][facility][yy] < 0.5*daily_hospital_census[admission_type][facility][yy-1]:
                #if a large rebound occurs, 'smooth out' the drop (i.e., assume data recording error)
                if yy < len(daily_hospital_census[admission_type][facility]) - 14:
                  #look for a rebound during the next two weeks
                  for xxx in range(1, 14):
                    if daily_hospital_census[admission_type][facility][yy+xxx] > 0.8*daily_hospital_census[admission_type][facility][yy-1]:
                      daily_hospital_census[admission_type][facility][yy] = daily_hospital_census[admission_type][facility][yy - 1] 
                else:
                  #if less than two weeks of data remain, assume its an error
                  daily_hospital_census[admission_type][facility][yy] = daily_hospital_census[admission_type][facility][yy - 1] 

    ##scale census/capacity data to final estimate of hospital capacity
    census_types = ['total_census', 'total_icu_census']
    capacity_types = ['total_capacity', 'total_icu_capacity']
    for census_name, capacity_name in zip(census_types, capacity_types):
      for yy in range(0, len(admissions_date[facility])):
        if admissions_date[facility][yy] >= start_value:
          if daily_hospital_census[capacity_name][facility][yy] > 0.0:
            daily_hospital_census[census_name][facility][yy] = daily_hospital_census[census_name][facility][yy] * daily_hospital_census[capacity_name][facility][-1]/daily_hospital_census[capacity_name][facility][yy]
            daily_hospital_census[capacity_name][facility][yy] = daily_hospital_census[capacity_name][facility][-1] * 1.0
            
    for yy in range(0, len(admissions_date[facility])):
      if admissions_date[facility][yy] >= start_value:
        if daily_hospital_census['total_surge'][facility][yy] > 0.0:
          daily_hospital_census['total_surge'][facility][yy] = daily_hospital_census['total_surge'][facility][-1] * 1.0

  ##aggregate hospitals to regions
  regional_census = {}
  admission_types = ['admissions', 'covid_census', 'total_census', 'total_capacity', 'covid_icu_census', 'total_icu_census', 'total_icu_capacity', 'total_vent_census', 'total_vent_capacity', 'total_surge']
  for admission_type in admission_types:
    regional_census[admission_type] = np.zeros(total_length.days + 1)

  #loop through all hospitals and collect aggregated regional data
  for facility in facility_list:  
    index_list = []
    for yy in range(0, len(admissions_date[facility])):
      if admissions_date[facility][yy] >= start_value and admissions_date[facility][yy] <= end_value:
        index_day = admissions_date[facility][yy] - start_value
        for admission_type in admission_types:
          if admission_type == 'total_surge':
            regional_census[admission_type][index_day.days] += daily_hospital_census[admission_type][facility][yy] + daily_hospital_census['total_capacity'][facility][yy]
          else:
            regional_census[admission_type][index_day.days] += daily_hospital_census[admission_type][facility][yy]
        index_list.append(admissions_date[facility][yy])
  
    
  if regional_name == 'none':
    pass
  else:
    if not os.path.isdir('covid_timeseries_agg'):
      os.mkdir('covid_timeseries_agg')
    regional_icu_census = pd.DataFrame(regional_census['total_icu_census'])
    regional_icu_census.to_csv('covid_timeseries_agg/regional_icu_timeseries_' + regional_name + '.csv')
    regional_icu_cov_census = pd.DataFrame(regional_census['covid_icu_census'])
    regional_icu_cov_census.to_csv('covid_timeseries_agg/regional_cov_icu_timeseries_' + regional_name + '.csv')
    regional_icu_capacity = pd.DataFrame(regional_census['total_icu_capacity'])
    regional_icu_capacity.to_csv('covid_timeseries_agg/regional_icu_capacity_timeseries_' + regional_name + '.csv')

  return regional_census

def calibrate_msdrg_changes(inpatient_data, procedure_type_list, admission_type_list, mdc_list, period_range, start_val, capacity_ratio):

  #get dates for inpatient data
  inpatient_data['Dates'] = make_date_list(inpatient_data, 2018, 1, 1)
  inpatient_data = inpatient_data.set_index('Dates')
  
  #find pre-covid baseline (2018 & 2019)
  start_baseline_date = datetime.strptime('01/01/2018', '%d/%m/%Y')
  end_baseline_date = datetime.strptime('01/01/2020', '%d/%m/%Y')
  start_covid_date = datetime.strptime('10/03/2020', '%d/%m/%Y')
  #get 7-day MA of admission/msdrg in each admission category across entire timeseries
  inpatient_data_ma = get_inpatient_moving_averages(inpatient_data, procedure_type_list, admission_type_list, mdc_list, capacity_ratio)
  #get 7-day MA of admission/msdrg for each admission category (seasonal averages)
  daily_admission, daily_msdrg = calculate_baseline_seasonal_patterns(inpatient_data_ma, procedure_type_list, admission_type_list, mdc_list, start_baseline_date, end_baseline_date)
  
  #get the ratio of total admissions & msdrg in Jan/Feb 2020 to the average from the same period in 2018-19 for each admission type  
  daily_admission_ratio, daily_msdrg_ratio = calculate_pre_covid_trends(inpatient_data_ma, daily_admission, daily_msdrg, mdc_list, admission_type_list, end_baseline_date, start_covid_date)
  
  leap_year_date = datetime.strptime('29/02/2020', '%d/%m/%Y')
  start_validation_date = datetime.strptime('01/03/2020', '%d/%m/%Y')
  end_validation_date = datetime.strptime('01/02/2021', '%d/%m/%Y')
  #get total admissions/msdrg during 2020/21 as a pct of the same day of the year in 2018/2019
  tot_admissions, tot_msdrg = calculate_covid_changes(inpatient_data_ma, daily_admission, daily_msdrg, procedure_type_list, admission_type_list, mdc_list, end_baseline_date, start_validation_date, end_validation_date, leap_year_date)
  #parameterize a logit function to fit the observed changes in admissions/msdrg per admit for 3 different periods during the covid period, 
      
  function_parameters, msdrg_per_admit = calibrate_msdrg_codes(tot_admissions['series'], tot_msdrg['series'], daily_admission, daily_msdrg, daily_admission_ratio, daily_msdrg_ratio, admission_type_list, mdc_list, period_range, start_val)
  
  return daily_admission, daily_msdrg, daily_admission_ratio, daily_msdrg_ratio, tot_admissions, tot_msdrg, msdrg_per_admit, function_parameters

def make_date_list(inpatient_data, start_year, start_month, start_date):
  dates_list = []
  days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  days_in_month2 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  current_month = start_month
  current_date = start_date
  current_year = start_year
  #inpatient data is daily starting 1/1/2018
  #make datestring list
  for x in range(0, len(inpatient_data['ADMIT'])):
    date_string = str(current_date).zfill(2)
    month_string = str(current_month).zfill(2)
    year_string = str(current_year)
    datetime_date = datetime.strptime(date_string + '/' + month_string + '/' + year_string, '%d/%m/%Y')
    dates_list.append(datetime_date)
    current_date += 1
    if current_year == 2020:
      if current_date == days_in_month2[current_month-1] + 1:
        current_date = 1
        current_month += 1
    else:
      if current_date == days_in_month[current_month-1] + 1:
        current_date = 1
        current_month += 1
    if current_month == 13:
      current_month = 1
      current_year += 1
  return dates_list

def get_inpatient_moving_averages(inpatient_data, procedure_type_list, admission_type_list, mdc_list, capacity_ratio):
  #this function takes inpatient types and calculates 7-day moving average of admission
  #get total msdrg/admissions on a rolling weekly average
  ##use 4-group categories (type_list)
  for pt in procedure_type_list:
    for at in admission_type_list:
      typel = pt + '_' + at
      inpatient_data[typel + '_MA'] = inpatient_data[typel].rolling(window = 7).mean() * capacity_ratio
      inpatient_data['ADMIT_' + typel + '_MA'] = inpatient_data['ADMIT_' + typel].rolling(window = 7).mean() * capacity_ratio
  #make list of all the mdc categories
  #get rolling average msdrg/admissions for each of the MDC categories (admited via ER and scheduled)
  for mdc_num in mdc_list:
    for at in admission_type_list:
      inpatient_data[at + '_' + mdc_num + '_MA'] = inpatient_data[at + '_' + mdc_num].rolling(window = 7).mean() * capacity_ratio
      inpatient_data['ADMIT_' + at + '_' + mdc_num + '_MA'] = inpatient_data['ADMIT_' + at + '_' + mdc_num].rolling(window = 7).mean() * capacity_ratio
  #get 7-day moving average for covid-like illnesses
  inpatient_data['COVID_MA'] = inpatient_data['COVID'].rolling(window = 7).mean() * capacity_ratio
  inpatient_data['ADMIT_COVID_MA'] = inpatient_data['ADMIT_COVID'].rolling(window = 7).mean() * capacity_ratio

  return inpatient_data
   
def calculate_baseline_seasonal_patterns(inpatient_data, procedure_type_list, admission_type_list, mdc_list, start_sum_date, end_sum_date):
  #this function takes timeseries (2018-2019) of inpatient admissions and msdrg scores and calculates baseline seasonal averages
  days_in_sample = end_sum_date - start_sum_date
  current_date = start_sum_date + timedelta(0)
  total_msdrg = 0.0
  
  #initialize array of average values for each day-of-year
  #for both 4- and 52- category groups
  daily_msdrg = {}
  daily_admission = {}
  for procedure_type in procedure_type_list:
    for admission_type in admission_type_list:
      daily_msdrg[procedure_type + '_' + admission_type] = np.zeros(365)
      daily_admission[procedure_type + '_' + admission_type] = np.zeros(365)
  for mdc_num in mdc_list:
    for admission_type in admission_type_list:
      daily_admission[admission_type + '_' + mdc_num] = np.zeros(365)
      daily_msdrg[admission_type + '_' + mdc_num] = np.zeros(365)
  daily_msdrg['COVID'] = np.zeros(365)
  daily_admission['COVID'] = np.zeros(365)
  
  x = 0
  #calculate day-of-year averages for 2018 and 2019
  bad_start_data_days = 11
  while current_date < end_sum_date:
    x += 1
    day_of_year = current_date.timetuple().tm_yday - 1
    #4-category groups
    for procedure_type in procedure_type_list:
      for admission_type in admission_type_list:
       #seven-day moving averages can't be calculated for the first six days of the timeseries
        if day_of_year == bad_start_data_days:
          daily_msdrg[procedure_type + '_' + admission_type][day_of_year] += inpatient_data.loc[current_date, procedure_type + '_' + admission_type + '_MA']/2.0
          daily_admission[procedure_type + '_' + admission_type][day_of_year] += inpatient_data.loc[current_date,'ADMIT_' + procedure_type + '_' + admission_type + '_MA']/2.0
          for xxx in range(0, bad_start_data_days):
            daily_msdrg[procedure_type + '_' + admission_type][xxx] += inpatient_data.loc[current_date, procedure_type + '_' + admission_type + '_MA']/2.0
            daily_admission[procedure_type + '_' + admission_type][xxx] += inpatient_data.loc[current_date,'ADMIT_' + procedure_type + '_' + admission_type + '_MA']/2.0
        elif day_of_year > bad_start_data_days:
          daily_msdrg[procedure_type + '_' + admission_type][day_of_year] += inpatient_data.loc[current_date, procedure_type + '_' + admission_type + '_MA']/2.0
          daily_admission[procedure_type + '_' + admission_type][day_of_year] += inpatient_data.loc[current_date,'ADMIT_' + procedure_type + '_' + admission_type + '_MA']/2.0
    #52-category groups (only use 2019)
    for mdc_num in mdc_list:
      for admission_type in admission_type_list:
        if mdc_num == '14':
          if current_date.year == 2019:
            daily_admission[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date,'ADMIT_' + admission_type + '_' + mdc_num + '_MA']
            daily_msdrg[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date, admission_type + '_' + mdc_num + '_MA']
        else:
          if day_of_year == bad_start_data_days:
            daily_admission[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date,'ADMIT_' + admission_type + '_' + mdc_num + '_MA']/2.0
            daily_msdrg[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date, admission_type + '_' + mdc_num + '_MA']/2.0
            for xxx in range(0, bad_start_data_days):
              daily_admission[admission_type + '_' + mdc_num][xxx] += inpatient_data.loc[current_date,'ADMIT_' + admission_type + '_' + mdc_num + '_MA']/2.0
              daily_msdrg[admission_type + '_' + mdc_num][xxx] += inpatient_data.loc[current_date, admission_type + '_' + mdc_num + '_MA']/2.0
          elif day_of_year > bad_start_data_days:
            daily_admission[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date,'ADMIT_' + admission_type + '_' + mdc_num + '_MA']/2.0
            daily_msdrg[admission_type + '_' + mdc_num][day_of_year] += inpatient_data.loc[current_date, admission_type + '_' + mdc_num + '_MA']/2.0
        
    #calculate day-of-year averages for covid code baselines
    if day_of_year == bad_start_data_days:
      daily_msdrg['COVID'][day_of_year] += inpatient_data.loc[current_date, 'COVID_MA']/2.0
      daily_admission['COVID'][day_of_year] += inpatient_data.loc[current_date, 'ADMIT_COVID_MA']/2.0
      for xxx in range(0, bad_start_data_days):
        daily_msdrg['COVID'][xxx] += inpatient_data.loc[current_date, 'COVID_MA']/2.0
        daily_admission['COVID'][xxx] += inpatient_data.loc[current_date, 'ADMIT_COVID_MA']/2.0
    elif day_of_year > bad_start_data_days:
      daily_msdrg['COVID'][day_of_year] += inpatient_data.loc[current_date, 'COVID_MA']/2.0
      daily_admission['COVID'][day_of_year] += inpatient_data.loc[current_date, 'ADMIT_COVID_MA']/2.0
    current_date = start_sum_date + timedelta(x)

  return daily_admission, daily_msdrg

def calculate_pre_covid_trends(inpatient_data, daily_admission, daily_msdrg, mdc_list, admission_type_list, end_sum_date, start_covid_date):
  #during Jan - March 2020, calculate pre-covid 2020 trend compared to 2018-2019 baseline
  #this is a single value for each category group (i.e., what was the pre-covid trend for 2020 admissions?)
  day_of_year = 0
  daily_admission_ratio = {}
  daily_msdrg_ratio = {}
  for mdc_num in mdc_list:
    for admission_type in admission_type_list:
      daily_admission_ratio[admission_type + '_' + mdc_num] = 0.0
      daily_msdrg_ratio[admission_type + '_' + mdc_num] = 0.0
    
  ##sum total admissions/msdrg through March 10th
  current_date = end_sum_date + timedelta(0)
  while current_date < start_covid_date:
    for mdc_num in mdc_list:
      for admission_type in admission_type_list:
        daily_msdrg_ratio[admission_type + '_' + mdc_num] += inpatient_data.loc[current_date, admission_type + '_' + mdc_num + '_MA']
        daily_admission_ratio[admission_type + '_' + mdc_num] += inpatient_data.loc[current_date, 'ADMIT_' + admission_type + '_' + mdc_num + '_MA'] 
    current_date = end_sum_date + timedelta(day_of_year)
    day_of_year += 1
  #compare total admissions/msdrg in 2020 to the totals in 2018-19 through March 10th
  for mdc_num in mdc_list:
    for admission_type in admission_type_list:
      daily_msdrg_ratio[admission_type + '_' + mdc_num] = daily_msdrg_ratio[admission_type + '_' + mdc_num] / np.sum(daily_msdrg[admission_type + '_' + mdc_num][:day_of_year])
      daily_admission_ratio[admission_type + '_' + mdc_num] = daily_admission_ratio[admission_type + '_' + mdc_num] / np.sum(daily_admission[admission_type + '_' + mdc_num][:day_of_year])

  return daily_admission_ratio, daily_msdrg_ratio

def calculate_covid_changes(inpatient_data, daily_admission, daily_msdrg, procedure_type_list, admission_type_list, mdc_list, end_baseline_date, start_validation_date, end_validation_date, leap_year_date):
  #Find the percentage of the 2018-2019 average that was experienced in 2020 for each admission type
  tot_msdrg = {}#total msdrg
  tot_admissions = {}#total admissions
  #keep track of timeseries for 3 time periods (baseline: 1/1/20-3/10/20, series: 3/10/20-1/31/21, complete 1/1/20 - 1/31/21)
  for x in ['baseline', 'series', 'complete']:
    tot_msdrg[x] = {}
    tot_admissions[x] = {}
    #get group for each admission type (MED/SURG, EI/IP)
    for procedure_type in procedure_type_list:
      for admission_type in admission_type_list:
        type_name = procedure_type + '_' + admission_type
        tot_msdrg[x][type_name] = []
        tot_admissions[x][type_name] = []
    #also for each MDC group admission type (EI/IP, 1-26 + PRE + COVID)
    for mdc_num in mdc_list:
      for admit_form in admission_type_list:
        tot_msdrg[x][admit_form + '_' + mdc_num] = []
        tot_admissions[x][admit_form + '_' + mdc_num] = []
    tot_msdrg[x]['COVID'] = []
    tot_admissions[x]['COVID'] = []
 
  #begin simulation - covering period from the end of the baseline period (12/31/2019)
  #through the end of the data period (2/1/2021)
  current_date = end_baseline_date + timedelta(0)
  complete_dates = []
  x = 0
  while current_date < end_validation_date:
    if current_date > leap_year_date and current_date.year == 2020:
      day_of_year = current_date.timetuple().tm_yday - 2
    else:
      day_of_year = current_date.timetuple().tm_yday - 1
      
    msdrg_baseline_doy = 0.0
    admission_baseline_doy = 0.0
    #get total admissions/msdrg
    for procedure_type in procedure_type_list:
      for admission_type in admission_type_list:
        type_name = procedure_type + '_' + admission_type
        msdrg_baseline_doy += daily_msdrg[type_name][day_of_year]
        admission_baseline_doy += daily_admission[type_name][day_of_year]

    #find admissions/msdrg as the % of the same day in 2018/19 for all of 2020 and 21
    if current_date.year == 2020 or current_date.year == 2021:
      complete_dates.append(current_date)
      #find 2020-21 7-day MA as a % of the baseline from 2018
      #for admissions - each group is presented as a % of total admissions on that day of the year
      admission_baseline_doy = 0.0
      msdrg_baseline_doy = 0.0
      for procedure_type in procedure_type_list:
        for admission_type in admission_type_list:
          type_name = procedure_type + '_' + admission_type
          admission_baseline_doy += daily_admission[type_name][day_of_year]# sum of all admission baseline values
          msdrg_baseline_doy +=  daily_msdrg[type_name][day_of_year]# sum of all admission baseline values
      
      #find daily admissions as a % of the baseline from that day of the year in 2018 - 2019
      for procedure_type in procedure_type_list:
        for admission_type in admission_type_list:
          type_name = procedure_type + '_' + admission_type
          tot_admissions['complete'][type_name].append(inpatient_data.loc[current_date, 'ADMIT_' + type_name + '_MA']/admission_baseline_doy)
          tot_msdrg['complete'][type_name].append(inpatient_data.loc[current_date, type_name + '_MA']/msdrg_baseline_doy)
      for mdc_num in mdc_list:
        for admit_form in admission_type_list:
          admit_cat = admit_form + '_' + mdc_num
          tot_admissions['complete'][admit_cat].append(inpatient_data.loc[current_date, 'ADMIT_'+ admit_cat + '_MA']/admission_baseline_doy)
          tot_msdrg['complete'][admit_cat].append(inpatient_data.loc[current_date, admit_cat]/msdrg_baseline_doy)
      
      tot_admissions['complete']['COVID'].append(inpatient_data.loc[current_date, 'ADMIT_COVID_MA']/admission_baseline_doy)
      tot_msdrg['complete']['COVID'].append(inpatient_data.loc[current_date, 'COVID_MA']/msdrg_baseline_doy)
      
    #if we are before 3/1/2020, find the 'baseline' % of 2018/19 admissions/msdrg that were being observed in 1/2020 and 2/2020, before covid happened
    #(i.e., we want to know if we were starting 2020 at like 105% of 2018/19 levels to account for growth in the 'baseline' for 2020
    #get timeseries of deviations from the baseline pre-covid (i.e., the pre-existing variability of the data)
    if current_date < start_validation_date:
      #what was the baseline (pre-covid) admission rate for covid-like admission codes (repiratory stuff)
      tot_msdrg['baseline']['COVID'].append(max(inpatient_data.loc[current_date, 'COVID_MA'] - daily_msdrg['COVID'][day_of_year], 0.0)/msdrg_baseline_doy)
      tot_admissions['baseline']['COVID'].append(max(inpatient_data.loc[current_date, 'ADMIT_COVID_MA'] - daily_admission['COVID'][day_of_year], 0.0)/admission_baseline_doy)
      for procedure_type in procedure_type_list:
        for admission_type in admission_type_list:
          type_name = procedure_type + '_' + admission_type
          if daily_msdrg[type_name][day_of_year] > 0.0:
            tot_msdrg['baseline'][type_name].append(inpatient_data.loc[current_date, type_name + '_MA']/daily_msdrg[type_name][day_of_year])
          else:
            tot_msdrg['baseline'][type_name].append(0.0)
          if daily_admission[type_name][day_of_year] > 0.0:
            tot_admissions['baseline'][type_name].append(inpatient_data.loc[current_date, 'ADMIT_' + type_name + '_MA']/daily_admission[type_name][day_of_year])
          else:
            tot_admissions['baseline'][type_name].append(0.0)
          
      for mdc_num in mdc_list:
        for admit_form in admission_type_list:
          admit_cat = admit_form + '_' + mdc_num
          if daily_admission[admit_cat][day_of_year] > 0.0:
            tot_admissions['baseline'][admit_cat].append(inpatient_data.loc[current_date, 'ADMIT_' + admit_cat]/(daily_admission[admit_cat][day_of_year]))
          else:
            tot_admissions['baseline'][admit_cat].append(0.0)
          if daily_msdrg[admit_cat][day_of_year] > 0.0:
            tot_msdrg['baseline'][admit_cat].append(inpatient_data.loc[current_date, admit_cat]/(daily_msdrg[admit_cat][day_of_year]))
          else:
            tot_msdrg['baseline'][admit_cat].append(0.0)
    #after 3/1/2020, do the same thing as above but create a seperate 'covid' series   
    else:
      tot_msdrg['series']['COVID'].append(max(inpatient_data.loc[current_date, 'COVID_MA'] - daily_msdrg['COVID'][day_of_year], 0.0))
      tot_admissions['series']['COVID'].append(max(inpatient_data.loc[current_date, 'ADMIT_COVID_MA'] - daily_admission['COVID'][day_of_year], 0.0))
      for procedure_type in procedure_type_list:
        for admission_type in admission_type_list:
          type_name = procedure_type + '_' + admission_type
          if daily_msdrg[type_name][day_of_year] > 0.0:
            tot_msdrg['series'][type_name].append(inpatient_data.loc[current_date, type_name + '_MA'])
          else:
            tot_msdrg['series'][type_name].append(0.0)
          if daily_admission[type_name][day_of_year] > 0.0:
            tot_admissions['series'][type_name].append(inpatient_data.loc[current_date, 'ADMIT_' + type_name + '_MA'])
          else:
            tot_admissions['series'][type_name].append(0.0)
      for mdc_num in mdc_list:
        for admit_form in admission_type_list:
          admit_cat = admit_form + '_' + mdc_num
          tot_admissions['series'][admit_cat].append(inpatient_data.loc[current_date, 'ADMIT_'+ admit_cat + '_MA'])
          tot_msdrg['series'][admit_cat].append(inpatient_data.loc[current_date, admit_cat + '_MA'])
      
    x += 1
    current_date = end_baseline_date + timedelta(x)

  #recast baseline series as a numpy array
  for procedure_type in procedure_type_list:
    for admission_type in admission_type_list:
      type_name = procedure_type + '_' + admission_type
      tot_admissions['complete'][type_name] = np.asarray(tot_admissions['complete'][type_name])
      tot_msdrg['complete'][type_name] = np.asarray(tot_msdrg['complete'][type_name])
  for mdc_num in mdc_list:
    for admit_form in admission_type_list:
      admit_cat = admit_form + '_' + mdc_num
      tot_admissions['complete'][admit_cat] = np.asarray(tot_admissions['complete'][admit_cat])
      tot_msdrg['complete'][admit_cat] = np.asarray(tot_msdrg['complete'][admit_cat])
  tot_admissions['complete']['COVID'] = np.asarray(tot_admissions['complete']['COVID'])
  tot_msdrg['complete']['COVID'] = np.asarray(tot_msdrg['complete']['COVID'])
  
  return tot_admissions, tot_msdrg

def calibrate_msdrg_codes(admission_type_series, msdrg_series, daily_admissions, daily_msdrg, daily_admission_ratio, daily_msdrg_ratio, admission_type_list, mdc_list, period_range, timeseries_start):
  #this function calibrates 3 curves (for 3 distinct periods) that estimate the change through tim in admissions/msdrg for each admission type
  #the three periods include:
  #period 1 - reduction in admissions/msdrg immediately after non-emergency procedure cancellations
  #period 2 - gradual increase in admissions msdrg immediately after non-emergency procedure bans are lifted
  #period 3 - no mandatory cancellation but voluntary decreate in admissions/msdrg due to rising covid cases        
      
  #all functions take the form : y = s - (s - e)/(1 + e^(-x * (t - z))) (sigmoid function between point s and e) - this is a normal sigmoid function when s = 0 and e = 1
  #function parameters to calibrate
  parameter_list = ['s', 'e', 'x', 'z']
  function_parameters = {}
  for par in parameter_list:
    function_parameters[par + '_1'] = {}
    function_parameters[par + '_2'] = {}

  msdrg_per_admit = {}
  #create loop to calibrate segmented admission functions for each admit type
  all_obs = np.zeros(365)
  all_pred = np.zeros(365)
  for mdc_num in mdc_list:
    for admit_type in admission_type_list:
      admit_cat = admit_type + '_' + mdc_num

      #make estimation of non-covid icu population based on admission rates
      #find how far above/below average the covid-year admissions were for each admission type
      #note: admission_type_series starts 3/1, daily_admissions starts 1/1
      new_admission_series = np.zeros(len(admission_type_series[admit_cat]))
      new_msdrg_series = np.zeros(len(admission_type_series[admit_cat]))
      doy_start_timeseries = timeseries_start.timetuple().tm_yday
      end_of_calendar_year = 365 - doy_start_timeseries
      #(daily admissions * daily_admission_ratio) adjusts the 2018-2019 average for levels observed in early 2020
      new_admission_series[:end_of_calendar_year] = admission_type_series[admit_cat][:end_of_calendar_year] - daily_admissions[admit_cat][doy_start_timeseries:] * daily_admission_ratio[admit_cat]
      new_admission_series[end_of_calendar_year:] = admission_type_series[admit_cat][end_of_calendar_year:] - daily_admissions[admit_cat][:(len(new_admission_series) - end_of_calendar_year)] * daily_admission_ratio[admit_cat]
      new_msdrg_series[:end_of_calendar_year] = msdrg_series[admit_cat][:end_of_calendar_year] - daily_msdrg[admit_cat][doy_start_timeseries:] * daily_msdrg_ratio[admit_cat]
      new_msdrg_series[end_of_calendar_year:] = msdrg_series[admit_cat][end_of_calendar_year:] - daily_msdrg[admit_cat][:(len(new_admission_series) - end_of_calendar_year)] * daily_msdrg_ratio[admit_cat]
      
      for period_num, period_direction in zip(['1', '2', '3'], [False, True, False]):
        #get beginning and ending values for curve-fitting
        start_date1 = period_range[period_num]['start'][0]
        start_date2 = period_range[period_num]['start'][1]
        end_date1 = period_range[period_num]['end'][0]
        end_date2 = period_range[period_num]['end'][1]
        type_start = np.mean(new_admission_series[start_date1:start_date2])
        type_end = np.mean(new_admission_series[end_date1:end_date2])     
        period_admission_series = np.zeros(end_date2 - start_date1)
        
        #seperate out admissions from each period from the overall record
        for xxx in range(start_date1, end_date2):
          period_admission_series[xxx-start_date1] = new_admission_series[xxx] * 1.0
        if period_num == '1':        
          try:
            #fit logit function to the data from the segment
            popt, pcov = curve_fit(f = estimate_logit_syn, xdata = np.arange(end_date2 - start_date1), ydata = period_admission_series, p0 = (type_start, type_end, 1.0, 10.0), maxfev = 50000)
          except:
            popt = (type_start, type_end, 0.1, (end_date2 - start_date1) / 2.0)#default function parameters

        else:
          try:
            epsilon = 0.000001
            #use end_val (function_parameters['e_1']) from previous segment as the 'start_val' for this segment
            popt, pcov = curve_fit(f = estimate_logit_syn, xdata = np.arange(end_date2 - start_date1), ydata = period_admission_series, p0 = (final_val, type_end, 1.0, 10.0 + start_date1), bounds = ([final_val - epsilon, -9999, 0.1, 0.0], [final_val + epsilon, 9999, 100.0, end_date2 - start_date1]), maxfev = 50000)
          except:
#            try:
#              popt, pcov = curve_fit(f = estimate_logit_syn, xdata = np.arange(end_date2 - start_date1), ydata = period_admission_series, p0 = (type_start, type_end, 1.0, 10.0), maxfev = 50000)
#            except:
            popt = (final_val, type_end, 0.1, (end_date2 - start_date1) / 2.0)
        
        current_parameter_category = 'P' + period_num + '_' + str(mdc_num) + '_' + admit_type
        for param_cnt, param_use in enumerate(parameter_list):
          function_parameters[param_use + '_1'][current_parameter_category] = popt[param_cnt] * 1.0
        #seperate out average msdrg from each period from the overall record
        #seperate out average msdrg from each period from the overall record
        est_vals = []
        for x in range(start_date1, end_date2):
          estimate = estimate_logit_syn(x - start_date1, function_parameters['s_1'][current_parameter_category], function_parameters['e_1'][current_parameter_category], function_parameters['x_1'][current_parameter_category], function_parameters['z_1'][current_parameter_category])
          est_vals.append(estimate)
          all_pred[x] += estimate
          all_obs[x] += period_admission_series[x-start_date1]
        final_val = estimate_logit_syn(end_date1 - start_date1, function_parameters['s_1'][current_parameter_category], function_parameters['e_1'][current_parameter_category], function_parameters['x_1'][current_parameter_category], function_parameters['z_1'][current_parameter_category])
        
      for period_num, period_direction in zip(['1', '2', '3'], [False, True, False]):
        #get beginning and ending values for curve-fitting
        start_date1 = period_range[period_num]['start'][0]
        start_date2 = period_range[period_num]['start'][1]
        end_date1 = period_range[period_num]['end'][0]
        end_date2 = period_range[period_num]['end'][1]
        type_start = np.mean(new_msdrg_series[start_date1:start_date2])
        type_end = np.mean(new_msdrg_series[end_date1:end_date2])             
        period_msdrg_series = np.zeros(end_date2 - start_date1)
        for xxx in range(start_date1, end_date2):
          period_msdrg_series[xxx-start_date1] = new_msdrg_series[xxx] * 1.0
        if period_num == '1':        
          try:
            #fit logic function to the data from the current segment
            popt, pcov = curve_fit(f = estimate_logit_syn, xdata = np.arange(end_date2 - start_date1), ydata = period_msdrg_series, p0 = (type_start, type_end, 1.0, 10.0), maxfev = 50000)
          except:
            popt = (type_start, type_end, 1, (end_date2 - start_date1) / 2.0)

        else:
          try:
            epsilon = 0.000001
            #use end_val (function_parameters['e_1']) from previous segment as the 'start_val' for this segment
            popt, pcov = curve_fit(f = estimate_logit_syn, xdata = np.arange(end_date2 - start_date1), ydata = period_msdrg_series, p0 = (final_val, type_end,  1.0, (end_date2 - start_date1)/2.0), bounds = ([final_val - epsilon, -9999, 0.1, 0.0], [final_val + epsilon, 9999, 100.0, end_date2 - start_date1]),  maxfev = 50000)
          except:
            popt = (type_start, type_end,  1, (end_date2 - start_date1) / 2.0)
        current_parameter_category = 'P' + period_num + '_' + str(mdc_num) + '_' + admit_type
        for param_cnt, param_use in enumerate(parameter_list):
          if param_cnt < 2:
            function_parameters[param_use + '_2'][current_parameter_category] = popt[param_cnt] * 1.0
          else:
            function_parameters[param_use + '_2'][current_parameter_category] = popt[param_cnt] * 1.0
        est_vals = []
        for x in range(start_date1, end_date2):
          estimate = estimate_logit_syn(x - start_date1, function_parameters['s_2'][current_parameter_category], function_parameters['e_2'][current_parameter_category], function_parameters['x_2'][current_parameter_category], function_parameters['z_2'][current_parameter_category])
          est_vals.append(estimate)
          all_pred[x] += estimate
          all_obs[x] += period_msdrg_series[x-start_date1]
        final_val = estimate_logit_syn(end_date1 - start_date1, function_parameters['s_2'][current_parameter_category], function_parameters['e_2'][current_parameter_category], function_parameters['x_2'][current_parameter_category], function_parameters['z_2'][current_parameter_category])

  return function_parameters, new_msdrg_series

def estimate_logit_syn(current_date, start_val, end_val, min_x, min_z):
  #logit function for optimization
  logit_val = start_val - (start_val - end_val)/(1 + np.exp((-1*min_x)*(current_date - min_z )))

  return logit_val


def get_admissions_standards(prob_type_list, machine_type_list):
  #this function reads the probability of an admitted patient using a given oxygen machine as a function of time since admission
  scenario_lists = {}
  integrated_probability_dict = {}
  for mt in machine_type_list:#calculate probabilities for different oxygen machines
    integrated_probability_dict[mt] = {}
    for pt_cnt, pt in enumerate(prob_type_list):#calculate probabilities using either unc or mt sinai data (mt. sinai data has 'high' and 'low' estimations
      if pt_cnt == 0:
        if mt == 'icu' or mt == 'vents':
          survival_probs = pd.read_csv('oxygen_use_data/' + mt + '_probs_unc.csv', index_col = 0)
        else:
          survival_probs = pd.read_csv('oxygen_use_data/' + mt + pt + '.csv', index_col = 0)
      else:           
        survival_probs = pd.read_csv('oxygen_use_data/' + mt + pt + '.csv', index_col = 0)
      scenario_lists[mt + pt] = []
      counter = 0
      for x in survival_probs:# this loops through the different 'cohorts' - individual probabilities calculated for different cohorts (so the probabilities as a function of time are a semi-random variable)
        #each scnenario is 'named' and then the list of names are also saved
        if pt_cnt == 0:
          integrated_probability_dict[mt][x] = survival_probs[x]
          scenario_lists[mt + pt].append(x)
        else:
          integrated_probability_dict[mt][str(counter) + '_level_' + str(pt_cnt)] = survival_probs[x]
          scenario_lists[mt + pt].append(str(counter) + '_level_' + str(pt_cnt))
        counter += 1
  return scenario_lists, integrated_probability_dict


def calculate_oxygen_usage(hospitalizations, flow_lookahead, machine_survival_probabilities, scenario_lists, machine_type_list, probability_type):
  #this function takes the oxygen usage estimations from patient flow data
  #and estimates the patients on each oxygen types based on a hosptial admissions timeseries (either historical or synthetic)
  synthetic_length = len(hospitalizations)
  oxygen_usage = {}
  #different oxygen types - room air, 02, icu, vents
  for mt in machine_type_list:
    oxygen_usage[mt] = np.zeros(synthetic_length + flow_lookahead)
  #for each daily admission total, calculate the % of those patients on each oxygen device
  flow_count = 0
  for day_count in range(0, synthetic_length):
    for mt in machine_type_list:
      #randomly select a cohort for their survial probabilities on each oxygen type
      realization_list = scenario_lists[mt + probability_type]
      flow_rlz = realization_list[flow_count]
      flow_count += 1
      if flow_count == len(realization_list):
        flow_count = 0
      machine_probs = machine_survival_probabilities[mt][flow_rlz]
      #add the estimates of future oxygen device usage from today's admissions to the estimate from previous days admissiosn to get a hospital census estimate
      for future_day in range(0, flow_lookahead):
        oxygen_usage[mt][day_count + future_day] += machine_probs.loc[future_day] * hospitalizations[day_count]
    
  total_beds = np.asarray(oxygen_usage['room_air'] + oxygen_usage['o2'])
  total_icu = np.asarray(oxygen_usage['icu'])
  total_vents = np.asarray(oxygen_usage['vents'])

  return total_beds, total_icu, total_vents  
  

def calibrate_icu_census(total_census_project, admission_type_series, start_value, calibration_start, calibration_end, admission_type_list, mdc_list):  

  #loop through hospital census data from unc-only hospital system, and the entire study are hospital system
  total_zeros = True
  covid_zeros = True
  #Loop through all the days of the study period
  for x in range(0,len(total_census_project['total_icu_census'])):
    #Find the first non-zero icu census day
    if total_zeros and total_census_project['total_icu_census'][x] > 0.0:
      first_pos_value = x * 1
      total_zeros = False
    #Find the first non-zero 'covid patients in the icu' day
    if covid_zeros and total_census_project['covid_icu_census'][x] > 0.0:
      #Find the growth in covid patients for the first 5 days with data
      av_slope = 0.0
      for z in range(x+1, x + 6):
        av_slope += (total_census_project['covid_icu_census'][z] - total_census_project['covid_icu_census'][z - 1])/ 5
        
      #Looping backwards from the day with the first covid patient data to the day with the first icu patient data,
      #fill-in empty covid icu patient data based on the growth rate from the first 5 days with covid icu patient data
      #(this basically just assumes linear growth in covid patients before they starting keeping daily covid census data
      for y in range(x - 1, first_pos_value, -1):
        total_census_project['covid_icu_census'][y] = total_census_project['covid_icu_census'][y + 1] - min(av_slope, total_census_project['covid_icu_census'][y + 1]/5)
        av_slope = av_slope * 0.0
      covid_zeros = False

  #find non-covid icu patients (after no-data backfill)   
  non_covid_hospital_project = total_census_project['total_icu_census'] - total_census_project['covid_icu_census']
  
  #We want to find change in non-covid icu census based on change in hospital admissions
  #first set a 'baseline' period with normal admissions/icu populations
  baseline_start = datetime.strptime('09-10-2020', '%m-%d-%Y')
  baseline_end = datetime.strptime('09-30-2020', '%m-%d-%Y')
  baseline_range_start = baseline_start.timetuple().tm_yday - start_value.timetuple().tm_yday + (baseline_start.year - start_value.year) * 365
  baseline_range_end = baseline_end.timetuple().tm_yday - start_value.timetuple().tm_yday + (baseline_end.year - start_value.year) * 365
  
  #then set a 'calibration range'
  calibration_range_start = calibration_start.timetuple().tm_yday - start_value.timetuple().tm_yday + (calibration_start.year - start_value.year) * 365
  calibration_range_end = calibration_end.timetuple().tm_yday - start_value.timetuple().tm_yday + (calibration_end.year - start_value.year) * 365
  
  #Set up a multivariate linear regression, number of days x number of admission groups (51 - 25 MDC groups EI/IP + constant column)
  n_cols = len(mdc_list) * len(admission_type_list) + 1
  A = np.zeros((calibration_range_end - calibration_range_start, n_cols))
  #column values are equal to the change in admissions compared to baseline values
  for mdc_cnt, mdc_num in enumerate(mdc_list):
    for admit_cnt, admit_type in enumerate(admission_type_list):
      baseline_val = np.mean(admission_type_series[admit_type + '_' + mdc_num][baseline_range_start:baseline_range_end])
      for xx in range(calibration_range_start, calibration_range_end):
        A[xx-calibration_range_start, mdc_cnt*len(admission_type_list) + admit_cnt] = baseline_val - admission_type_series[admit_type + '_' + mdc_num][xx]
  A[:,n_cols-1] = np.ones(calibration_range_end - calibration_range_start)#constant coefficient
  #solving for regression coefficients on admission change vs. baseline that best predict icu census change vs. baseline
  values_alt = np.zeros(calibration_range_end - calibration_range_start)
  for xx in range(calibration_range_start, calibration_range_end):
    values_alt[xx-calibration_range_start] = np.mean(non_covid_hospital_project[baseline_range_start:baseline_range_end]) - non_covid_hospital_project[xx]
  #linear regression, return coefficients
  sol = lsq_linear(A, values_alt, bounds = (0, np.inf))
  constants = sol['x']
  synthetic_vals = np.zeros(len(values_alt))
  for x in range(0, len(values_alt)):
    synthetic_vals[x] += np.dot(A[x,:], constants)

  return constants

  
def make_synthetic_series(total_admissions, days_extended, days_cutoff, calibration_start_date, timeseries_start_date, timeseries_end_date, total_pop, start_simulation, total_synthetic_length):
  #parameters for the SEIR simulations
  calibrated_growth_rate = 0.0126#growth rate used to 'fill-in' early timeseries record w/o data
  projected_growth_rate = 0.8#r_eff to use for 'falling covid' portion of simulated record
  calibration_start = calibration_start_date.timetuple().tm_yday - timeseries_start_date.timetuple().tm_yday + (calibration_start_date.year - timeseries_start_date.year) * 365
  num_realizations = 100
  
  #get average new hospital admissions at the beginning of the calibration period (September 20 - December 26)
  #calibration period is used to calibrate SEIR model at the beginning of the simulation period
  initial_value = np.mean(total_admissions[calibration_start:(calibration_start+5)])
        
  #determine what the residual is between observed new hospitalizations and estimate from constant growth rate
  error_list = np.zeros(20)
  for x in range(calibration_start, calibration_start + 20):
    error_list[x-calibration_start] = (total_admissions[x] - initial_value * np.power(1.0 + calibrated_growth_rate, x - calibration_start))/( initial_value * np.power(1.0 + calibrated_growth_rate, x - calibration_start) )
  #back-fill hospital admissions in the beginning of the timeseries based on the observed growth rates when the data begins
  empty_toggle = 0
  empty_counter = 0
  for x in range(1, len(total_admissions)):
    if total_admissions[-1*x] == 0.0 and empty_toggle == 0:
      empty_toggle = 1
      start_value = np.mean(total_admissions[-1*(x-1):-1*(x-5)])
      estimated_value = start_value/(1 + calibrated_growth_rate)
      error_term = np.random.randint(len(error_list))
      total_admissions[-1*x] = estimated_value * (1.0 + error_list[error_term])
    elif total_admissions[-1*x] == 0.0:
      estimated_value = estimated_value / (1 + calibrated_growth_rate)
      error_term = np.random.randint(len(error_list))
      total_admissions[-1*x] = estimated_value * (1.0 + error_list[error_term])
  
  #initialize seir model realizations
  seir_model = {}
  seir_components = ['s', 'e', 'i', 'r_eff', 'h', 'ma_reff']
  for component in seir_components:
    seir_model[component] = np.zeros((total_synthetic_length, num_realizations))
  #initial synthetic hospitalization value
  #last day of observed data
  last_day_used = start_simulation - days_cutoff
  #when does the reproductive number begin to drop?
  begin_reduction_day = last_day_used + days_extended
  #values for randomization of realizations
  ar_residual_counter = 186 + np.random.randint(len(total_admissions) - 200, size = (num_realizations, total_synthetic_length))
  timeseries_cutoff = timeseries_end_date - timedelta(10)
  for realn in range(0, num_realizations):
    #for each realization, we randomize seihr model parameters
    days_of_infection = np.random.randint(8,10)#how long is a person infectious
    days_of_exposure = np.random.randint(2, 5)#how long does it take to become infectious after exposure
    hosp_rate = 0.015 + np.random.rand(1) * 0.01#what is the rate of infections that become hospitalized
    
    #estimation of model parameters during observed timeperiod
    seir_timeseries = {}
    for component in ['s', 'e', 'i', 'r_eff', 'h_res', 'ma_eff']:
      seir_timeseries[component] = np.zeros(len(total_admissions))
    seir_timeseries['s'][0:7] = np.ones(7)*total_pop
    residuals_ma = np.zeros(len(total_admissions) - 14)
    #find SEIR variables from observed hospitalizations
    for x in range(1, len(total_admissions)):
      if timeseries_start_date + timedelta(x) > timeseries_cutoff:
        break
      if x < len(total_admissions) - 7:
        #the infectious population at a timestep is calculated from the 7-day moving average new hospitalizations, divided by the hospitalization rate, and multiplied by the number of days a person is infectious
        seir_timeseries['i'][x] = max(np.mean(total_admissions[x:(x+7)]) * days_of_infection/ hosp_rate, 0.0)
        #the newly exposed population is equal to the change in infectious population, PLUS the population of infectious people who have recovered, and the total exposed population is the newly exposed time the average days of exposure
        seir_timeseries['e'][x] = (seir_timeseries['i'][x] - seir_timeseries['i'][x-1] + seir_timeseries['i'][x - 1]/days_of_infection) * days_of_exposure
        #the susceptible population is equal to the old susceptible population, minum the 'newly' exposed population (change in exposed population plus the number of exposed > infected)
        seir_timeseries['s'][x] = seir_timeseries['s'][x-1] - (seir_timeseries['e'][x] - seir_timeseries['e'][x-1] + seir_timeseries['e'][x - 1]/days_of_exposure)
        #the r_eff value is calculated using the s, e, and i populations
        seir_timeseries['r_eff'][x] = max(days_of_infection * (seir_timeseries['e'][x] - seir_timeseries['e'][x-1] + seir_timeseries['e'][x - 1]/days_of_exposure) * total_pop / (seir_timeseries['s'][x] * seir_timeseries['i'][x]), 0.0)        
        #we also calculate the residual between the 7-day moving average hospitalizations and the daily hospitalization rate
        seir_timeseries['h_res'][x] = (total_admissions[x+7] - np.mean(total_admissions[x:(x+7)]))/np.mean(total_admissions[x:(x+7)])
      if x >= 14:
        #smooth out estimation of r_eff with the moving average, find residuals between MA and estimated value
        seir_timeseries['ma_eff'][x] = np.mean(seir_timeseries['r_eff'][(x-14):x])
        residuals_ma[x-14] = seir_timeseries['r_eff'][x] - seir_timeseries['ma_eff'][x]
    #use an autoregressive model to simulate daily r_eff for the simulated future
    ar_length = 2
    ar_estimates2, ar_residuals2, ar_coef2 = make_ar_model(ar_length, residuals_ma)
    ar_term1 = np.zeros(total_synthetic_length)
    
    #get average initial reff from the end of observed data
    starting_r = np.mean(seir_timeseries['ma_eff'][(last_day_used-40):last_day_used-20])
    ending_r = projected_growth_rate * 1.0
    seir_model_components = ['s', 'e', 'i', 'r_eff']
    for x in range(0, total_synthetic_length):
      random_error_term = ar_residual_counter[realn, x]
      if x < last_day_used:
        for smc in seir_model_components:
          #the beginning of the model is based on observations for s/e/i/h
          seir_model[smc][x, realn] = seir_timeseries[smc][x] * 1.0
        #hospitalizations use 7-day moving average
        if x > 0:
          seir_model['h'][x, realn] = np.mean(total_admissions[max(x-7, 0):x])
        if x > 6:
          #we want to use an autoregressive model to simluate this term to apply to the synthetic series
          #(i.e., the seihr model gives us the moving average value and we want the actual value)
          ar_term1[x] = seir_timeseries['r_eff'][x] - seir_timeseries['ma_eff'][x]
      else:
        if x < len(total_admissions) - 10:
          ar_term1[x] = seir_timeseries['r_eff'][x] - seir_timeseries['ma_eff'][x]
        else:
          #calculate residual between r_eff and its 7-day moving average with this function
          ar_term1[x] = ar_term1[x-1] * ar_coef2[0] +ar_term1[x-2] * ar_coef2[1] + ar_residuals2[x - len(total_admissions) + 10]
        #begin_reduction_day is the timestep at which r_eff starts to decline from its observed value
        if x < begin_reduction_day:
          seir_model['r_eff'][x, realn] = max(ar_term1[x] + starting_r, 0.0)
        else:
          #decline in r_eff happens over a 10-day period
          seir_model['r_eff'][x, realn] = max(ar_term1[x] + ending_r + (starting_r - ending_r) * max(0.0, 1.0 - float(x - begin_reduction_day)/10), 0.0)
        #calculate simulated seir values
        if x > 0:
          seir_model['e'][x, realn] = seir_model['e'][x-1, realn] - seir_model['e'][x-1, realn]/days_of_exposure + seir_model['r_eff'][x, realn] * seir_model['i'][x-1, realn] * seir_model['s'][x-1, realn]/(total_pop * days_of_infection)
          seir_model['s'][x, realn] = seir_model['s'][x-1, realn] - seir_model['r_eff'][x, realn] * seir_model['i'][x-1, realn] * seir_model['s'][x-1, realn]/(total_pop * days_of_infection)
          seir_model['i'][x, realn] =  seir_model['i'][x-1, realn] + seir_model['e'][x-1, realn]/days_of_exposure - seir_model['i'][x-1, realn]/days_of_infection
        mean_admit = seir_model['i'][x, realn] * hosp_rate / days_of_infection
        seir_model['h'][x, realn] = (1.0 +  seir_timeseries['h_res'][random_error_term]) * mean_admit
        
    for x in range(0, total_synthetic_length):
      if x > 14:
        seir_model['ma_reff'][x, realn] = np.mean(seir_model['r_eff'][(x-14):x, realn])
      else:
        seir_model['ma_reff'][x, realn] = np.mean(seir_model['r_eff'][:(x+1), realn])

  gradations = 7
  seir_percentiles = {}
  for seir_component in ['ma_reff', 'h', 'e', 'i', 's']:
    #translate seir from numpy array (total_synthetic_length x num_realizations) into dictionary with the values at distribution percentiles (gradations x total_synthetic_length)
    seir_percentiles[seir_component] = discritize_distribution(seir_model[seir_component], total_synthetic_length, num_realizations, gradations)
  date_index = []
  for t in range(0, total_synthetic_length):
    current_value = timeseries_start_date + timedelta(t)
    date_index.append(current_value)

  return seir_percentiles, date_index
  
def make_ar_model(ar_length, error_list):
  #this function calculates an autoregressive function for the timeseries 'error_list' with lag = ar_length
  dependent = error_list[ar_length:]
  independents = np.zeros((len(error_list[:-1*ar_length]), ar_length))
  #lag the regression inputs
  for x in range(0, ar_length):
    independents[:,x] = error_list[(ar_length - x - 1):(-1*(x+1))]
  #do linear regression
  coef_ar = np.linalg.lstsq(independents, dependent, rcond=None)[0]
  ar_estimate = np.zeros(len(error_list))
  ar_residuals = np.zeros(len(error_list))
  #estimate values with autoregressive function, calculate residuals
  for x in range(ar_length, len(error_list)):
    ar_estimate[x] = 0.0
    for y in range(0, ar_length):
      ar_estimate[x] += coef_ar[y] * error_list[x-y-1]
    ar_residuals[x] = error_list[x] - ar_estimate[x]
  return ar_estimate, ar_residuals, coef_ar

 
def discritize_distribution(distribution_values, distribution_length, numRealizations, gradations):
  new_values = np.zeros(distribution_length)
  ensemble_level_dict = {}
  for z in range(0, gradations + 1):
    ensemble_level_dict[str(z)] = np.zeros(distribution_length)
    
  for yy in range(0, distribution_length):
    value_range = np.zeros(numRealizations)
    for x in range(0, numRealizations):
      value_range[x] = distribution_values[yy][x]
    sorted_range = np.sort(value_range)
    for z in range(0, gradations):
      ensemble_level = int(np.floor(len(sorted_range) * z/gradations))
      ensemble_level_dict[str(z)][yy] = sorted_range[ensemble_level]
    ensemble_level_dict[str(gradations)][yy] = sorted_range[-1]
    
  return ensemble_level_dict
  
def calc_non_covid_icu(admissions_by_type, msdrg_by_type, period_range, admission_type_list, mdc_list):
  #get integer values of the start/end points for the different historical periods

  #set 'baseline' icu conditions to September 2020 (we dont' have icu data pre-covid so this is the best we can do for now but we should be able to get icu data from NCTRaCS soon)
  baseline_period_slice = slice(period_range['3']['start'][0],period_range['3']['start'][1])
  msdrg_per_admit = {}
  for adm_cnt, admit_type in enumerate(admission_type_list):
    for mdc_cnt, mdc_num in enumerate(mdc_list):
      #find timeseries of msdrg per admission for each admission type
      admission_group = admit_type + '_' + mdc_num      
      msdrg_per_admit[admission_group] = np.zeros(len(admissions_by_type[admission_group]))
      if admissions_by_type[admission_group][0] == 0.0:
        #if zero admissions find first day of non-zero admissions
        for x in range(0, len(admissions_by_type[admission_group])):
          if admissions_by_type[admission_group][x] > 0.0:            
            prev_value_count = msdrg_by_type[admission_group][x]/admissions_by_type[admission_group][x]
            break
      #calculated estimated non-covid icu population and the average msdrg per admission
      baseline_admission = np.mean(admissions_by_type[admission_group][baseline_period_slice])
      for x in range(0, len(admissions_by_type[admission_group])):
        if admissions_by_type[admission_group][x] == 0.0:
          msdrg_per_admit[admission_group][x] = prev_value_count * 1.0
        else:  
          msdrg_per_admit[admission_group][x] = msdrg_by_type[admission_group][x]/admissions_by_type[admission_group][x]
          prev_value_count =  msdrg_per_admit[admission_group][x] * 1.0
          
  return msdrg_per_admit

def simulate_icu_usage(total_census, admissions_by_type, msdrg_per_admit, daily_ave_admissions, daily_admission_ratio, daily_ave_msdrg, daily_msdrg_ratio, covid_admission_timeseries, covid_icu_timeseries, icu_coefs, function_parameters, period_range, admission_type_list, mdc_list, observation_start_date, calibration_start_date, start_simulation, synthetic_length, total_icu_capacity):
  #this function calculates non-covid admissions, msdrg per admission, and icu populations
  #timeseries are calculated based on 'triggers' that toggle elective surgical procedures on and off depending on the icu census values
  day_start_index = observation_start_date.timetuple().tm_yday
  calibration_start_index = calibration_start_date.timetuple().tm_yday
  capacity_counter = 0
  begin_er_return = start_simulation + 15
  #get 'baseline' values for each admission group - this is approximating 'normal' conditions for the hospital
  baseline_period_slice = slice(period_range['3']['start'][0],period_range['3']['start'][1])
  baseline_admission_values = {}
  for mdc_num in mdc_list:
    for admit_type in admission_type_list:    
      admission_group = admit_type + '_' + mdc_num    
      baseline_admission_values[admission_group] =  np.mean(admissions_by_type[admission_group][baseline_period_slice])
  baseline_icu = np.mean(total_census['total_icu_census'][baseline_period_slice] - total_census['covid_icu_census'][baseline_period_slice])
  #initialize timeseries arrays for observed and synthetic values of variables we are interested in
  observed_values = {}
  synthetic_values = {}
  for type_use in ['admissions', 'msdrg', 'icu']:
    observed_values[type_use] = np.zeros(start_simulation)
    #synthetic timeseries take different values when we have action/no action & we also want to keep track of errors
    for syn_cat in ['action', 'no_action', 'errors', 'baseline']:
      if type_use != 'icu' or syn_cat == 'errors' or syn_cat == 'baseline':
        synthetic_values[type_use + '_' + syn_cat] = np.zeros(synthetic_length)
      else:
        synthetic_values[type_use + '_' + syn_cat] = np.ones(synthetic_length) * icu_coefs[-1]#intiailize with the linear cconstant from regression, rest of calc comes later
  tot_covid_admissions = 0.0
  tot_covid_msdrg = 0.0
  for syn_x in range(0, start_simulation):
    #set the 'period' of the observed data so we know which function to use to estimate non-covid admissions
    period_num = '3'
    for period_use in range(1, 4):
      if syn_x < period_range[str(period_use)]['end'][0]:
        period_num = str(period_use)
        break
    day_count = syn_x + day_start_index
    if day_count >= 365:
      day_count -= 365
      
    #what index day did the current 'period' start
    start_date_function = period_range[period_num]['start'][0]
      
    #find observed and 'estimated' values for admissions, msdrg, and icu during the period for which we have observed data
    observed_values['icu'][syn_x] = total_census['total_icu_census'][syn_x] - total_census['covid_icu_census'][syn_x]
   
    if observed_values['icu'][syn_x] < 1.0:
      use_synthetic = True
      observed_values['icu'][syn_x] = icu_coefs[-1]
    else:
      use_synthetic = False
    for mdc_cnt, mdc_num in enumerate(mdc_list):
      for adm_cnt, admit_type in enumerate(admission_type_list):      
        admission_group = admit_type + '_' + mdc_num      
        int_admission = admissions_by_type[admission_group][syn_x]
        int_msdrg = msdrg_per_admit[admission_group][syn_x]
        
        if use_synthetic:
          observed_values['icu'][syn_x] += (baseline_admission_values[admission_group] - int_admission) * icu_coefs[mdc_cnt * len(admission_type_list) + adm_cnt]
        observed_values['admissions'][syn_x] += int_admission * 1.0
        observed_values['msdrg'][syn_x] += int_msdrg
        
        synthetic_timestep = syn_x - start_date_function
        param_use = 'P' + period_num + '_' + mdc_num + '_' + admit_type
        #get synthetic admissions from this patient group
        syn_admission_delta = estimate_logit_syn(synthetic_timestep, function_parameters['s_1'][param_use], function_parameters['e_1'][param_use], function_parameters['x_1'][param_use], function_parameters['z_1'][param_use])#calculate difference between actual admissions and expected admissions
        syn_msdrg_per = estimate_logit_syn(synthetic_timestep, function_parameters['s_2'][param_use], function_parameters['e_2'][param_use], function_parameters['x_2'][param_use], function_parameters['z_2'][param_use])#calculate expected msdrg/admission
        expected_admissions = daily_ave_admissions[admission_group][day_count] * daily_admission_ratio[admission_group]#calculate expected admissions for this day of the year
        expected_msdrg = daily_ave_msdrg[admission_group][day_count] * daily_msdrg_ratio[admission_group]#calculate expected admissions for this day of the year
        total_admissions = syn_admission_delta + expected_admissions
        total_msdrg = syn_msdrg_per + expected_msdrg
        #find synthetic values at timestep
        for syn_cat in ['_action', '_no_action']:
          synthetic_values['icu' + syn_cat][syn_x] += (baseline_admission_values[admission_group] - int_admission) * icu_coefs[mdc_cnt * len(admission_type_list) + adm_cnt]
          synthetic_values['admissions' + syn_cat][syn_x] += total_admissions * 1.0
          synthetic_values['msdrg' + syn_cat][syn_x] += total_msdrg * 1.0
        synthetic_values['admissions_baseline'][syn_x] += expected_admissions   
        synthetic_values['msdrg_baseline'][syn_x] += expected_msdrg   
    
    synthetic_values['msdrg_no_action'][syn_x] +=  msdrg_per_admit['COVID'][syn_x]
    synthetic_values['msdrg_action'][syn_x] +=  msdrg_per_admit['COVID'][syn_x]
    observed_values['msdrg'][syn_x] +=  msdrg_per_admit['COVID'][syn_x]
    tot_covid_admissions += admissions_by_type['COVID'][syn_x]
    tot_covid_msdrg += msdrg_per_admit['COVID'][syn_x]
    synthetic_values['icu_action'][syn_x] = baseline_icu - synthetic_values['icu_action'][syn_x]
    synthetic_values['icu_no_action'][syn_x] = baseline_icu - synthetic_values['icu_no_action'][syn_x]
    if use_synthetic:
      observed_values['icu'][syn_x] = baseline_icu - observed_values['icu'][syn_x]
    for data_type in ['icu', 'admissions', 'msdrg']:
      synthetic_values[data_type + '_errors'][syn_x] = observed_values[data_type][syn_x] - synthetic_values[data_type + '_action'][syn_x]
  #simulated icu census timeseries include residuals estimated based on a joint pdf w/ hospital admissions
  av_msdrg_per_admit = tot_covid_admissions / tot_covid_msdrg
  counter = 0
  icu_residual_distribution = {}
  #get 'bins' of hospital admissions for getting icu residual distributions
  admit_bins_left = np.asarray(pd.qcut(pd.Series(total_census['admissions'][(calibration_start_index-day_start_index):start_simulation]), q = 10).cat.categories.left)
  admit_bins_right = np.asarray(pd.qcut(pd.Series(total_census['admissions'][(calibration_start_index-day_start_index):start_simulation]), q = 10).cat.categories.right)
  for lfb, rtb in zip(admit_bins_left, admit_bins_right):
    icu_residual_distribution[str(counter)] = []
    for hist_range in range((calibration_start_index-day_start_index), start_simulation):
      daily_admissions = total_census['admissions'][hist_range]
      if daily_admissions >= lfb and daily_admissions < rtb:
        icu_residual_distribution[str(counter)].append(synthetic_values['icu_errors'][hist_range])
    counter += 1

  #simulated admissions and msdrg timeseries include residuals estimated from observed data with autoregressive functions
  rng = np.random.default_rng(234123415125321)
  ar_residual_distribution = {}
  for type_use in ['admissions', 'msdrg', 'icu']:
    ar_residual_distribution[type_use] = {}
    if type_use == 'icu':
      ar_est, ar_res, ar_coef = make_ar_model(2, synthetic_values[type_use + '_errors'][(start_simulation-110):(start_simulation-70)])
    else:
      ar_est, ar_res, ar_coef = make_ar_model(2, synthetic_values[type_use + '_errors'][(start_simulation-180):(start_simulation-30)])
    
    #get 'bins' of hospital admissions for getting icu residual distributions
    admit_bins_left = np.asarray(pd.qcut(pd.Series(ar_est), q = 5).cat.categories.left)
    admit_bins_right = np.asarray(pd.qcut(pd.Series(ar_est), q = 5).cat.categories.right)
    for lfb, rtb in zip(admit_bins_left, admit_bins_right):
      ar_residual_distribution[type_use][rtb] = []
      for hist_range in range(0, len(ar_est)):
        if ar_est[hist_range] >= lfb and ar_est[hist_range] < rtb:
          ar_residual_distribution[type_use][rtb].append(ar_res[hist_range])
    
    for xx in range(start_simulation, synthetic_length):
      found_bin = 0
      for rtb in admit_bins_right:
        if synthetic_values[type_use + '_errors'][xx] < rtb:
          counter = min(int(rng.random() * len(ar_residual_distribution[type_use][rtb])), len(ar_residual_distribution[type_use][rtb]) - 1)
          ar_error = ar_residual_distribution[type_use][rtb][counter] * 1.0
          found_bin = 1
          break
      if found_bin == 0:
        counter = min(int(rng.random() * len(ar_residual_distribution[type_use][rtb])), len(ar_residual_distribution[type_use][rtb]) - 1)
      ar_error = ar_residual_distribution[type_use][rtb][counter] * 1.0    
      synthetic_values[type_use + '_errors'][xx] = synthetic_values[type_use + '_errors'][xx-1]*ar_coef[0] + synthetic_values[type_use + '_errors'][xx-2]*ar_coef[1] + ar_error      
   
  #simulate future after end of observations   
  capacity_counter = 0
  reduction_toggle = 0
  reduction_toggle_na = 0    
  max_icu_trend = 0.0
  for syn_x in range(start_simulation, synthetic_length):
    day_count = syn_x + day_start_index
    while day_count >= 365:
      day_count -= 365
    
    #at each timestep, estimate the current icu census in both scenarios (w/ and w/o potential hospital actions)
    current_icu_census = synthetic_values['icu_action'][syn_x-1] + covid_icu_timeseries[syn_x - 1]
    current_icu_census_na = synthetic_values['icu_no_action'][syn_x-1] + covid_icu_timeseries[syn_x - 1]
    max_icu_trend = max(np.mean(covid_icu_timeseries[(syn_x - 10):syn_x]), max_icu_trend)
    curr_icu_trend = np.mean(covid_icu_timeseries[(syn_x - 10):(syn_x - 1)])
    
    #used to trigger the proper form of the hospital admission function
    use_recover_cancelled = False
    use_recover_voluntary = False
    use_cancelled_procedures = False
    use_recovery_no_action = False
    #based on the capacity, find which function to describe non-covid hospital admissions (falling from cancelled procedures = p1, falling from voluntary avoidance = p3, recovering = p2)
    #if elective procedures are cancelled (capacity_counter = 1), and icu census levels fall below 80% of icu capacity, lift the restrictions (capacity counter = 2)
    if current_icu_census < total_icu_capacity * 0.8 and capacity_counter == 1 and curr_icu_trend < 0.8 * max_icu_trend:
      capacity_counter = 2
      begin_scheduled = syn_x * 1
      timestep_cancelled_to_recovery = begin_scheduled - start_cancellations
    #if icu census levels reach 97% of icu capacity, or elective procedures were previously cancelled (capacity_counter = 1) and restrictions have not been lifted, cancel elective procedures in this timestep
    if current_icu_census > total_icu_capacity * 0.97 or capacity_counter == 1:
      if capacity_counter == 0 or capacity_counter > 1:
        start_cancellations = syn_x * 1
        timestep_start_from_voluntary = start_cancellations - period_range['3']['start'][0]
        capacity_counter = 1
      use_cancelled_procedures = True
      timestep_cancellations = syn_x - start_cancellations + 14
    #if elective procedures are cancelled (capacity_counter > 0) and the current icu cenus is less than 80% of the capacity, allow elective procedures - once they are allowed (capacity_counter == 2), don't switch them back on
    elif capacity_counter == 2:#if elective procedures have been cancelled, then lifted, continue to have patients return
      use_recover_cancelled = True
      timestep_cancelled_recovery = syn_x - begin_scheduled
    elif (syn_x > begin_er_return and current_icu_census < total_icu_capacity * 0.8 and capacity_counter == 0) or reduction_toggle == 1:#if elective procedures were never cancelled, but pressure in the icu has been released, introduce voluntary return to hospital admissions
      if reduction_toggle == 0:
        begin_reopen = syn_x * 1
        timestep_start_from_voluntary = begin_reopen - period_range['3']['start'][0]
        reduction_toggle += 1      
      use_recover_voluntary = True
      timestep_voluntary_recovery = syn_x - begin_reopen
    else:#if none of those conditions, continue voluntary reduction in admissions due to covid wave
      timestep_voluntary_reductions = syn_x - period_range['3']['start'][0]


    #calculate non-hospital admissions in the no-action scenario (either recovering or falling from voluntary avoidance)
    #for scenario w/ no hospital decisions, either voluntary reduction in hospital admissions or voluntary patient return
    if (syn_x > begin_er_return and current_icu_census_na < total_icu_capacity * 0.8) or reduction_toggle_na == 1:
      if reduction_toggle_na == 0:
        begin_reopen_na = syn_x * 1
        timestep_start_from_voluntary_na = begin_reopen_na - period_range['3']['start'][0]
        reduction_toggle_na += 1
      use_recovery_no_action = True
      timestep_voluntary_recovery_na = syn_x - begin_reopen_na
    else:
      timestep_voluntary_reductions_na = syn_x - period_range['3']['start'][0]
    
    #find observed and 'estimated' values for admissions, msdrg, and icu during the period for which we have observed data
    for mdc_cnt, mdc_num in enumerate(mdc_list):
      for adm_cnt, admit_type in enumerate(admission_type_list):   
        #calculate admissions and msdrg per admit for each patient group and admit type (inpatient/er)      
        admission_group = admit_type + '_' + mdc_num      
        param_type = mdc_num + '_' + admit_type
        #get synthetic admissions from this patient group
        expected_admissions = daily_ave_admissions[admission_group][day_count] * daily_admission_ratio[admission_group]#calculate expected admissions for this day of the year
        expected_msdrg = daily_ave_msdrg[admission_group][day_count] * daily_msdrg_ratio[admission_group]        
        if use_recover_cancelled:
          #the start values in this period depend on when it is triggered and must be calculated
          trend_period_use = 'P3_' + param_type
          start_admit_prev = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          start_msdrg_prev = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
          trend_period_use = 'P1_' + param_type
          start_admit = estimate_logit_syn(timestep_cancelled_to_recovery, start_admit_prev, function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])
          start_msdrg = estimate_logit_syn(timestep_cancelled_to_recovery, start_msdrg_prev, function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])
          #estimate admissions/msdrg per admit from logit functions
          end_admit = np.mean(admissions_by_type[admission_group][:10]) - expected_admissions
          end_msdrg = estimate_logit_syn(0, function_parameters['s_2']['P1_' + param_type], function_parameters['e_2']['P1_' + param_type], function_parameters['x_2']['P1_' + param_type], function_parameters['z_2']['P1_' + param_type])
          trend_period_use = 'P2_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_cancelled_recovery, start_admit, end_admit, function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_cancelled_recovery, start_msdrg, end_msdrg, function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
        elif use_recover_voluntary:
          #the start values in this period depend on when it is triggered and must be calculated          
          trend_period_use = 'P3_' + param_type
          start_admit = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          start_msdrg = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
          end_admit = np.mean(admissions_by_type[admission_group][:10]) - expected_admissions
          end_msdrg =  estimate_logit_syn(0, function_parameters['s_2']['P1_' + param_type], function_parameters['e_2']['P1_' + param_type], function_parameters['x_2']['P1_' + param_type], function_parameters['z_2']['P1_' + param_type])
          #estimate admissions/msdrg per admit from logit functions
          trend_period_use = 'P2_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_voluntary_recovery, start_admit, end_admit, function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_voluntary_recovery, start_msdrg, end_msdrg, function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission        
        elif use_cancelled_procedures:
          #start/end values are known, estimate admissions/msdrg per admit from logit functions
          trend_period_use = 'P3_' + param_type
          start_admit = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          start_msdrg = estimate_logit_syn(timestep_start_from_voluntary, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
          trend_period_use = 'P1_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_cancellations, start_admit, function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_cancellations, start_msdrg, function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
        else:
          #start/end values are known, estimate admissions/msdrg per admit from logit functions
          trend_period_use = 'P3_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_voluntary_reductions, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_voluntary_reductions, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
        
        expected_admissions = daily_ave_admissions[admission_group][day_count] * daily_admission_ratio[admission_group]#calculate expected admissions for this day of the year
        expected_msdrg = daily_ave_msdrg[admission_group][day_count] * daily_msdrg_ratio[admission_group]#calculate expected admissions for this day of the year
        total_admissions = syn_admission_delta + expected_admissions
        total_msdrg = syn_msdrg_per + expected_msdrg
        #find synthetic values at timestep
        synthetic_values['icu_action'][syn_x] += (baseline_admission_values[admission_group] - syn_admission_delta - np.mean(daily_ave_admissions[admission_group]) * daily_admission_ratio[admission_group]) * icu_coefs[mdc_cnt * len(admission_type_list) + adm_cnt]
        synthetic_values['admissions_action'][syn_x] += total_admissions * 1.0
        
        synthetic_values['msdrg_action'][syn_x] += total_msdrg
        synthetic_values['admissions_baseline'][syn_x] += expected_admissions    
        synthetic_values['msdrg_baseline'][syn_x] += expected_msdrg

        #find values of admission/msdrg in the 'no action' scenarios - same as above but there is no option to cancel procedures or recover from procedure cancellation        
        if use_recovery_no_action:
          trend_period_use = 'P3_' + param_type
          start_admit = estimate_logit_syn(timestep_start_from_voluntary_na, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          start_msdrg = estimate_logit_syn(timestep_start_from_voluntary_na, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
          end_admit = np.mean(admissions_by_type[admission_group][:10]) - expected_admissions
          end_msdrg =  estimate_logit_syn(0, function_parameters['s_2']['P1_' + param_type], function_parameters['e_2']['P1_' + param_type], function_parameters['x_2']['P1_' + param_type], function_parameters['z_2']['P1_' + param_type])
          #estimate admissions/msdrg per admit from logit functions
          trend_period_use = 'P2_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_voluntary_recovery_na, start_admit, end_admit, function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_voluntary_recovery_na, start_msdrg, end_msdrg, function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission        
        else:
          trend_period_use = 'P3_' + param_type
          syn_admission_delta = estimate_logit_syn(timestep_voluntary_reductions_na, function_parameters['s_1'][trend_period_use], function_parameters['e_1'][trend_period_use], function_parameters['x_1'][trend_period_use], function_parameters['z_1'][trend_period_use])#calculate difference between actual admissions and expected admissions
          syn_msdrg_per = estimate_logit_syn(timestep_voluntary_reductions_na, function_parameters['s_2'][trend_period_use], function_parameters['e_2'][trend_period_use], function_parameters['x_2'][trend_period_use], function_parameters['z_2'][trend_period_use])#calculate expected msdrg/admission
        
        total_admissions = syn_admission_delta + expected_admissions
        total_msdrg = syn_msdrg_per + expected_msdrg
        #find synthetic values at timestep
        synthetic_values['icu_no_action'][syn_x] += (baseline_admission_values[admission_group] - syn_admission_delta - np.mean(daily_ave_admissions[admission_group]) * daily_admission_ratio[admission_group]) * icu_coefs[mdc_cnt * len(admission_type_list) + adm_cnt]
        synthetic_values['admissions_no_action'][syn_x] += total_admissions * 1.0
        synthetic_values['msdrg_no_action'][syn_x] += total_msdrg            
            
    #add in daily series of random errors for each variable type
    synthetic_values['icu_action'][syn_x] = baseline_icu - synthetic_values['icu_action'][syn_x]
    synthetic_values['icu_no_action'][syn_x] = baseline_icu - synthetic_values['icu_no_action'][syn_x]
    synthetic_values['icu_baseline'][syn_x] = total_icu_capacity * 1.0
    synthetic_values['msdrg_no_action'][syn_x] += av_msdrg_per_admit * covid_admission_timeseries[syn_x]
    synthetic_values['msdrg_action'][syn_x] += av_msdrg_per_admit * covid_admission_timeseries[syn_x]

    #add in random error to variable timeseries
    for type_use in ['icu', 'admissions', 'msdrg']:
      for syn_type in ['_action', '_no_action']:
        synthetic_values[type_use + syn_type][syn_x] += synthetic_values[type_use + '_errors'][syn_x]
    
  return observed_values, synthetic_values

