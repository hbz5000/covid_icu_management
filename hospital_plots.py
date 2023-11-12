import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import scipy.stats as stats

def plot_admission_baselines(admission_complete, group_type, filename):
  #this function plots historical hospital inpatient admissions
  #by either # of admissions or total MS-DRG points of admissions
  baseline_series_date = datetime.strptime('01/01/2020', '%d/%m/%Y')#start of inpatient timeseries input data
  end_series_date = datetime.strptime('01/02/2021', '%d/%m/%Y')#end of inpatient timeseries input data

  #create a datetime index of the timeseries
  complete_dates = []
  x = 0
  current_date = baseline_series_date + timedelta(x)
  while current_date < end_series_date:
    if current_date.year == 2020 or current_date.year == 2021:
      complete_dates.append(current_date)
    x += 1
    current_date = baseline_series_date + timedelta(x)

  #animated plot that shows inpatient admissions by: medical scheduled, surgical scheduled, medical emergency, surgical emergency, and covid
  for ani_num in range(0, 5):
    fig, ax = plt.subplots(figsize = (16, 8))
    ax.fill_between(complete_dates, np.zeros(len(admission_complete['MED_IP'])), admission_complete['MED_IP'], facecolor = 'steelblue', edgecolor = 'black', alpha = 0.7)
    legend_elements = [Patch(facecolor = 'steelblue', edgecolor = 'black', alpha = 0.8, label = 'Scheduled Medical')]
    num_cols = 1
    if ani_num > 0:
      num_cols = 1
      ax.fill_between(complete_dates, admission_complete['MED_IP'], admission_complete['MED_IP'] + admission_complete['SURG_IP'],  facecolor = 'teal', edgecolor = 'black', alpha = 0.7)
      legend_elements.append(Patch(facecolor = 'teal', edgecolor = 'black', alpha = 0.8, label = 'Scheduled Surgical'))                         
    if ani_num > 1:
      num_cols = 2
      ax.fill_between(complete_dates, admission_complete['MED_IP'] + admission_complete['SURG_IP'], admission_complete['MED_IP'] + admission_complete['SURG_IP'] + admission_complete['MED_EI'], facecolor = 'coral', edgecolor = 'black', alpha = 0.7)
      legend_elements.append(Patch(facecolor = 'coral', edgecolor = 'black', alpha = 0.8, label = 'Emergency Department Medical'))
    if ani_num > 2:
      num_cols = 2
      ax.fill_between(complete_dates, admission_complete['MED_IP'] + admission_complete['SURG_IP'] + admission_complete['MED_EI'], admission_complete['MED_IP'] + admission_complete['SURG_IP'] + admission_complete['MED_EI']  + admission_complete['SURG_EI'], facecolor = 'goldenrod', edgecolor = 'black', alpha = 0.7)
      legend_elements.append(Patch(facecolor = 'goldenrod', edgecolor = 'black', alpha = 0.8, label = 'Emergency Department Surgical'))
    if ani_num > 3:
      num_cols = 3
      ax.fill_between(complete_dates, admission_complete['MED_IP'] + admission_complete['SURG_IP'] + admission_complete['MED_EI']  + admission_complete['SURG_EI'], admission_complete['MED_IP'] + admission_complete['SURG_IP'] + admission_complete['MED_EI']  + admission_complete['SURG_EI'] + admission_complete['COVID'], facecolor = 'crimson', edgecolor = 'black', alpha = 0.7)
      legend_elements.append(Patch(facecolor = 'crimson', edgecolor = 'black', alpha = 0.8, label = 'COVID'))
    ax.legend(handles=legend_elements, loc='upper left', ncol = num_cols, prop={'family':'Gill Sans MT','weight':'bold','size':16})

    #gray out area showing the period in which elective surgeries were cancelled
    ax.fill_between([datetime(2020, 3, 16, 0, 0), datetime(2020, 5, 1, 0, 0)],[-700, -700], [700, 700], alpha = 0.15, color = 'black')

    #format plot
    ax.set_ylabel('Inpatient Admissions,\n% of 2018-2019 Baseline', fontsize = 24, weight = 'bold', fontname = 'Gill Sans MT')    
    ax.set_ylim([0, 1.375])
    ax.set_xlim([complete_dates[0], complete_dates[-20]])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals = 0))
    myFmt = mdates.DateFormatter('%b %Y')
    ax.xaxis.set_major_formatter(myFmt)    
    ax.xaxis.set_major_locator(mdates.MonthLocator((1,3,5,7,9,11)))
    for item in (ax.get_xticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')  
    for item in (ax.get_yticklabels()):
      item.set_fontsize(20)
      item.set_fontname('Gill Sans MT')
    fndct ={'family':'Gill Sans MT', 'fontweight': 'bold'}
    props = dict(boxstyle='round', facecolor='beige', alpha=0.8)
    ax.text(0.3, 0.8, 'Elective Procedure\nSuspension', transform=ax.transAxes, fontsize=16, ha='center', va='center', bbox=props, fontdict = fndct)
    plt.savefig('hospital_admissions/' + filename + '_' + str(ani_num) + '.png', dpi = 300)
    plt.close()  

def plot_msdrg_distributions(ms_drgs):
  #this function plots the distribution of inpatient ms-drg scores
  #get list of msdrg codes for surgical/medical procedures
  ms_drg_surg = ms_drgs[ms_drgs['TYPE'] == 'SURG']
  ms_drg_med = ms_drgs[ms_drgs['TYPE'] == 'MED']
  #we want distribution of MS-DRG weights of the available med/surg procedures
  pos = np.linspace(0, max(ms_drg_surg['Weights'].astype(float)), 101)
  pos2 = np.linspace(0, max(ms_drg_med['Weights'].astype(float)), 101)
  #make a kernal density curve of the distribution
  kde_est = stats.gaussian_kde(ms_drg_surg['Weights'].astype(float))
  kde_est2 = stats.gaussian_kde(ms_drg_med['Weights'].astype(float))

  #plot pdfs
  fig, ax = plt.subplots()
  ax.fill_between(pos, np.zeros(len(pos)), kde_est(pos), edgecolor = 'black', alpha = 0.6, facecolor = 'teal')
  ax.fill_between(pos2, np.zeros(len(pos2)), kde_est2(pos2), edgecolor = 'black', alpha = 0.6, facecolor = 'steelblue')
  #format plot
  legend_elements = [Patch(facecolor = 'steelblue', edgecolor = 'black', alpha = 0.8, label = 'Medical Patients'), 
                   Patch(facecolor = 'teal', edgecolor = 'black', alpha = 0.8, label = 'Surgical Patients')]
  ax.legend(handles=legend_elements, loc='upper right', ncol = 1, prop={'family':'Gill Sans MT','weight':'bold','size':14})
  ax.set_xlim([0, max(ms_drg_surg['Weights'].astype(float))])
  ax.set_xlabel('MS-DRG Weight', fontname = 'Gill Sans MT', fontsize=14, fontweight='bold')
  ax.set_ylabel('Frequency', fontname = 'Gill Sans MT', fontsize=14, fontweight='bold')
  ax.set_yticklabels('')
  ax.set_xlim([0, 10])
  ax.set_ylim([0, max(np.max(kde_est(pos)), np.max(kde_est2(pos2)))])
  plt.savefig('hospital_admissions/msdrg_distribution.png', dpi = 300)
  plt.close()

def plot_seir_sens(date_index, seir_ensemble, start_simulation):
  #this function plots the range of the distribution of each seir model component through time
  gradations = 7#distributions are summarized at 7 different 'percentile' levels
  colorsp = sns.color_palette('rocket', 5)
  fig, ax = plt.subplots(5)
  alpha_val = 0.8
  seir_cnt = 0
  #loop through each quadrant and plot the range
  for seir_component, coef in zip(['h', 'e', 'i', 's', 'ma_reff'], [1.0, 1000.0, 1000.0, 1000000.0, 1.0]):
    ax[seir_cnt].fill_between(date_index[:start_simulation], seir_ensemble[seir_component][str(0)][:start_simulation]/coef,  seir_ensemble[seir_component][str(gradations)][:start_simulation]/coef, alpha = alpha_val, color = colorsp[seir_cnt])
    ax[seir_cnt].set_ylim([0.0, 1.1*np.max(seir_ensemble[seir_component][str(gradations)])/coef])
    ax[seir_cnt].set_xlim([date_index[0], date_index[start_simulation]])
    seir_cnt += 1
  #format plot
  ax[4].plot(date_index[:start_simulation], np.zeros(start_simulation), color = 'black', linewidth = 0.5)
  for x in range(0, 4):
    ax[x].set_xticklabels('')
  ax[4].set_ylim([0.5, 2.5])
  ax[3].set_ylim([1.0, 1.1*np.max(seir_ensemble['s'][str(gradations)])/1000000.0])
  ax[0].set_ylabel(' COVID-19 \n Hospital \nAdmissions')
  ax[1].set_ylabel('  Exposed  \n(Thousands)')
  ax[2].set_ylabel('  Infected  \n(Thousands)')
  ax[3].set_ylabel(' Remaining \nSusceptible\n (Million)')
  ax[4].set_ylabel('Reproductive\n    Rate    \n(2-week MA)')
  ax[4].xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
  myFmt = mdates.DateFormatter('%b %Y')
  ax[4].xaxis.set_major_formatter(myFmt)  
  ax[4].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  plt.tight_layout()
  fig.savefig('manuscript_figures/transmission_scenarios_sens.png')
  plt.close()
  
def plot_seir_timeseries(date_index, seir_ensemble_low, seir_ensemble_high, start_simulation):
  #this function plots the median value from the seir output distribution for the different 
  #seir model components under the 'high' and 'low' transmission scenarios
  colorsp = sns.color_palette('rocket', 5)
  fig, ax = plt.subplots(5)
  seir_cnt = 0
  #plot the median value from each distribution
  for seir_component, seir_coef in zip(['h', 'e', 'i', 's', 'ma_reff'], [1.0, 1000.0, 1000.0, 1000000.0, 1.0]):
    ax[seir_cnt].fill_between(date_index, np.zeros(len(seir_ensemble_low[seir_component][str(3)]/seir_coef)), seir_ensemble_low[seir_component][str(3)]/seir_coef, alpha = 0.4, color = colorsp[seir_cnt])
    ax[seir_cnt].plot(date_index[(start_simulation):], seir_ensemble_high[seir_component][str(3)][(start_simulation):]/seir_coef, color = colorsp[0], linewidth = 2.0)
    seir_cnt += 1
  #format plot
  for x in range(0, 4):
    ax[x].set_xticklabels('')
  for x in range(0, 5):
    ax[x].set_xlim([date_index[0], date_index[-1]])
  seir_cnt = 0
  for seir_component, seir_coef, seir_mult in zip(['h', 'e', 'i', 's', 'ma_reff'], [1.0, 1000.0, 1000.0, 1000000.0, 1.0], [1.3, 1.1, 1.1, 1.1, 2.0]):
    ax[seir_cnt].set_ylim([0.0, seir_mult*max(np.max(seir_ensemble_low[seir_component][str(3)]), np.max(seir_ensemble_high[seir_component][str(3)]))/seir_coef])
    seir_cnt += 1
  ax[0].set_ylabel(' COVID-19 \n Hospital \nAdmissions')
  ax[1].set_ylabel('  Exposed  \n(Thousands)')
  ax[2].set_ylabel('  Infected  \n(Thousands)')
  ax[3].set_ylabel(' Remaining \nSusceptible\n (Million)')
  ax[4].set_ylabel('Reproductive\n    Rate    \n(2-week MA)')
  ax[4].xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
  myFmt = mdates.DateFormatter('%b %Y')
  ax[4].xaxis.set_major_formatter(myFmt)  
  ax[4].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  props = dict(boxstyle='round', facecolor='beige', alpha=0.8)
  fndct ={'family':'Gill Sans MT', 'fontweight': 'bold'}
  for x in range(0, 5):
    ax[x].fill_between([date_index[start_simulation], date_index[-1]],[-700, -700], [700, 700], alpha = 0.15, color = 'black')
  for x in range(3, 4):
    legend_elements = [Line2D([0], [0], color=colorsp[x], lw = 2, label = 'High Transmission Scenario'),  
                     Patch(facecolor = colorsp[x], edgecolor = 'black', alpha = 0.4, label = 'Low Tranmission Scenario')]
    ax[x].legend(handles=legend_elements, loc='upper right', ncol = 1, prop={'family':'Gill Sans MT','weight':'bold','size':7})
  ax[0].text(0.875, 0.8, 'Simulated Period', transform=ax[0].transAxes, fontsize=9, ha='center', va='center', bbox=props, fontdict = fndct)
  ax[0].text(0.4, 0.8, 'Observed Period', transform=ax[0].transAxes, fontsize=9, ha='center', va='center', bbox=props, fontdict = fndct)
  ax[0].set_ylim([0, 100])
  ax[1].set_ylim([0, 20])
  ax[2].set_ylim([0, 50])
  ax[3].set_ylim([0, 1.75])
  ax[4].set_ylim([0, 2.0])
  
  plt.tight_layout()
  fig.savefig('manuscript_figures/transmission_scenarios.png')
  plt.close()
  
def plot_admissions_figures(total_census, observed_values_timeseries, synthetic_value_timeseries_low, synthetic_value_timeseries_high, covid_admissions_low, covid_admissions_high, covid_icu_low, covid_icu_high, observation_start_date, simulation_start, date_index, ani_plot):
  #this function plots observed and simulated timeseries for hospital admissions, icu census, and total msdrg (revenue)
  #set up figure    
  day_of_year = observation_start_date.timetuple().tm_yday
  total_synthetic_length = len(synthetic_value_timeseries_high['admissions_action'])
  delta_admissions = {}
  for tp in ['admissions', 'msdrg', 'cum_msdrg', 'icu']:
    for sce in ['low', 'high', 'noaction', 'obs']:
      delta_admissions[tp + '_' + sce] = np.zeros(total_synthetic_length)
  daily_count = 0
  for x in range(0, len(synthetic_value_timeseries_high['admissions_action'])):
    while day_of_year >= 365:
      day_of_year -= 365
    if x < simulation_start:
      delta_admissions['admissions_obs'][x] = observed_values_timeseries['admissions'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
      delta_admissions['admissions_high'][x] = observed_values_timeseries['admissions'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
      delta_admissions['admissions_noaction'][x] = observed_values_timeseries['admissions'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
      delta_admissions['admissions_low'][x] = observed_values_timeseries['admissions'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
    else:
      delta_admissions['admissions_high'][x] = synthetic_value_timeseries_high['admissions_action'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
      delta_admissions['admissions_noaction'][x] = synthetic_value_timeseries_high['admissions_no_action'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
      delta_admissions['admissions_low'][x] = synthetic_value_timeseries_low['admissions_action'][x] - synthetic_value_timeseries_high['admissions_baseline'][x]
    if x < simulation_start:
      delta_admissions['msdrg_obs'][x] = (observed_values_timeseries['msdrg'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x]) /synthetic_value_timeseries_high['msdrg_baseline'][x]
      delta_admissions['msdrg_high'][x] = (observed_values_timeseries['msdrg'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x]) /synthetic_value_timeseries_high['msdrg_baseline'][x]
      delta_admissions['msdrg_noaction'][x] = (observed_values_timeseries['msdrg'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x]) /synthetic_value_timeseries_high['msdrg_baseline'][x]
      delta_admissions['msdrg_low'][x] = (observed_values_timeseries['msdrg'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x]) /synthetic_value_timeseries_high['msdrg_baseline'][x]

      delta_admissions['cum_msdrg_obs'][x] = (np.sum(observed_values_timeseries['msdrg'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
      delta_admissions['cum_msdrg_high'][x] = (np.sum(observed_values_timeseries['msdrg'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
      delta_admissions['cum_msdrg_noaction'][x] = (np.sum(observed_values_timeseries['msdrg'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
      delta_admissions['cum_msdrg_low'][x] = (np.sum(observed_values_timeseries['msdrg'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
    else:
      delta_admissions['cum_msdrg_high'][x] = (np.sum(synthetic_value_timeseries_high['msdrg_action'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
      delta_admissions['cum_msdrg_noaction'][x] = (np.sum(synthetic_value_timeseries_high['msdrg_no_action'][:x]) - np.sum(synthetic_value_timeseries_high['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_high['msdrg_baseline'][1:365])
      delta_admissions['cum_msdrg_low'][x] = (np.sum(synthetic_value_timeseries_low['msdrg_action'][:x]) - np.sum(synthetic_value_timeseries_low['msdrg_baseline'][:x]))/np.sum(synthetic_value_timeseries_low['msdrg_baseline'][1:365])
      delta_admissions['msdrg_high'][x] = (synthetic_value_timeseries_high['msdrg_action'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x])/synthetic_value_timeseries_high['msdrg_baseline'][x]
      delta_admissions['msdrg_noaction'][x] = (synthetic_value_timeseries_high['msdrg_no_action'][x] - synthetic_value_timeseries_high['msdrg_baseline'][x])/synthetic_value_timeseries_high['msdrg_baseline'][x]
      delta_admissions['msdrg_low'][x] = (synthetic_value_timeseries_low['msdrg_action'][x] - synthetic_value_timeseries_low['msdrg_baseline'][x])/synthetic_value_timeseries_low['msdrg_baseline'][x]
    if x < simulation_start:
      if total_census['total_icu_census'][x] > 0.0:
        delta_admissions['icu_obs'][x] = total_census['total_icu_capacity'][-1] - total_census['total_icu_census'][x]
        delta_admissions['icu_high'][x] = total_census['total_icu_capacity'][-1] - total_census['total_icu_census'][x]
        delta_admissions['icu_noaction'][x] = total_census['total_icu_capacity'][-1] - total_census['total_icu_census'][x]
        delta_admissions['icu_low'][x] = total_census['total_icu_capacity'][-1] - total_census['total_icu_census'][x]
      else:
        delta_admissions['icu_obs'][x] = total_census['total_icu_capacity'][-1] - synthetic_value_timeseries_high['icu_action'][x] - covid_icu_high[x]
        delta_admissions['icu_high'][x] = total_census['total_icu_capacity'][-1] - synthetic_value_timeseries_high['icu_action'][x] - covid_icu_high[x]
        delta_admissions['icu_noaction'][x] = total_census['total_icu_capacity'][-1] -  synthetic_value_timeseries_high['icu_action'][x] - covid_icu_high[x]
        delta_admissions['icu_low'][x] = total_census['total_icu_capacity'][-1] -  synthetic_value_timeseries_high['icu_action'][x] - covid_icu_high[x]  
    else:
      delta_admissions['icu_high'][x] = total_census['total_icu_capacity'][-1] - synthetic_value_timeseries_high['icu_action'][x] - covid_icu_high[x]
      delta_admissions['icu_noaction'][x] = total_census['total_icu_capacity'][-1] - synthetic_value_timeseries_high['icu_no_action'][x] - covid_icu_high[x]
      delta_admissions['icu_low'][x] = total_census['total_icu_capacity'][-1] - synthetic_value_timeseries_low['icu_action'][x] - covid_icu_low[x]
    day_of_year += 1
  fig, ax = plt.subplots(3, figsize=(16, 9))
  ax2 = ax[2].twinx()
  ax1 = ax[1].twinx()
  ax0 = ax[0].twinx()
    
  if ani_plot == 2 or ani_plot == 3:

    ax0.plot(date_index, covid_admissions_high, color = 'darkgoldenrod', linewidth = 2.5)
    ax[0].plot(date_index, delta_admissions['admissions_noaction'], color = 'crimson', linewidth = 2.5)

    ax1.plot(date_index, covid_icu_high[:len(date_index)], color = 'darkgoldenrod', linewidth = 2.5)
    ax[1].plot(date_index, delta_admissions['icu_noaction'], color = 'crimson', linewidth = 2.5)

    ax2.plot(date_index, delta_admissions['cum_msdrg_noaction'], color = 'darkgoldenrod', linewidth = 2.5, linestyle = '--') 
    ax[2].plot(date_index,  delta_admissions['msdrg_noaction'], color = 'crimson', linewidth = 2.5)

  if ani_plot == 3:
  
    ax[0].plot(date_index, delta_admissions['admissions_high'], color = 'darkgreen', linewidth = 2.5)
    ax[1].plot(date_index, delta_admissions['icu_high'], color = 'darkgreen', linewidth = 2.5)
    ax[2].plot(date_index, delta_admissions['msdrg_high'], color = 'darkgreen', linewidth = 2.5)
    ax2.plot(date_index, delta_admissions['cum_msdrg_high'], color = 'darkgoldenrod', linewidth = 2.5)
    
  if ani_plot > 0:
    ax[0].plot(date_index, delta_admissions['admissions_low'], color = 'mediumblue', linewidth = 2.5)
    ax[1].plot(date_index, delta_admissions['icu_low'], color = 'mediumblue', linewidth = 2.5)
    ax[2].plot(date_index, delta_admissions['msdrg_low'], color = 'mediumblue', linewidth = 2.5)
    
  if ani_plot == 0:
    ax0.fill_between(date_index[:simulation_start], np.zeros(simulation_start), covid_admissions_low[:simulation_start], alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred',  linewidth = 2.5)
    ax1.fill_between(date_index[:simulation_start], np.zeros(simulation_start), covid_icu_low[:simulation_start], alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred', linewidth = 2.5)
    ax2.fill_between(date_index[:simulation_start], np.zeros(simulation_start), delta_admissions['cum_msdrg_obs'][:simulation_start], alpha = 0.4, color = 'darkgoldenrod')
  else:
    ax0.fill_between(date_index,  np.zeros(len(date_index)), covid_admissions_low, alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred',  linewidth = 2.5)
    ax1.fill_between(date_index, np.zeros(len(date_index)), covid_icu_low[:len(date_index)], alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred', linewidth = 2.5)
    ax2.fill_between(date_index, np.zeros(len(date_index)),  delta_admissions['cum_msdrg_low'], alpha = 0.4, color = 'darkgoldenrod')
  
  
  ax[0].plot(date_index[:simulation_start], delta_admissions['admissions_obs'][:simulation_start], color = 'black', linewidth = 2.5)    
  ax[1].plot(date_index[:simulation_start], delta_admissions['icu_obs'][:simulation_start], color = 'black', linewidth = 2.5)
  ax[2].plot(date_index[:simulation_start], delta_admissions['msdrg_obs'][:simulation_start], color = 'black', linewidth = 2.5)

  ax[0].plot([date_index[0], date_index[-1]], [1.0, 1.0], color = 'black', linewidth = 1.5)
  ax[1].plot([date_index[0], date_index[-1]], [0.0, 0.0], color = 'black', linewidth = 1.5)
  ax[2].plot([date_index[0], date_index[-1]], [0.0, 0.0], color = 'black', linewidth = 1.5)
    
  ax[0].set_xlim([date_index[0], date_index[-1]])
  ax0.set_xlim([date_index[0], date_index[-1]])
  ax[1].set_xlim([date_index[0], date_index[-1]])
  ax1.set_xlim([date_index[0], date_index[-1]])
  ax[2].set_xlim([date_index[0], date_index[-1]])
  ax2.set_xlim([date_index[0], date_index[-1]])

  ax[0].set_ylim([-200, 250])
  ax0.set_ylim([-200, 250])
#  ax[0].set_ylim([0.5, 1.4])
  ax[1].set_ylim([-50, 325])
  ax1.set_ylim([-50, 325])
  ax0.set_xticklabels('')
  ax[1].set_xticklabels('')
  ax1.set_xticklabels('')
  plt.setp(ax[2].get_xticklabels(), fontsize=16)
    
  ax0.set_ylabel('Daily COVID\nAdmissions', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[0].set_ylabel('Daily Admissions\nChange from\n2018-2019 Baseline', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax1.set_ylabel('COVID ICU\nPopulation', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[1].set_ylabel('Available ICU\nCapacity', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[2].set_ylabel('Daily Revenue\nChange (from 2019)', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax2.set_ylabel('Cumulative Annual Revenue\nChange (from 2019)', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[2].yaxis.label.set_color('black')
  ax2.yaxis.label.set_color('darkgoldenrod')
  ax1.yaxis.label.set_color('darkgoldenrod')
  ax[1].yaxis.label.set_color('black')
  ax0.yaxis.label.set_color('darkgoldenrod')
  ax[0].yaxis.label.set_color('black')

  ax[0].set_zorder(ax0.get_zorder()+1) # put ax in front of ax2
  ax[0].patch.set_visible(False) # hide the 'canvas'
  ax[1].set_zorder(ax1.get_zorder()+1) # put ax in front of ax2
  ax[1].patch.set_visible(False) # hide the 'canvas'
  ax[2].set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
  ax[2].patch.set_visible(False) # hide the 'canvas'


  myFmt = mdates.DateFormatter('%b %Y')
  ax[2].xaxis.set_major_formatter(myFmt)    

  ax[2].set_ylim([-0.5, 0.35])
  ax2.set_ylim([-0.15, 0.105])
  ax[2].set_yticks([(float(x) - 5.0)/10.0 for x in range(0, 9)])
  ax[2].set_yticklabels(['{:,.0%}'.format((float(x) - 5.0)/10.0) for x in range(0, 9)])
  ax2.set_yticks([(float(x) - 15.0)/100.0 for x in range(0, 30, 5)])
  ax2.set_yticklabels(['{:,.0%}'.format((float(x) - 15.0)/100.0) for x in range(0, 30, 5)])
#    vals = ax[0].get_yticks()
#    ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])

  props = dict(boxstyle='round', facecolor='beige', alpha=0.8)
  fndct ={'family':'Gill Sans MT', 'fontweight': 'bold'}
  for x in range(0, 3):
    ax[x].fill_between([date_index[17], date_index[59]], [-700, -700], [700, 700], alpha = 0.15, color = 'black') 
    ax[x].fill_between([date_index[simulation_start], date_index[-1]],[-700, -700], [700, 700], alpha = 0.15, color = 'black')
  ax[0].text(0.08, 1.125, 'Elective Procedure\nSuspension', transform=ax[0].transAxes, fontsize=18, ha='center', va='center', bbox=props, fontdict = fndct)
  ax[0].text(0.85, 1.075, 'Simulated Period', transform=ax[0].transAxes, fontsize=18, ha='center', va='center', bbox=props, fontdict = fndct)
  if ani_plot == 2 or ani_plot == 3:
    ax[1].text(0.6, 0.065, 'Patients turned away from ICU', transform=ax[1].transAxes, fontsize=12, ha='center', va='center', bbox=props, fontdict = fndct)
  if ani_plot == 0:
    legend_elements = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 Admissions'), 
                       Line2D([0], [0], color='black', lw = 5, label = 'Non-COVID-19 Admssions, Observed')]
    legend_elements2 = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 ICU Patients'), 
                       Line2D([0], [0], color='black', lw = 5, label = 'Available ICU Beds, Observed')]
    legend_elements3 = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Total Revenue Loss')]

  if ani_plot == 1:
    legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Non-COVID-19 Admssions, Low Transmission'),  
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 Admissions'), 
                       Line2D([0], [0], color='black', lw = 5, label = 'Non-COVID-19 Admssions, Observed')]
    legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Available ICU Beds, Low Transmission'),  
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 ICU Patients'), 
                       Line2D([0], [0], color='black', lw = 5, label = 'Available ICU Beds, Observed')]
    legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Daily Revenue Loss, Low Transmission'),  
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Total Revenue Loss')]

  if ani_plot == 2:
    legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Non-COVID-19 Admssions, Low Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Non-COVID-19 Admssions, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 Admissions'), 
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'COVID-19 Admssions, High'),
                       Line2D([0], [0], color='black', lw = 5, label = 'Non-COVID-19 Admssions, Observed')]
    legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Available ICU Beds, Low Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Available ICU Beds, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 ICU Patients'), 
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'COVID-19 ICU Patients, High'),
                       Line2D([0], [0], color='black', lw = 5, label = 'Available ICU Beds, Observed')]
    legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Daily Revenue Loss, Low Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Daily Revenue Loss, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Total Revenue Loss'),
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, ls='-.', label = 'Total Revenue Loss, High (No Action)')]
  if ani_plot == 3:
    legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Non-COVID-19 Admssions, Low Transmission'),  
                       Line2D([0], [0], color='darkgreen', lw = 5, label = 'Non-COVID-19 Admssions, High Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Non-COVID-19 Admssions, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 Admissions'), 
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'COVID-19 Admssions, High'),
                       Line2D([0], [0], color='black', lw = 5, label = 'Non-COVID-19 Admssions, Observed')]
    legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Available ICU Beds, Low Transmission'),  
                       Line2D([0], [0], color='darkgreen', lw = 5, label = 'Available ICU Beds, High Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Available ICU Beds, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID-19 ICU Patients'), 
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'COVID-19 ICU Patients, High'),
                       Line2D([0], [0], color='black', lw = 5, label = 'Available ICU Beds, Observed')]
    legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Daily Revenue Loss, Low Transmission'),  
                       Line2D([0], [0], color='darkgreen', lw = 5, label = 'Daily Revenue Loss, High Transmission'),  
                       Line2D([0], [0], color='crimson', lw = 5, label = 'Daily Revenue Loss, High (No Action)'),
                       Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Total Revenue Loss'),
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'Total Revenue Losss, High'), 
                       Line2D([0], [0], color='darkgoldenrod', lw = 5, ls='-.', label = 'Total Revenue Loss, High (No Action)')]


  ax[0].legend(handles=legend_elements, loc='upper center', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  ax[1].legend(handles=legend_elements2, loc='upper center', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  ax[2].legend(handles=legend_elements3, loc='upper left', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  plt.tight_layout()

  fig.savefig('manuscript_figures/synthetic_hospital_plots_' + str(ani_plot) + '.png')
  plt.close()
  
def plot_observed_admissions_figures(total_census, admission_series, procedure_type_list, admission_type_list, daily_admission, daily_msdrg, observed_values_timeseries, baseline_msdrg, covid_icu_high, date_index, observation_start_date):
  #this function plots observed and simulated timeseries for hospital admissions, icu census, and total msdrg (revenue)
  #set up figure    
  day_of_year = observation_start_date.timetuple().tm_yday
  total_observed_length = len(observed_values_timeseries['admissions'])
  date_index = date_index[:total_observed_length]
  delta_admissions = {}
  date_index_alt1 = []
  date_index_alt2 = []
  delta_admissions['admissions_e'] = np.zeros(total_observed_length)
  delta_admissions['admissions_s'] = np.zeros(total_observed_length)
  delta_admissions['msdrg_obs'] = np.zeros(total_observed_length)
  delta_admissions['cum_msdrg_obs'] = np.zeros(total_observed_length)
  delta_admissions['available_icu'] = []
  delta_admissions['covid_icu_observed'] = []
  delta_admissions['covid_icu_predicted'] = np.zeros(total_observed_length)
  for x in range(0, total_observed_length):
    total_e_baseline = 0.0
    total_s_baseline = 0.0
    total_e_obs = 0.0
    total_s_obs = 0.0
    if day_of_year + x >= 365:
      current_doy = day_of_year + x - 365
    else:
      current_doy = day_of_year + x
    for proc in procedure_type_list:
      for adm in admission_type_list:
        cat = proc + '_' + adm
        
        if adm == 'EI':
          total_e_baseline += daily_admission[cat][current_doy]
          total_e_obs += admission_series[cat][x]
        elif adm == 'IP':
          total_s_baseline += daily_admission[cat][current_doy]
          total_s_obs += admission_series[cat][x]
    delta_admissions['admissions_e'][x] = float(total_e_obs) / float(total_e_baseline)
    delta_admissions['admissions_s'][x] = float(total_s_obs) / float(total_s_baseline)
    
    delta_admissions['msdrg_obs'][x] = (observed_values_timeseries['msdrg'][x] - baseline_msdrg[x]) / baseline_msdrg[x]
    delta_admissions['cum_msdrg_obs'][x] = (np.sum(observed_values_timeseries['msdrg'][:x]) - np.sum(baseline_msdrg[:x]))/np.sum(baseline_msdrg[:365])
    if total_census['total_icu_census'][x] > 0.0:
      delta_admissions['available_icu'].append(total_census['total_icu_capacity'][-1] - total_census['total_icu_census'][x])
      date_index_alt1.append(date_index[x])
    if x > 89:
      delta_admissions['covid_icu_observed'].append(total_census['covid_icu_census'][x])
      date_index_alt2.append(date_index[x])
    delta_admissions['covid_icu_predicted'][x] = covid_icu_high[x]
  fig, ax = plt.subplots(3, figsize=(16, 9))
  ax2 = ax[2].twinx()
  ax1 = ax[1].twinx()
  ax[0].fill_between(date_index, delta_admissions['admissions_s'], color = 'steelblue', alpha = 0.8, linewidth = 2.0)
  ax[0].fill_between(date_index, delta_admissions['admissions_e'], color = 'forestgreen', alpha = 0.6, linewidth = 2.0)
  ax1.plot(date_index_alt2, delta_admissions['covid_icu_observed'], color = 'crimson', linewidth = 3.0)
  ax[1].plot(date_index_alt1, delta_admissions['available_icu'], color = 'black', linewidth = 3.0)
  ax1.fill_between(date_index, delta_admissions['covid_icu_predicted'], color = 'darkgoldenrod', alpha = 0.4, linewidth = 2.5)
  ax2.fill_between(date_index, delta_admissions['cum_msdrg_obs'], color = 'darkgoldenrod', alpha = 0.4, linewidth = 2.5)
  ax[2].plot(date_index, delta_admissions['msdrg_obs'], color = 'black', linewidth = 3.0)
  
  ax[0].plot([date_index[0], date_index[-1]], [1.0, 1.0], color = 'black', linewidth = 1.5)
  ax[1].plot([date_index[0], date_index[-1]], [0.0, 0.0], color = 'black', linewidth = 1.5)
  ax[2].plot([date_index[0], date_index[-1]], [0.0, 0.0], color = 'black', linewidth = 1.5)
    
  ax[0].set_xlim([date_index[0], date_index[-1]])
  ax[1].set_xlim([date_index[0], date_index[-1]])
  ax1.set_xlim([date_index[0], date_index[-1]])
  ax[2].set_xlim([date_index[0], date_index[-1]])
  ax2.set_xlim([date_index[0], date_index[-1]])

  ax[0].set_ylim([0.5, 1.4])
  ax[1].set_ylim([-50, 325])
  ax1.set_ylim([-50, 325])
  ax[1].set_xticklabels('')
  ax1.set_xticklabels('')
  plt.setp(ax[2].get_xticklabels(), fontsize=16)
    
  ax[0].set_ylabel('Daily Admissions\nChange from\n2018-2019 Baseline', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax1.set_ylabel('COVID ICU\nPopulation', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[1].set_ylabel('Available ICU\nCapacity', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[2].set_ylabel('Daily Revenue\nChange (from 2019)', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax2.set_ylabel('Cumulative Annual Revenue\nChange (from 2019)', fontname = 'Gill Sans MT', fontsize=18, fontweight='bold')
  ax[2].yaxis.label.set_color('black')
  ax2.yaxis.label.set_color('darkgoldenrod')
  ax1.yaxis.label.set_color('darkgoldenrod')
  ax[1].yaxis.label.set_color('black')
  ax[0].yaxis.label.set_color('black')

  ax[0].patch.set_visible(False) # hide the 'canvas'
  ax[1].set_zorder(ax1.get_zorder()+1) # put ax in front of ax2
  ax[1].patch.set_visible(False) # hide the 'canvas'
  ax[2].set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
  ax[2].patch.set_visible(False) # hide the 'canvas'


  myFmt = mdates.DateFormatter('%b %Y')
  ax[2].xaxis.set_major_formatter(myFmt)    

  ax[2].set_ylim([-0.5, 0.35])
  ax2.set_ylim([-0.10, 0.07])
  ax[2].set_yticks([(float(x) - 5.0)/10.0 for x in range(0, 9)])
  ax[2].set_yticklabels(['{:,.0%}'.format((float(x) - 5.0)/10.0) for x in range(0, 9)])
  ax2.set_yticks([(float(x) - 5.0)/50.0 for x in range(0, 9)])
  ax2.set_yticklabels(['{:,.0%}'.format((float(x) - 5.0)/50.0) for x in range(0, 9)])
  ax[0].set_yticks([(float(x))/10.0 for x in range(6, 15, 2)])
  ax[0].set_yticklabels(['{:,.0%}'.format((float(x))/10.0) for x in range(6, 15, 2)])
#    vals = ax[0].get_yticks()
#    ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])

  props = dict(boxstyle='round', facecolor='beige', alpha=0.8)
  fndct ={'family':'Gill Sans MT', 'fontweight': 'bold'}
  for x in range(0, 3):
    ax[x].fill_between([date_index[17], date_index[59]], [-700, -700], [700, 700], alpha = 0.15, color = 'black') 
  ax[0].text(0.08, 1.125, 'Elective Procedure\nSuspension', transform=ax[0].transAxes, fontsize=18, ha='center', va='center', bbox=props, fontdict = fndct)
  legend_elements = [Patch(facecolor = 'steelblue', edgecolor = 'black', alpha = 0.8, label = 'Scheduled Admissions'),
                     Patch(facecolor = 'forestgreen', edgecolor = 'black', alpha = 0.8, label = 'Emergency Department Admissions')]
  legend_elements2 = [Line2D([0], [0], color='crimson', lw = 5, label = 'Observed COVID-19 ICU Patients'),  
                     Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.4, label = 'Estimated COVID-19 ICU Patients'), 
                     Line2D([0], [0], color='black', lw = 5, label = 'Available ICU Beds, Observed')]
  legend_elements3 = [Line2D([0], [0], color='black', lw = 5, label = 'Daily Revenue Change'),  
                     Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Cumulative Revenue Change')]


  ax[0].legend(handles=legend_elements, loc='upper right', ncol = 1, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  ax[1].legend(handles=legend_elements2, loc='upper center', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  ax[2].legend(handles=legend_elements3, loc='upper left', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':12})
  plt.tight_layout()

  fig.savefig('manuscript_figures/synthetic_observed_hospital_plots_' + '.png')
  plt.close()
  
  
def plot_other_admissions():
  for plot_num in range(0, 3):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('lightslategray')
    ax0 = ax.twinx()
    ax0.set_facecolor('lightslategray')
    if plot_num == 0:
      ax0.fill_between(date_index, np.zeros(len(date_index)), ensemble_dict_admit_low[str(3)], alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred',  linewidth = 2.5)
      if ani_plot > 0:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_er_low[str(3)][start_simulation:], color = 'mediumblue', linewidth = 4)
      if ani_plot == 2 or ani_plot == 3:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_admit_high[str(3)][start_simulation:], color = 'darkgoldenrod', linewidth = 4)
        ax0.plot(date_index['total'][start_simulation:], ensemble_dict_er_na[str(3)][start_simulation:], color = 'crimson', linewidth = 4)
      if ani_plot == 3:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_er_high[str(3)][start_simulation:], color = 'darkgreen', linewidth = 4)
    if plot_num == 1:
      ax0.fill_between(date_index['total'], np.zeros(len(date_index['total'])), ensemble_dict_tot_icu_low[str(3)], alpha = 0.4, facecolor = 'darkgoldenrod', edgecolor = 'indianred', linewidth = 4)
      if ani_plot > 0:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_low['3'][start_simulation:], color = 'mediumblue', linewidth = 4)
      if ani_plot == 2 or ani_plot == 3:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_tot_icu_high[str(3)][start_simulation:], color = 'darkgoldenrod', linewidth = 4)
        ax0.plot(date_index['total'][start_simulation:], ensemble_dict_na['3'][start_simulation:], color = 'crimson', linewidth = 4)
      if ani_plot == 3:
        ax.plot(date_index['total'][start_simulation:], ensemble_dict_high['3'][start_simulation:], color = 'darkgreen', linewidth = 4)
    if plot_num == 2:
      ax0.fill_between(date_index['total'], np.zeros(len(date_index['total'])), ensemble_cum_msdrg_low[str(3)], alpha = 0.4, color = 'darkgoldenrod')
      if ani_plot > 0:
        ax.plot(date_index['total'][start_simulation:], ensemble_msdrg_low['3'][start_simulation:], color = 'mediumblue', linewidth = 4)
      if ani_plot == 2 or ani_plot == 3:
        ax0.plot(date_index['total'][start_simulation:], ensemble_cum_msdrg_na[str(3)][start_simulation:], color = 'darkgoldenrod', linewidth = 4, linestyle = '--') 
        ax.plot(date_index['total'][start_simulation:], ensemble_msdrg_na['3'][start_simulation:], color = 'crimson', linewidth = 4)
      if ani_plot == 3:
        ax.plot(date_index['total'][start_simulation:], ensemble_msdrg_high['3'][start_simulation:], color = 'darkgreen', linewidth = 4)
        ax0.plot(date_index['total'][start_simulation:], ensemble_cum_msdrg_high[str(3)][start_simulation:], color = 'darkgoldenrod', linewidth = 4)
        
    ax.plot([date_index['total'][0], date_index['total'][-1]], [0.0, 0.0], color = 'black', linewidth = 2.5)
    
    ax.set_xlim([date_index['total'][start_simulation], date_index['total'][-1]])
    ax0.set_xlim([date_index['total'][start_simulation], date_index['total'][-1]])
    if plot_num == 0:
      ax.set_ylim([-225, 150])
      ax0.set_ylim([-225, 150])
      ax.set_ylabel('Daily Admissions Change from 2018-2019 Baseline', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')
      ax0.set_ylabel('Daily COVID Admissions', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')

    elif plot_num == 1:
      ax.set_ylim([-25, 250])
      ax0.set_ylim([-25, 250])
      ax0.set_ylabel('COVID ICU Population', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')
      ax.set_ylabel('Available ICU Capacity', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')
      if ani_plot == 2 or ani_plot == 3:
        ax.text(0.7, 0.065, 'Patients turned away from ICU', transform=ax.transAxes, fontsize=24, ha='center', va='center', bbox=props, fontdict = fndct)
    elif plot_num == 2:
      ax.set_ylabel('Daily Revenue Change (from 2019)', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')
      ax0.set_ylabel('Cumulative Annual Revenue Change (from 2019)', fontname = 'Gill Sans MT', fontsize=24, fontweight='bold')
      ax.set_ylim([-0.5, 0.35])
      ax0.set_ylim([-0.10, 0.07])
      ax.set_yticks([(float(x) - 5.0)/10.0 for x in range(0, 9)])
      ax.set_yticklabels(['{:,.0%}'.format((float(x) - 5.0)/10.0) for x in range(0, 9)])
      ax0.set_yticks([(float(x) - 5.0)/50.0 for x in range(0, 9)])
      ax0.set_yticklabels(['{:,.0%}'.format((float(x) - 5.0)/50.0) for x in range(0, 9)])
    plt.setp(ax.get_xticklabels(), fontsize=16)
    
    ax.yaxis.label.set_color('black')
    ax0.yaxis.label.set_color('darkgoldenrod')

    ax.set_zorder(ax0.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False) # hide the 'canvas'

    myFmt = mdates.DateFormatter('%b %Y')
    ax.xaxis.set_major_formatter(myFmt)    
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth = range(1,13)))

    props = dict(boxstyle='round', facecolor='beige', alpha=0.8)
    fndct ={'family':'Gill Sans MT', 'fontweight': 'bold'}
    if ani_plot == 0:
      legend_elements = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID Admissions')]
      legend_elements2 = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID ICU')]
      legend_elements3 = [Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Revenue Loss (Cumulative)')]

    if ani_plot == 1:
      legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID Admissions')]
      legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID ICU')]
      legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Revenue Loss (Cumulative)')]

    if ani_plot == 2:
      legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspensions'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID Admissions'), 
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'High Transmission')]
      legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspension'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID ICU'), 
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'High Tranmission')]
      legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspension'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Revenue Loss (Cumulative)'),
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, ls='-.', label = 'High, No Suspension')]
    if ani_plot == 3:
      legend_elements = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Tranmission'),  
                         Line2D([0], [0], color='darkgreen', lw = 5, label = 'High, w/ Suspension'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspension'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID Admissions'), 
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'High Transmission')]
      legend_elements2 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Line2D([0], [0], color='darkgreen', lw = 5, label = 'High w/ Suspension'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspension'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'COVID ICU'), 
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'High Transmission')]
      legend_elements3 = [Line2D([0], [0], color='mediumblue', lw = 5, label = 'Low Transmission'),  
                         Line2D([0], [0], color='darkgreen', lw = 5, label = 'High w/ Suspension'),  
                         Line2D([0], [0], color='crimson', lw = 5, label = 'High, No Suspension'),
                         Patch(facecolor = 'darkgoldenrod', edgecolor = 'black', alpha = 0.8, label = 'Revenue Loss (Cumulative)'),
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, label = 'High w/ Suspension'), 
                         Line2D([0], [0], color='darkgoldenrod', lw = 5, ls='-.', label = 'High, No Suspension')]
    if plot_num == 0:
      ax.legend(handles=legend_elements, loc='upper right', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':24})
    if plot_num == 1:
      ax.legend(handles=legend_elements2, loc='upper right', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':24})
    if plot_num == 2:
      ax.legend(handles=legend_elements3, loc='upper right', ncol = 2, prop={'family':'Gill Sans MT','weight':'bold','size':24})
    for item in (ax.get_xticklabels()):
      item.set_fontsize(24)
      item.set_fontname('Gill Sans MT')  
    for item in (ax.get_yticklabels()):
      item.set_fontsize(24)
      item.set_fontname('Arial')  
    for item in (ax0.get_yticklabels()):
      item.set_fontsize(24)
      item.set_fontname('Arial')  
    plt.tight_layout()

    fig.savefig('admissions_icu_revenue_' + alt_label_transmission + '_' + alt_label_admissions + '_' + str(ani_plot) + '_' +  str(plot_num) + '.png')
    plt.close()


  print('admissions_icu_revenue_' + alt_label_transmission + '_' + alt_label_admissions + '.csv')


    
#    ax[0].fill_between(admission_complete_list, np.zeros(len(admission_complete_list)), admission_complete['MED_IP'] + admission_complete['SURG_IP'], alpha = 0.6, facecolor = 'steelblue', edgecolor = 'black')
#    ax[0].fill_between(admission_complete_list, np.zeros(len(admission_complete_list)), admission_complete['MED_EI'] + admission_complete['SURG_EI'], alpha = 0.6, facecolor = 'teal', edgecolor = 'black')

    #for z in range(0,gradations):
      #alpha_val = np.power(np.power(z - (gradations-1)/2,2), 0.5)/ ( (gradations - 1) /2 )
#      alpha_val = alpha_list[z]
#      ax1.fill_between(date_index['total'], ensemble_dict[str(z)], ensemble_dict[str(z+1)], alpha = 1 - alpha_val, color = 'teal', linewidth = 2.0)
 
 
#    for z in range(1,gradations-1): 
#      alpha_val = alpha_list[z]
#      ax[1].fill_between(date_index['total'], ensemble_dict_tot_icu[str(z)], ensemble_dict_tot_icu[str(z+1)], alpha = 1 - alpha_val, color = 'indianred', linewidth = 0.5)

    #for z in range(1,gradations-1):
      #alpha_val = alpha_list[z]
      #ax[2].fill_between(date_index['total'], ensemble_msdrg[str(z)], ensemble_msdrg[str(z+1)], alpha = 1 - alpha_val, color = 'forestgreen', linewidth = 2.0)

#    for z in range(1,gradations-1):
#      alpha_val = alpha_list[z]
#      ax[0].fill_between(date_index['total'], ensemble_dict_admit[str(z)], ensemble_dict_admit[str(z+1)], alpha = 1 - alpha_val, color = 'indianred', linewidth = 2.0)

#    for z in range(0,gradations):
#      alpha_val = alpha_list[z]
#      ax[0].fill_between(date_index['total'], ensemble_dict_er[str(z)], ensemble_dict_er[str(z+1)], alpha = 1 - alpha_val, color = 'teal', linewidth = 2.0)

#    for z in range(0,gradations):
#      alpha_val = alpha_list[z]
#      ax[0].fill_between(date_index['total'], ensemble_dict_ip[str(z)], ensemble_dict_ip[str(z+1)], alpha = 1 - alpha_val, color = 'steelblue', linewidth = 2.0)
#    for z in range(0,gradations):
#      alpha_val = alpha_list[z]
#      ax[2].fill_between(date_index['total'], ensemble_cum_msdrg[str(z)], ensemble_cum_msdrg[str(z+1)], alpha = 1 - alpha_val, color = 'darkgoldenrod', linewidth = 2.0)

#  historical_dates=np.asarray(date_index['total'][:len(total_census['icu_covid'])])
#  plotting_dates= total_census['icu_covid'] > 0.0


def plot_icu_calibration_estimates(constants, total_census, total_census_project):

  estimations = np.zeros(288)
  residuals = np.zeros(288)
  estimations_alt1 = np.zeros(288)
  estimations_alt2 = np.zeros(288)
  residuals_alt1 = np.zeros(288)
  residuals_alt2 = np.zeros(288)

  non_covid_hospital = total_census['icu_total'] - total_census['icu_covid']
  non_covid_hospital_project = total_census_project['icu_total'] - total_census_project['icu_covid']
  alt_1_multiplier = total_census_project['icu_capacity'][-1]/total_census['icu_capacity'][-1]
  alt_2_multiplier = (np.mean(non_covid_hospital_project[190:210]) - np.mean(non_covid_hospital_project[40:50]))/(np.mean(non_covid_hospital[190:210]) - np.mean(non_covid_hospital[40:50]))
  for x in range(0, 51):
    estimations += A[:,x] * constants[x]
    estimations_alt1 += A[:,x] * constants[x] * alt_1_multiplier
    estimations_alt2 += A[:,x] * constants[x] * alt_2_multiplier
  for x in range(0, 288):
    residuals[x] = estimations[x] - values_alt[x]
    residuals_alt1[x] = estimations_alt1[x] - values_alt[x]
    residuals_alt2[x] = estimations_alt2[x] - values_alt[x]

  ar_estimates, ar_residuals, ar_coef = make_ar_model(3, residuals)
  ar_estimates_alt1, ar_residuals_alt1, ar_coef_alt1 = make_ar_model(3, residuals_alt1)
  ar_estimates_alt2, ar_residuals_alt2, ar_coef_alt2 = make_ar_model(3, residuals_alt2)
  num_realizations = 500
  observed_length = len(admission_type_series['IP_01'])
  icu_range = np.zeros((observed_length, num_realizations))
  icu_range_alt1 = np.zeros((observed_length, num_realizations))
  icu_range_alt2 = np.zeros((observed_length, num_realizations))
  start_value = datetime.strptime('03-01-2020', '%m-%d-%Y')
  for realn in range(0, num_realizations):
    date_index = []
    estimation = np.zeros(observed_length)
    estimation_alt1 = np.zeros(observed_length)
    estimation_alt2 = np.zeros(observed_length)
    ar_error = np.zeros(observed_length)
    ar_error_alt1 = np.zeros(observed_length)
    ar_error_alt2 = np.zeros(observed_length)
    for x in range(0, observed_length):
      for mdc_num in range(1, 26):
        baseline_ip = np.mean(admission_type_series['IP_' + str(mdc_num).zfill(2)][190:210])
        baseline_ei = np.mean(admission_type_series['EI_' + str(mdc_num).zfill(2)][190:210])
        estimation[x] += (baseline_ip - admission_type_series['IP_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2]
        estimation[x] += (baseline_ei - admission_type_series['EI_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2 + 1]
        estimation_alt1[x] += (baseline_ip - admission_type_series['IP_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2] * alt_1_multiplier
        estimation_alt1[x] += (baseline_ei - admission_type_series['EI_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2 + 1] * alt_1_multiplier
        estimation_alt2[x] += (baseline_ip - admission_type_series['IP_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2] * alt_2_multiplier
        estimation_alt2[x] += (baseline_ei - admission_type_series['EI_' + str(mdc_num).zfill(2)][x]) * constants[(mdc_num-1)*2 + 1] * alt_2_multiplier

      estimation[x] += constants[-1]
      estimation_alt1[x] += constants[-1] * alt_1_multiplier
      estimation_alt2[x] += constants[-1] * alt_2_multiplier
      date_index.append(start_value + timedelta(x))
      random_error_int = np.random.randint(288)
      random_error = ar_residuals[random_error_int]
      random_error_alt1 = ar_residuals_alt1[random_error_int]
      random_error_alt2 = ar_residuals_alt2[random_error_int]
      if x < 3:
        for xx in range(0, x):
          ar_error[x] += ar_coef[xx] * ar_error[x-xx-1]
          ar_error_alt1[x] += ar_coef_alt1[xx] * ar_error_alt1[x-xx-1]
          ar_error_alt2[x] += ar_coef_alt2[xx] * ar_error_alt2[x-xx-1]
        ar_error[x] += random_error
        ar_error_alt1[x] += random_error_alt1
        ar_error_alt2[x] += random_error_alt2
      else:
        for xx in range(0, 3):
          ar_error[x] += ar_coef[xx] * ar_error[x-xx-1]
          ar_error_alt1[x] += ar_coef_alt1[xx] * ar_error_alt1[x-xx-1]
          ar_error_alt2[x] += ar_coef_alt2[xx] * ar_error_alt2[x-xx-1]
        ar_error[x] += random_error
        ar_error_alt1[x] += random_error_alt1
        ar_error_alt2[x] += random_error_alt2
      
      icu_range[x, realn] = np.mean(non_covid_hospital[190:210]) - (estimation[x] - ar_error[x])
      icu_range_alt1[x, realn] = np.mean(non_covid_hospital_project[190:210]) - (estimation_alt1[x] - ar_error_alt1[x])
      icu_range_alt2[x, realn] = np.mean(non_covid_hospital_project[190:210]) - (estimation_alt2[x] - ar_error_alt2[x])
  
  gradations = 7
  ensemble_icu = discritize_distribution(icu_range, 325, num_realizations, gradations)
  ensemble_icu_alt1 = discritize_distribution(icu_range_alt1, 325, num_realizations, gradations)
  ensemble_icu_alt2 = discritize_distribution(icu_range_alt2, 325, num_realizations, gradations)
  colorsp = sns.color_palette('rocket', 3)  
  prediction_fill = {}
  for x, ens in zip(['UNC', 'Trianglev1', 'Trianglev2'], [ensemble_icu, ensemble_icu_alt1, ensemble_icu_alt2]):
    prediction_fill[x] = np.zeros(gradations + 1)
    for xx in range(37, 325):
      for z in range(0, gradations + 1):
        if x == 'UNC':
          if non_covid_hospital[xx] > ens['0'][xx] and non_covid_hospital[xx] < ens[str(z)][xx]:
            prediction_fill[x][z] += 1.0/(288.0)
        else:
          if non_covid_hospital_project[xx] > ens['0'][xx] and non_covid_hospital_project[xx] < ens[str(z)][xx]:
            prediction_fill[x][z] += 1.0/(288.0)
        
  cdf_locs = np.zeros(gradations + 1)
  for x in range(1, gradations + 1):
    cdf_locs[x] = float(x)/float(gradations)


def plot_icu_capacities():
  icu_cap = pd.read_csv('regional_icu_capacity_timeseries.csv')
  icu_census = pd.read_csv('regional_icu_timeseries.csv')
  icu_covid = pd.read_csv('regional_cov_icu_timeseries.csv')
  icu_covid = np.asarray(icu_covid['0'])
  icu_census = np.asarray(icu_census['0'])
  icu_cap = np.asarray(icu_cap['0'])
  fig, ax = plt.subplots()
  start_date = datetime(2020, 3, 1, 0, 0, 0)
  first_date = start_date + timedelta(100)
  end_date = start_date + timedelta(317)
  date_list = []
  for x in range(100, 318):
    date_list.append(start_date + timedelta(x))
  ax.plot([first_date, end_date], np.ones(2) * icu_cap[len(icu_cap)-1], linewidth = 10.0, color = 'black')
  ax.fill_between(date_list, np.zeros(len(date_list)), icu_covid[100:318], facecolor = 'goldenrod', edgecolor = 'black', linewidth = 1.0)
  ax.fill_between(date_list, icu_covid[100:318], icu_census[100:318], facecolor = 'lightslategray', edgecolor = 'black', linewidth = 1.0)
  ax.set_xlim([first_date, end_date])
  ax.set_ylim([0.0, icu_cap[len(icu_cap)-1] + 1])
  ax.set_ylabel('Regional ICU Census', fontsize = 28, weight = 'bold', fontname = 'Gill Sans MT')
  ax.set_xticks([datetime(2020, 7, 1, 0, 0, 0), datetime(2020, 7, 1, 0, 0, 0),datetime(2020, 8, 1, 0, 0, 0),datetime(2020, 9, 1, 0, 0, 0),datetime(2020, 10, 1, 0, 0, 0), datetime(2020, 11, 1, 0, 0, 0), datetime(2020, 12, 1, 0, 0, 0), datetime(2021, 1, 1, 0, 0, 0)])
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  legend_elements = []
  legend_elements = [Line2D([0], [0], color='black', lw = 4, label='Regional ICU Capacity')]
  legend_elements.append(Patch(facecolor='lightslategray', edgecolor='black', label='Non-COVID ICU Census', alpha = 1.0))
  legend_elements.append(Patch(facecolor='goldenrod', edgecolor='black', label='COVID ICU Census', alpha = 1.0))
  ax.legend(handles=legend_elements, loc='center left', prop={'family':'Gill Sans MT','weight':'bold','size':26})
  for item in (ax.get_xticklabels()):
    item.set_fontsize(22)
    item.set_fontname('Gill Sans MT')  
  for item in (ax.get_yticklabels()):
    item.set_fontsize(22)
    item.set_fontname('Gill Sans MT')  
  plt.show()

