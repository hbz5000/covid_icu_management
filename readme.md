
# Protecting Public Health and Preserving the Financial Viability of North Carolina’s Critical Health Care Facilities during Infectious Disease Outbreaks

**Evaluating the impact of hospital management measures on pandemic operations and hospital financial health**

Harrison Zeff<sup>1\*</sup>, Nicholas DeFelice<sup>2</sup>, Yufei Su<sup>1</sup>, Bethany Percha<sup>2</sup>, Stephen Teitelbaum<sup>1</sup>, Laura McGuinn<sup>2</sup>, and Gregory W. Characklis<sup>1</sup>,

<sup>1 </sup> Center on Financial Risk in Environmental Systems, University of North Carolina at Chapel Hill
<sup>2 </sup> Department of Environmental Medicine and Public Health, Icahn School of Medicine at Mount Sinai
<sup>3 </sup> Mount Sinai Health System


\* corresponding author:  hbz5000@gmail.com

## Abstract
While hospitals’ primary emphasis during the COVID-19 pandemic has been on ensuring sufficient health-related resource capacity (e.g., ICU beds, ventilators) to serve admitted patients, the impacts of the pandemic on the financial viability of hospitals has also become a critical concern. Data from the period March 2020-Janaury 2021 suggest that the halt to elective and non-emergency inpatient procedures, combined with a reduction in emergency room procedures, led to losses equal to 6% of revenue from inpatient procedures, or about $765 million. This study finds that societal measures to reduce the community transmission rates have a larger impact on available healthcare capacity and hospital financial losses than hospital-level decisions. This study illustrates the tradeoffs between hospital capacity, quality of care, and financial risk faced by health care facilities throughout the U.S. as a result of COVID-19, providing potential insights for many hospitals seeking to navigate these uncertain scenarios through adaptive decision-making.

## Reproduce the experiment

1. Install python library dependencies, inluding
   a. pandas
   b. datetime
   c. numpy
   d. geopandas
   e. scipy
   f. matplotlib
   g. seaborn
   
2. Download this github repo and set as working directory
3. Run the following scripts to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `hospital_simulations.py` | Script to execture the hospital simulation model

4. The script will use data in the folders oxygen_use_data and state_hospital_data
5. The script will create the folder covid_timeseries_agg, which has the relevant data aggregated to the study area (Research Triangle) and the folders hospital_admissions and manuscript_figures, which contain summary output figures
