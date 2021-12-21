# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from datetime import timedelta


# %%
### Read Input Data
cgm_df = pd.read_csv("./data/CGMData.csv", low_memory=False)
insulin_df = pd.read_csv("./data/InsulinData.csv", low_memory=False)


# %%
### Date_Time_Stamp insertion for data segmentation purpose
cgm_df.insert(3, "date_time_stamp", pd.to_datetime(cgm_df['Date'] + ' ' + cgm_df['Time']))
cgm_df['Date'] = pd.to_datetime(cgm_df['Date'])
cp_df = cgm_df[['Date', 'Time', 'date_time_stamp', 'Sensor Glucose (mg/dL)']]   # Filter only the needed fields to cp_df dataframe


# %%
### Split MANUAL and AUTO MODE
auto_mode_on_rows = insulin_df.loc[insulin_df['Alarm'] == "AUTO MODE ACTIVE PLGM OFF", ['Date', 'Time']]

# Consider only the earliest AUTO MODE ACTIVE PLGM OFF Signal
auto_mode_on_rows = pd.to_datetime(auto_mode_on_rows['Date'] + ' ' + auto_mode_on_rows['Time'])
auto_mode_on_timestamp = min(auto_mode_on_rows) # Earliest
cp_df = cp_df.set_index(['date_time_stamp'])
cp_df.sort_index(inplace=True)
manual_df = cp_df[:auto_mode_on_timestamp]  # DataFrame for MANUAL MODE
auto_df = cp_df[auto_mode_on_timestamp:]    # DataFrame for AUTO MODE
dfs = {}    # Dict of manual and auto mode dataframes
dfs['Manual Mode'] = manual_df
dfs['Auto Mode'] = auto_df


# %%

w_days = ['Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', 'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', 'Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)']

o_night = ['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', 'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', 'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)']

day_time = ['Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', 'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', 'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)']

# Construct a result dataframe with headers and indices
result_df = pd.DataFrame(columns=o_night+day_time+w_days, index=['Manual Mode', 'Auto Mode'])

for mode, df in dfs.items():
    # Loop for Manual and Auto Modes
    g = df.groupby('Date')
    temp_mode_df = pd.DataFrame(columns=result_df.columns)  # Temporary DF to store the in_range percentage of all days in mode

    for day, days_df in g:
        # Loop for each day in a particular mode
        temp_day_df = pd.DataFrame(columns=temp_mode_df.columns)    # Temporary dataframe to hold values of each day
        day_df = days_df

        overall_count = 288 # Assumed to be constant in-order to overcome any imbalance in data

        # From MidNight 12:00 AM to 06:00 AM
        night_time_df = day_df[:day+timedelta(hours=6)]
        if len(night_time_df)>0:

            # Aggregating SUM and COUNT of values in range 
            night_time_hyperglycemia = night_time_df.loc[night_time_df['Sensor Glucose (mg/dL)'] > 180].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            night_time_hyperglycemia_critical = night_time_df.loc[night_time_df['Sensor Glucose (mg/dL)'] > 250].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            night_time_range = night_time_df[night_time_df['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            night_time_range_secondary = night_time_df[night_time_df['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            night_time_hypoglycemia_level_1 = night_time_df.loc[night_time_df['Sensor Glucose (mg/dL)'] < 70].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            night_time_hypoglycemia_level_2 = night_time_df.loc[night_time_df['Sensor Glucose (mg/dL)'] < 54].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})

            # Calculating the percentage of values in range
            temp_day_df.at[0, 'Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (100 * (night_time_hyperglycemia.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (100 * (night_time_hyperglycemia_critical.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (100 * (night_time_range.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (100 * (night_time_range_secondary.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (100 * (night_time_hypoglycemia_level_1.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (100 * (night_time_hypoglycemia_level_2.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)

        # From 06:00 AM to 12:00 AM
        day_time_df = day_df[day+timedelta(hours=6):]
        if len(day_time_df) > 0:

            # Aggregating SUM and COUNT of values in range 
            day_time_hyperglycemia = day_time_df.loc[day_time_df['Sensor Glucose (mg/dL)'] > 180].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            day_time_hyperglycemia_critical = day_time_df.loc[day_time_df['Sensor Glucose (mg/dL)'] > 250].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            day_time_range = day_time_df[day_time_df['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            day_time_range_secondary = day_time_df[day_time_df['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            day_time_hypoglycemia_level_1 = day_time_df.loc[day_time_df['Sensor Glucose (mg/dL)'] < 70].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
            day_time_hypoglycemia_level_2 = day_time_df.loc[day_time_df['Sensor Glucose (mg/dL)'] < 54].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})

            # Calculating the percentage of values in range
            temp_day_df.at[0, 'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (100 * (day_time_hyperglycemia.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (100 * (day_time_hyperglycemia_critical.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (100 * (day_time_range.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (100 * (day_time_range_secondary.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (100 * (day_time_hypoglycemia_level_1.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
            temp_day_df.at[0, 'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (100 * (day_time_hypoglycemia_level_2.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)

        # whole_day_overall = day_df.agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})

        # Aggregating SUM and COUNT of values in range 
        whole_day_hyperglycemia = day_df.loc[day_df['Sensor Glucose (mg/dL)'] > 180].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
        whole_day_hyperglycemia_critical = day_df.loc[day_df['Sensor Glucose (mg/dL)'] > 250].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
        whole_day_range = day_df[day_df['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
        whole_day_range_secondary = day_df[day_df['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
        whole_day_hypoglycemia_level_1 = day_df.loc[day_df['Sensor Glucose (mg/dL)'] < 70].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})
        whole_day_hypoglycemia_level_2 = day_df.loc[day_df['Sensor Glucose (mg/dL)'] < 54].agg({'Sensor Glucose (mg/dL)': ['sum', 'count']})

        # Calculating the percentage of values in range
        temp_day_df.at[0, 'Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = (100 * (whole_day_hyperglycemia.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
        temp_day_df.at[0, 'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = (100 * (whole_day_hyperglycemia_critical.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
        temp_day_df.at[0, 'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = (100 * (whole_day_range.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
        temp_day_df.at[0, 'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = (100 * (whole_day_range_secondary.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
        temp_day_df.at[0, 'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = (100 * (whole_day_hypoglycemia_level_1.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)
        temp_day_df.at[0, 'Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = (100 * (whole_day_hypoglycemia_level_2.loc['count', 'Sensor Glucose (mg/dL)']) / overall_count)

        # Append the day in-range percentage values to temp_mode_df
        temp_mode_df = pd.concat([temp_mode_df, temp_day_df], ignore_index=True)
        
    # Calculate the mean of each column for a mode df that contains in-range percentage values for every day and add to the final result_df
    result_df.loc[mode] = temp_mode_df.mean()

# result_df.to_csv('Results.csv', index=False, header=False)  # Strip dataframe of headers and indices for Results.csv      
result_df.to_csv('./Results/Performance_Metrics_Results.csv')




