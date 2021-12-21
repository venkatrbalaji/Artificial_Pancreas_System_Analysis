# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from math import ceil
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from mlxtend.frequent_patterns import apriori, association_rules


# %%
def extract_meal_and_no_meal_instances(cgm_data_file, insulin_data_file):
    if ".xls" in cgm_data_file:
        cgm_df = pd.read_excel(cgm_data_file, parse_dates=[['Date', 'Time']])
    elif ".csv" in cgm_data_file:
        cgm_df = pd.read_csv(cgm_data_file, parse_dates=[
                             ['Date', 'Time']], low_memory=False)

    if ".xls" in insulin_data_file:
        insulin_df = pd.read_excel(
            insulin_data_file, parse_dates=[['Date', 'Time']])
    elif ".csv" in insulin_data_file:
        insulin_df = pd.read_csv(insulin_data_file, parse_dates=[
                                 ['Date', 'Time']], low_memory=False)

    # Filter only the needed fields to cp_df dataframe
    cp_df = cgm_df[['Date_Time', 'Sensor Glucose (mg/dL)']]
    cp_df = cp_df.set_index(['Date_Time'])
    cp_df.sort_index(inplace=True)

    # Filter only the needed fields to cp_ins_df dataframe
    cp_ins_df = insulin_df[['Date_Time', 'BWZ Estimate (U)', 'BWZ Carb Input (grams)']]

    # extract rows with Carb/meal intake values > 0
    meal_intake_rows = cp_ins_df.loc[cp_ins_df['BWZ Carb Input (grams)'] > 0, [
        'Date_Time', 'BWZ Estimate (U)', 'BWZ Carb Input (grams)']]
    meal_intake_rows.sort_values(['Date_Time'], inplace=True)
    meal_intake_rows.reset_index(inplace=True)
    meal_intake_rows.drop('index', inplace=True, axis=1)
    
    valid_meal_data_times = meal_intake_rows

    rows_to_drop = []
    last_date = valid_meal_data_times['Date_Time'][0]-timedelta(hours=10)

    for ind, row in valid_meal_data_times.iterrows():
        if row['Date_Time'] < (last_date+timedelta(hours=4)):
            rows_to_drop.append(ind-1)
        last_date = row['Date_Time']

    valid_meal_data_times.drop(rows_to_drop, inplace=True)
    valid_meal_data_times.reset_index(inplace=True)

    # Extract Meal and No_meal window data
    meal_data = pd.DataFrame()
    no_meal_data = pd.DataFrame()

    for ind, row in valid_meal_data_times.iterrows():
        # meal_time window data
        m_data = cp_df[row['Date_Time'] -
                       timedelta(minutes=30):row['Date_Time']+timedelta(hours=2)]
        # no_meal_time window data
        n_m_data = cp_df[row['Date_Time'] +
                         timedelta(hours=2):row['Date_Time']+timedelta(hours=4)]

        m_data.reset_index(inplace=True)
        n_m_data.reset_index(inplace=True)

        # Avoid meal and no_meal data instances with less than 30 and 24 observations respectively on a particular time window
        # Avoid instances with more than 5 NaN values
        # if (len(m_data) >= 30) and (m_data['Sensor Glucose (mg/dL)'][:30].isna().sum() <= 28):
        if (len(m_data) >= 30):
            m_data = m_data['Sensor Glucose (mg/dL)'][:30]
            m_data.at[30] = round(row['BWZ Estimate (U)']) # Insulin Bolus Data
            meal_data = pd.concat(
                [meal_data, m_data], ignore_index=True, axis=1)

        if (len(n_m_data) >= 24) and (n_m_data['Sensor Glucose (mg/dL)'][:24].isna().sum() <= 5):
        # if (len(n_m_data) >= 24):
            no_meal_data = pd.concat(
                [no_meal_data, n_m_data['Sensor Glucose (mg/dL)'][:24]], ignore_index=True, axis=1)

    meal_data = meal_data.transpose()
    no_meal_data = no_meal_data.transpose()

    return [meal_data, no_meal_data]


# %%
def extract_transactions(data_matrix, cgm_min):
    columns = ["b_max", "b_meal", "insulin_bolus"]
    transaction_matrix = pd.DataFrame(columns=columns)

    for ind, data in data_matrix.iterrows():
        ins_bol = data.iloc[30]
        data.drop([30], inplace=True)
        max_cgm_of_meal = data.max()
        cgm_at_time_of_meal = data.iloc[10]

        if not (np.isnan([max_cgm_of_meal, cgm_at_time_of_meal, ins_bol]).any()):
            transaction = [int((max_cgm_of_meal-cgm_min)/20), int((cgm_at_time_of_meal-cgm_min)/20), int(ins_bol)]
            transaction_matrix.loc[ind] = ["{}_{}".format(x, y) for x, y in zip(columns, transaction)]
            # transaction_matrix.loc[ind] = transaction

        # NOTE: There are some NaNs in cgm_at_time_of_meal values

    return transaction_matrix

# %% [markdown]
# # ENTRY POINT

# %%
# Entry Point

# training_input_files = [
#     ["CGMData.csv", "InsulinData.csv"]]
training_input_files = [
    ["./data/CGMData.csv", "./data/InsulinData.csv"]]
    # ["data/CGMData670GPatient3.csv", "data/InsulinAndMealIntake670GPatient3.csv"]]

meal_train_data_matrix_1, no_meal_train_data_matrix_1 = extract_meal_and_no_meal_instances(
    training_input_files[0][0], training_input_files[0][1])


# %%
cgm_max = meal_train_data_matrix_1.max().max()
cgm_min = meal_train_data_matrix_1.min().min()

n_bins = ceil((cgm_max-cgm_min)/20)


# %%
transactions = extract_transactions(meal_train_data_matrix_1, cgm_min)


# %% [markdown]
# ## One-Hot Encoding

# %%
items = set()
for col in transactions:
    items.update(transactions[col].unique())
itemset = set(items)
encoded_vals = []
for index, row in transactions.iterrows():
    rowset = set(row) 
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)


# %%
freq_items = apriori(ohe_df, min_support=0.005, use_colnames=True, verbose=1)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))


# %%
freq_items_3 = freq_items[(freq_items['length']==3)]
freq_items_3.sort_values(['support'], inplace=True, ascending=False)
max_support = freq_items_3['support'].max()
freq_items_3.to_csv("./Results/Frequent_Itemsets_3.csv", index=False)

most_freq_itemset_3 = freq_items_3[(freq_items_3['support']==max_support)] # Export to 1.csv


# %%
rules = association_rules(freq_items, metric="confidence", min_threshold=0.0004)


# %%
# Add columns with antecedent length and rule of form "{Bmax, Bmeal} -> Ib"
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["is_rule_form"] = rules["antecedents"].apply(lambda x: all([((("b_meal" in str(i)) or ("b_max" in str(i))) and ("insulin" not in str(i))) for i in list(x)]))


# %%
rules_of_form = rules[(rules["is_rule_form"]) & (rules["antecedent_len"] == 2)]
# rules_of_form = rules[(rules["antecedent_len"] == 2)]
rules_of_form.sort_values(['confidence'], inplace=True, ascending=False)
max_confidence = rules_of_form['confidence'].max()

max_conf_rules = rules_of_form[(rules_of_form['confidence']==max_confidence)]   # Export to 2.csv
least_conf_rules = rules_of_form[(rules_of_form['confidence']<=0.15)]   # Export to 3.csv


# %%

most_freq_itemset_3['output'] = ""

for ind, row in most_freq_itemset_3.iterrows():
    res_list = [0,0,0]  #[b_max, b_meal, insulin_bolus]
    for ele in row['itemsets']:
        match = re.match(r"\D+(\d+)", ele)
        if "b_max" in ele:
            res_list[0] = match.group(1)
        elif "b_meal" in ele:
            res_list[1] = match.group(1)
        elif "insulin" in ele:
            res_list[2] = match.group(1)
    out_str = ", ".join(res_list)
    most_freq_itemset_3.loc[ind, 'output'] = "(" + out_str + ")"
        


# %%
def set_rule_output(df):

    df['output'] = ""

    for ind, row in df.iterrows():
        res_list = [0,0,0]  #[b_max, b_meal, insulin_bolus]
        for ele in (list(row['antecedents']) + list(row['consequents'])):
            match = re.match(r"\D+(\d+)", ele)
            if "b_max" in ele:
                res_list[0] = match.group(1)
            elif "b_meal" in ele:
                res_list[1] = match.group(1)
            elif "insulin" in ele:
                res_list[2] = match.group(1)
        ant = "{}, {}".format(res_list[0], res_list[1])
        cons = "{}".format(res_list[2])
        out_str = "{" + ant + "} ->" + cons
        df.loc[ind, 'output'] = out_str

# %% [markdown]
# # Export Results

# %%
set_rule_output(max_conf_rules)
set_rule_output(least_conf_rules)


# %%
# rules_of_form.to_csv("./Results/Asociation_Rules.csv", index=False, header=False)
# most_freq_itemset_3['output'].to_csv("./Results/Most_Frequent_Itemsets_3.csv", index=False, header=False)
# max_conf_rules['output'].to_csv("./Results/Highest_Confidence_Rules.csv", index=False, header=False)
# least_conf_rules['output'].to_csv("./Results/Low_Confidence_Rules.csv", index=False, header=False)
rules_of_form.to_csv("./Results/Asociation_Rules.csv", index=False)
most_freq_itemset_3['output'].to_csv("./Results/Most_Frequent_Itemsets_3.csv", index=False)
max_conf_rules['output'].to_csv("./Results/Highest_Confidence_Rules.csv", index=False)
least_conf_rules['output'].to_csv("./Results/Low_Confidence_Rules.csv", index=False)

