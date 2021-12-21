# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from math import isnan
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn import svm, metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle


# %%
### Read Input Data

# # Patient 1
# cgm_df = pd.read_csv("data/CGMData.csv", parse_dates=[['Date', 'Time']], low_memory=False)
# insulin_df = pd.read_csv("data/InsulinData.csv", parse_dates=[['Date', 'Time']], low_memory=False)


# Patient 2
# cgm_df = pd.read_excel("data/CGMData670GPatient3.xlsx", parse_dates=[['Date', 'Time']])
# insulin_df = pd.read_excel("data/InsulinAndMealIntake670GPatient3.xlsx", parse_dates=[['Date', 'Time']])


# cgm_df.head()


# %%
### Filter only the needed fields

def extract_meal_and_no_meal_instances(cgm_data_file, insulin_data_file):
    if ".xls" in cgm_data_file:
        cgm_df = pd.read_excel(cgm_data_file, parse_dates=[['Date', 'Time']])
    elif ".csv" in cgm_data_file:
        cgm_df = pd.read_csv(cgm_data_file, parse_dates=[['Date', 'Time']], low_memory=False)

    if ".xls" in insulin_data_file:
        insulin_df = pd.read_excel(insulin_data_file, parse_dates=[['Date', 'Time']])
    elif ".csv" in insulin_data_file:
        insulin_df = pd.read_csv(insulin_data_file, parse_dates=[['Date', 'Time']], low_memory=False)

    cp_df = cgm_df[['Date_Time', 'Sensor Glucose (mg/dL)']]   # Filter only the needed fields to cp_df dataframe
    cp_df = cp_df.set_index(['Date_Time'])
    cp_df.sort_index(inplace=True)

    cp_ins_df = insulin_df[['Date_Time', 'BWZ Carb Input (grams)']]   # Filter only the needed fields to cp_ins_df dataframe

    # extract rows with Carb/meal intake values > 0
    meal_intake_rows = cp_ins_df.loc[cp_ins_df['BWZ Carb Input (grams)'] > 0, ['Date_Time', 'BWZ Carb Input (grams)']]
    meal_intake_rows.sort_values(['Date_Time'], inplace=True)
    meal_intake_rows.reset_index(inplace=True)
    meal_intake_rows.drop('index', inplace=True, axis=1)
    # 'BWZ Carb Input (grams)'
    valid_meal_data_times = meal_intake_rows
    # print(valid_meal_data_times.shape)

    rows_to_drop = []
    last_date = valid_meal_data_times['Date_Time'][0]-timedelta(hours=10)

    for ind, row in valid_meal_data_times.iterrows():
        if row['Date_Time'] < (last_date+timedelta(hours=4)):
            rows_to_drop.append(ind-1)
        last_date = row['Date_Time']
        
    valid_meal_data_times.drop(rows_to_drop, inplace=True)
    valid_meal_data_times.reset_index(inplace=True)
    # print(valid_meal_data_times.shape)

    ####Extract Meal and No_meal window data
    meal_data = pd.DataFrame()
    no_meal_data = pd.DataFrame()

    for ind, row in valid_meal_data_times.iterrows():
        # meal_time window data
        m_data = cp_df[row['Date_Time']-timedelta(minutes=30):row['Date_Time']+timedelta(hours=2)]
        # no_meal_time window data
        n_m_data = cp_df[row['Date_Time']+timedelta(hours=2):row['Date_Time']+timedelta(hours=4)]

        m_data.reset_index(inplace=True)
        n_m_data.reset_index(inplace=True)

        # m_data.fillna(0)
        # n_m_data.fillna(0)

        # Avoid meal and no_meal data instances with less than 30 and 24 observations respectively on a particular time window
        # Avoid instances with more than 5 NaN values
        if (len(m_data) >= 30) and (m_data['Sensor Glucose (mg/dL)'][:30].isna().sum() <= 5):
        # if (len(m_data) >= 30):
            meal_data = pd.concat([meal_data, m_data['Sensor Glucose (mg/dL)'][:30]], ignore_index=True, axis=1)
        
        if (len(n_m_data) >= 24) and (n_m_data['Sensor Glucose (mg/dL)'][:24].isna().sum() <= 5):
        # if (len(n_m_data) >= 24):
            no_meal_data = pd.concat([no_meal_data, n_m_data['Sensor Glucose (mg/dL)'][:24]], ignore_index=True, axis=1)
        
    # print(meal_data.shape)
    # print(no_meal_data.shape)
    meal_data = meal_data.transpose()
    # meal_data.to_csv("meal.csv")
    no_meal_data = no_meal_data.transpose()

    # print(f"Meal Data shape: {meal_data.shape}")
    # print(f"NoMeal Data Shape: {no_meal_data.shape}")

    return [meal_data, no_meal_data]

# %% [markdown]
# FEATURE EXTRACTION - MEAL & NO MEAL

# %%
def calcSlope(series):
    res = np.polyfit(range(len(series)), series, 1)
    # print(f"All Results: {res}")
    return res[0]

def extract_features(data_matrix, features):

    feature_matrix = pd.DataFrame(columns=features)
    temp_features = pd.DataFrame(columns=data_matrix.columns)
    temp_features_ind = 0
    feature_matrix_ind = 0

    slope_sampling_size = 2

    data_mean = []

    # for di, dat in enumerate([meal_data, no_meal_data]):
    #     f_idx = 0
    iteration = 0
    d_matrix = None
    while iteration < 2:
        if iteration == 0:
            d_matrix = data_matrix
        elif iteration == 1:
            d_matrix = temp_features

        for ind, data in d_matrix.iterrows():
            # Max-Min Distance Feature

            f1_diff = data.max() - data.min()

            # Slope feature
            slope_res = data.rolling(slope_sampling_size).apply(calcSlope)
            slope_res = slope_res.fillna(0)
            # slope_res = slope_res[1:]
            zero_crossings = np.where(np.diff(np.sign(slope_res)))[0]   # Zero crossing indexes of slope based on sign change
            zero_crossings = np.hstack([zero_crossings, np.array(len(slope_res)-1)])
            zero_cross_dist_df = pd.DataFrame(columns=['cross_index', 'distance'])
            zc_idx = 0

            if slope_res.isna().sum() > 1:
                print("Slopes : {}".format(slope_res))
            # print("Zero Cross Indices: {}".format(zero_crossings))

            for idx, slope_idx in enumerate(zero_crossings):
                if (idx < 2) or (idx == (len(zero_crossings)-1)):
                    # Ignore crossing if it is first (0 & 1) or last slope values
                    pass
                else:
                    # Calculate the dist between Max and Min slopes on either sides of a zero crossing
                    # Max and Min sides depends on the sign of slope at zero crossing (if '-', the curve is increasing (Max->right, Min->left) and vice versa)
                    if slope_res[slope_idx] < 0:
                        # if isnan(min(slope_res[zero_crossings[idx-1]:slope_idx+1])):
                        #     print(min(slope_res[zero_crossings[idx-1]:slope_idx+1]))
                        # if isnan(max(slope_res[slope_idx:zero_crossings[idx+1]+1])):
                        #     print("No MAX: {}".format(slope_res[slope_idx:zero_crossings[idx+1]+1]))
                        dist = max(slope_res[slope_idx:zero_crossings[idx+1]+1]) - min(slope_res[zero_crossings[idx-1]:slope_idx+1])
                    else:
                        # if isnan(min(slope_res[slope_idx:zero_crossings[idx+1]+1])):
                        #     print(min(slope_res[slope_idx:zero_crossings[idx+1]+1]))
                        # if isnan(max(slope_res[zero_crossings[idx-1]:slope_idx+1])):
                        #     print("No MAX: {}".format(slope_res[zero_crossings[idx-1]:slope_idx+1]))
                        dist = max(slope_res[zero_crossings[idx-1]:slope_idx+1]) - min(slope_res[slope_idx:zero_crossings[idx+1]+1])
                    # if isnan(dist):
                    #     print("Dist: {}".format(dist))
                    zero_cross_dist_df.loc[zc_idx] = [slope_idx, dist]
                    zc_idx += 1

            zero_cross_dist_df.sort_values(['distance'], inplace=True, ascending=False)
            zero_cross_dist_df.reset_index(inplace=True)

            # if ((zero_cross_dist_df['distance'].isna().sum()) > 0):
            #     print("Nan in Zero cross Dist: {}".format(zero_cross_dist_df['distance'].isna().sum()))

            if ((zero_cross_dist_df['cross_index'].isna().sum()) > 0):
                print("Nan in Zero cross Index: {}".format(zero_cross_dist_df['cross_index'].isna().sum()))

            # print(zero_cross_dist_df.isna().sum())

            f2_slope_zero_cross_ordered_dist = []
            len_zero_cross_dist_df = len(zero_cross_dist_df)

            if len_zero_cross_dist_df < 3:
                if iteration==1:
                    # print("Length : {}".format(len_zero_cross_dist_df))
                    for i in range(3):
                        if i >= (len_zero_cross_dist_df):
                            # print("Adding mean at index {}".format(i))
                            # if (data_mean[(i*2)-1] == np.NaN) or (data_mean[(i*2)] == np.NaN):
                            #     print("Data Mean: {}, {}".format(data_mean[(i*2)-1], data_mean[(i*2)]))
                            f2_slope_zero_cross_ordered_dist.append(data_mean[(i*2)-1])
                            f2_slope_zero_cross_ordered_dist.append(data_mean[(i*2)])

                        else:
                            # if (zero_cross_dist_df.loc[i, ['distance']].values[0] == np.NaN) or (zero_cross_dist_df.loc[i, ['cross_index']].values[0] == np.NaN):
                            #     print("Zero cross dist: {}, {}".format(zero_cross_dist_df.loc[i, ['distance']].values[0], zero_cross_dist_df.loc[i, ['cross_index']].values[0]))
                            f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
                            f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])
                    # print("ZeroCross ordered Dist: {}".format(f2_slope_zero_cross_ordered_dist))
                else:
                    # Skip data that has less than three slope zero crossings
                    temp_features.loc[temp_features_ind] = data
                    temp_features_ind += 1
                    continue
            else:
                try:
                    # print("Length : {}".format(len_zero_cross_dist_df))
                    for i in range(3):
                        # print("I: {}".format(i))
                        f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
                        f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])
                except Exception as e:
                    print("My Key Error: ", str(e))

            # for i in range(3):
            #     if i >= len_zero_cross_dist_df:
            #         f2_slope_zero_cross_ordered_dist.append(0)
            #         f2_slope_zero_cross_ordered_dist.append(0)
            #     else:
            #         f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
            #         f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])
                
            # Max-Min Value Index range/distance Feature
            f3_slot_diff = abs(data.idxmax() - data.idxmin())

            # Frequency Domain Feature
            ## Normalize data
            norm_data = data - data.mean()

            yf = rfft(norm_data.values)
            xf = rfftfreq(len(norm_data))
            yf = np.abs(yf)
            
            ## Extract peaks
            peak_idxs, _ = find_peaks(yf)
            peaks = yf[peak_idxs]
            peaks.sort()
            peaks = peaks[::-1]
            f4_freq_domain = list(peaks[:3])
            if len(f4_freq_domain) < 3:
                if iteration == 1:
                    while len(f4_freq_domain) < 3:
                        f4_freq_domain.append(data_mean[8+len(f4_freq_domain)])
                else:
                    # Skip data that has less than 3 frequency peaks after FFT
                    # temp_features = pd.concat([temp_features, data])
                    temp_features.loc[temp_features_ind] = data
                    temp_features_ind += 1
                    continue
            #     for i in range(3-len(f4_freq_domain)):
            #         f4_freq_domain.append(0)

            feature_matrix.loc[feature_matrix_ind] = [f1_diff] + f2_slope_zero_cross_ordered_dist + [f3_slot_diff] + f4_freq_domain
            # if feature_matrix.loc[feature_matrix_ind].isna().sum() > 0:
                # print("NaN Feature Row: {}".format(feature_matrix.loc[feature_matrix_ind]))
            feature_matrix_ind += 1
        iteration += 1
        data_mean = feature_matrix.mean()
        # print(data_mean.isna().sum())
        # print(temp_features.head(25))

    print("Feature matrix shape: {}".format(feature_matrix.shape))
    return feature_matrix


# %%
def standardize(dataframe, get_attributes=False, mean_data=None, max_min_diff=None):
    if get_attributes:
        mean_data = dataframe.mean(axis=0)
        # print("MEAN DATA:")
        # print(mean_data)
        max_data = dataframe.max(axis=0)
        # print("MAX DATA:")
        # print(max_data)
        min_data = dataframe.min(axis=0)
        # print("MIN DATA:")
        # print(min_data)
        max_min_diff = max_data-min_data
        # print("MAX-MIN DATA:")
        # print(max_min_diff)
        dataframe = (dataframe - mean_data)/(max_min_diff)
        # normRawData = (rY - numpy.mean(rY))/(numpy.max(rY-numpy.mean(rY))-numpy.min(rY-numpy.mean(rY)))
        # print(dataframe)
        return (dataframe, mean_data, max_min_diff)
    else:
        dataframe = (dataframe - mean_data)/(max_min_diff)
        return dataframe


# %%
# cgm_df = pd.read_csv("data/CGMData.csv", parse_dates=[['Date', 'Time']], low_memory=False)
# insulin_df = pd.read_csv("data/InsulinData.csv", parse_dates=[['Date', 'Time']], low_memory=False)


# Patient 2
# cgm_df = pd.read_excel("data/CGMData670GPatient3.xlsx", parse_dates=[['Date', 'Time']])
# insulin_df = pd.read_excel("data/InsulinAndMealIntake670GPatient3.xlsx", parse_dates=[['Date', 'Time']])

def split_train_test_tests(meal_feature, no_meal_feature, features):
    ## TRAIN and TEST Splits

    # meal train_test_split() - Train: 80%, Test: 20%
    meal_x_train, meal_x_test, meal_y_train, meal_y_test = train_test_split(meal_feature.loc[:, features], meal_feature.loc[:, [target]], train_size=0.9)

    # no_meal train_test_split() - Train: 80%, Test: 20%
    no_meal_x_train, no_meal_x_test, no_meal_y_train, no_meal_y_test = train_test_split(no_meal_feature.loc[:, features], no_meal_feature.loc[:, [target]], train_size=0.9)

    # Combined Train Set
    meal_train = np.hstack([meal_x_train, meal_y_train])
    no_meal_train = np.hstack([no_meal_x_train, no_meal_y_train])

    all_feature_train_matrix = np.vstack([meal_train, no_meal_train])
    all_feature_train_matrix = pd.DataFrame(all_feature_train_matrix, columns=features+[target])

    # Combined Test Set
    meal_test = np.hstack([meal_x_test, meal_y_test])
    no_meal_test = np.hstack([no_meal_x_test, no_meal_y_test])

    all_feature_test_matrix = np.vstack([meal_test, no_meal_test])
    all_feature_test_matrix = pd.DataFrame(all_feature_test_matrix, columns=features+[target])
    # print(all_feature_train_matrix.shape)


    x_train = all_feature_train_matrix.loc[:, features].values
    y_train = all_feature_train_matrix.loc[:, [target]].values

    x_test = all_feature_test_matrix.loc[:, features].values
    y_test = all_feature_test_matrix.loc[:, [target]].values

    return (x_train, y_train, x_test, y_test)


def train_model(x_train, y_train, classifier, pca):

    # # Standardization

    # x_train, feature_mean, feature_max_min_diff = standardize(x_train, get_attributes = True)

    # # PCA

    # pca.fit(x_train)

    # To be used for model train & test
    train_pca = pca.transform(x_train)

    classifier.fit(train_pca, y_train.ravel())

    # return (feature_mean, feature_max_min_diff)

# %% [markdown]
# PCA

# %%

def test_model(x_test, y_test, feature_mean, feature_max_min_diff, classifier, pca):
    # Standardize
    x_test = standardize(x_test, False, feature_mean, feature_max_min_diff)

    # PCA
    test_pca = pca.transform(x_test)

    y_pred = classifier.predict(test_pca)

    # print(y_pred)

    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print("Precision : ", metrics.precision_score(y_test, y_pred))
    print("Recall : ", metrics.recall_score(y_test, y_pred))
    print("F1 : ", metrics.f1_score(y_test, y_pred))


# %%
def split_data_label_sets(data_matrix_1, data_matrix_2, label, target):

    train_data_1 = data_matrix_1.loc[:, features].to_numpy()
    train_data_2 = data_matrix_2.loc[:, features].to_numpy()

    train_label_1 = data_matrix_1.loc[:, [target]].to_numpy()
    train_label_2 = data_matrix_2.loc[:, [target]].to_numpy()

    x_train = np.vstack([train_data_1, train_data_2])
    y_train = np.vstack([train_label_1, train_label_2])

    return [x_train, y_train]

    

# %% [markdown]
# MAIN FUNCTION

# %%
#FROMAT: [cgm_data, insulin_data]
training_input_files = [["data/CGMData.csv", "data/InsulinData.csv"], ["data/CGMData670GPatient3.csv", "data/InsulinAndMealIntake670GPatient3.csv"]]

# cgm_df = pd.read_csv("data/CGMData670GPatient3", parse_dates=[['Date', 'Time']], low_memory=False)

# print(cgm_df.head())

meal_train_data_matrix_1, no_meal_train_data_matrix_1 = extract_meal_and_no_meal_instances(training_input_files[0][0], training_input_files[0][1])
meal_train_data_matrix_2, no_meal_train_data_matrix_2 = extract_meal_and_no_meal_instances(training_input_files[1][0], training_input_files[1][1])

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
            #   'gamma': ['scale', 'auto'],
              'kernel': ['linear']}

# classifier = svm.SVC(kernel='linear')
classifier = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0)
pca = PCA(0.97)

features = ['f1_diff', 'f2_slope_cross_dist_1', 'f2_slope_cross_slot_1', 'f2_slope_cross_dist_2', 'f2_slope_cross_slot_2', 'f2_slope_cross_dist_3', 'f2_slope_cross_slot_3', 'f3_slot_diff', 'f4_dom_freq_1', 'f4_dom_freq_2', 'f4_dom_freq_3']

# Combine 2 patient data for training
print("Train Data Matrix Shape: {}".format(meal_train_data_matrix_1.shape))
meal_feature_1 = extract_features(meal_train_data_matrix_1, features)
print("Train Data Matrix Shape: {}".format(no_meal_train_data_matrix_1.shape))
no_meal_feature_1 = extract_features(no_meal_train_data_matrix_1, features)

# meal_feature_1 = extract_features(meal_train_data_matrix_2, features)
# no_meal_feature_1 = extract_features(no_meal_train_data_matrix_2, features)

print("Train Data Matrix Shape: {}".format(meal_train_data_matrix_2.shape))
meal_feature_2 = extract_features(meal_train_data_matrix_2, features)
print("Train Data Matrix Shape: {}".format(no_meal_train_data_matrix_2.shape))
no_meal_feature_2 = extract_features(no_meal_train_data_matrix_2, features)

meal_feature_1 = pd.concat([meal_feature_1, meal_feature_2])
no_meal_feature_1 = pd.concat([no_meal_feature_1, no_meal_feature_2])

# print(meal_feature_1.shape)
# print(no_meal_feature_1.shape)

target = 'is_meal'
meal_feature_1[target] = 1
no_meal_feature_1[target] = 0


x_train_1, y_train_1, x_test_1, y_test_1 = split_train_test_tests(meal_feature_1, no_meal_feature_1, features)

x_train_1, y_train_1 = split_data_label_sets(meal_feature_1, no_meal_feature_1, features, target)

# print(x_train_1)

# Standardization

x_train_1, feature_mean_1, feature_max_min_diff_1 = standardize(x_train_1, get_attributes = True)

# print(feature_mean_1)
# print(feature_max_min_diff_1)

# PCA
# print(x_train_1)
pca.fit(x_train_1)


# feature_mean_1, feature_max_min_diff_1 = train_model(x_train_1, y_train_1, classifier, pca)
train_model(x_train_1, y_train_1, classifier, pca)

test_model(x_test_1, y_test_1, feature_mean_1, feature_max_min_diff_1, classifier, pca)

# Pickle all needed objects

obj_dict = {'classifier': classifier, 'pca': pca, 'feature_mean': feature_mean_1, 'feature_max_min_diff': feature_max_min_diff_1}

filename = 'model.pkl'
outfile = open(filename, 'wb')

pickle.dump(obj_dict, outfile)
outfile.close()


