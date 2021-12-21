# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from math import isnan
from numpy.core.numeric import NaN
# import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
# import pickle_compat
# pickle_compat.patch()


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

    rows_to_drop = []
    last_date = valid_meal_data_times['Date_Time'][0]-timedelta(hours=10)

    for ind, row in valid_meal_data_times.iterrows():
        if row['Date_Time'] < (last_date+timedelta(hours=4)):
            rows_to_drop.append(ind-1)
        last_date = row['Date_Time']
        
    valid_meal_data_times.drop(rows_to_drop, inplace=True)
    valid_meal_data_times.reset_index(inplace=True)

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

        # Avoid meal and no_meal data instances with less than 30 and 24 observations respectively on a particular time window
        # Avoid instances with more than 5 NaN values
        # if (len(m_data) >= 30) and (m_data['Sensor Glucose (mg/dL)'][:30].isna().sum() <= 5):
        if (len(m_data) >= 30):
            meal_data = pd.concat([meal_data, m_data['Sensor Glucose (mg/dL)'][:30]], ignore_index=True, axis=1)
        
        # if (len(n_m_data) >= 24) and (n_m_data['Sensor Glucose (mg/dL)'][:24].isna().sum() <= 5):
        if (len(n_m_data) >= 24):
            no_meal_data = pd.concat([no_meal_data, n_m_data['Sensor Glucose (mg/dL)'][:24]], ignore_index=True, axis=1)
        
    meal_data = meal_data.transpose()
    no_meal_data = no_meal_data.transpose()

    return [meal_data, no_meal_data]

# %% [markdown]
# FEATURE EXTRACTION - MEAL & NO MEAL

# %%
def calcSlope(series):
    res = np.polyfit(range(len(series)), series, 1)
    return res[0]

def extract_features(data_matrix, features):

    feature_matrix = pd.DataFrame(columns=features)

    slope_sampling_size = 2
    num_of_zcs = 3
    num_of_fds = 3

    app_ci = True
    app_dist = True

    for ind, data in data_matrix.iterrows():
        # Max-Min Distance Feature
        f1_diff = data.max() - data.min()
        if isnan(f1_diff):
            continue

        # Slope feature
        slope_res = data.rolling(slope_sampling_size).apply(calcSlope)
        slope_res = slope_res.fillna(0)
        zero_crossings = np.where(np.diff(np.sign(slope_res)))[0]   # Zero crossing indexes of slope
        zero_crossings = np.hstack([zero_crossings, np.array(len(slope_res)-1)])
        zero_cross_dist_df = pd.DataFrame(columns=['cross_index', 'distance'])
        zc_idx = 0
        for idx, slope_idx in enumerate(zero_crossings):
            if (idx < 2) or (idx == (len(zero_crossings)-1)):
                pass
            else:
                # Calculate the dist between Max and Min slopes on either sides of a zero crossing
                # Max and Min sides depends on the sign of slope at zero crossing (if '-', the curve is increasing (Max->right, Min->left) and vice versa)
                if slope_res[slope_idx] < 0:
                    dist = max(slope_res[slope_idx:zero_crossings[idx+1]+1]) - min(slope_res[zero_crossings[idx-1]:slope_idx+1])
                else:
                    dist = max(slope_res[zero_crossings[idx-1]:slope_idx+1]) - min(slope_res[slope_idx:zero_crossings[idx+1]+1])
                zero_cross_dist_df.loc[zc_idx] = [slope_idx, dist]
                zc_idx += 1

        zero_cross_dist_df.sort_values(['distance'], inplace=True, ascending=False)
        zero_cross_dist_df.reset_index(inplace=True)

        f2_slope_zero_cross_ordered_dist = []
        len_zero_cross_dist_df = len(zero_cross_dist_df)

        if len_zero_cross_dist_df < 3:
            # # Skip data that has less than three slope zero crossings
            continue

        # for i in range(len_zero_cross_dist_df):
        #     f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
        #     f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])

        for i in range(num_of_zcs):
            # if i >= len_zero_cross_dist_df:
            #     if (app_dist) and (app_ci):
            #         f2_slope_zero_cross_ordered_dist.append(feature_mean[(i*2)-1])
            #         f2_slope_zero_cross_ordered_dist.append(feature_mean[(i*2)])
            #     elif (app_ci) or (app_dist):
            #         f2_slope_zero_cross_ordered_dist.append(feature_mean[(1+i)])
            if i >= len_zero_cross_dist_df:
                if (app_dist) and (app_ci):
                    f2_slope_zero_cross_ordered_dist.append(0)
                    f2_slope_zero_cross_ordered_dist.append(0)
                elif (app_ci) or (app_dist):
                    f2_slope_zero_cross_ordered_dist.append(0)
            else:
                if (app_dist):
                    f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
                if (app_ci):
                    f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])
            
        # Max-Min Value Index range/distance Feature
        f3_slot_diff = abs(data.idxmax() - data.idxmin())
        if isnan(f3_slot_diff):
            continue

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
        f4_freq_domain = list(peaks[:num_of_fds])
        if len(f4_freq_domain) < num_of_fds:
            # # Skip data that has less than 3 frequency peaks after FFT
            continue
            # while len(f4_freq_domain) < num_of_fds:
            #     f4_freq_domain.append(0)
                # if (app_dist) and (app_ci):
                #     f4_freq_domain.append(feature_mean[2+(num_of_zcs*2)+len(f4_freq_domain)])
                # elif (app_dist) or (app_ci):
                #     f4_freq_domain.append(feature_mean[2+(num_of_zcs*1)+len(f4_freq_domain)])
        feature_matrix.loc[ind] = [f1_diff] + f2_slope_zero_cross_ordered_dist + [f3_slot_diff] + f4_freq_domain
            
    return feature_matrix


def split_train_test_tests(meal_feature, no_meal_feature, features):
    ## TRAIN and TEST Splits

    # meal train_test_split() - Train: 80%, Test: 20%
    meal_x_train, meal_x_test, meal_y_train, meal_y_test = train_test_split(meal_feature.loc[:, features], meal_feature.loc[:, [target]], train_size=0.8)

    # no_meal train_test_split() - Train: 80%, Test: 20%
    no_meal_x_train, no_meal_x_test, no_meal_y_train, no_meal_y_test = train_test_split(no_meal_feature.loc[:, features], no_meal_feature.loc[:, [target]], train_size=0.8)

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

    # To be used for model train & test
    train_pca = pca.transform(x_train)

    classifier.fit(train_pca, y_train.ravel())


def test_model(x_test, y_test, classifier, pca):

    x_test = StandardScaler().fit_transform(x_test)
    # PCA
    test_pca = pca.transform(x_test)

    y_pred = classifier.predict(test_pca)

    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print("Precision : ", metrics.precision_score(y_test, y_pred))
    print("Recall : ", metrics.recall_score(y_test, y_pred))
    print("F1 : ", metrics.f1_score(y_test, y_pred))

    return y_pred


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
# Entry Point

# %%
#FROMAT: [cgm_data, insulin_data]
training_input_files = [["data/CGMData.csv", "data/InsulinData.csv"], ["data/CGMData670GPatient3.csv", "data/InsulinAndMealIntake670GPatient3.csv"]]

meal_train_data_matrix_1, no_meal_train_data_matrix_1 = extract_meal_and_no_meal_instances(training_input_files[0][0], training_input_files[0][1])
meal_train_data_matrix_2, no_meal_train_data_matrix_2 = extract_meal_and_no_meal_instances(training_input_files[1][0], training_input_files[1][1])

param_grid = {'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 'scale', 'auto'],
              'shrinking': [True, False],
              'C': [1.1,1.25,1.5,2,3,3.5,4,4.5,5,5.5,6],
              'probability': [True, False],
              'kernel': ['linear', 'rbf']}

rf_param_grid = {
    'n_estimators': [50, 100, 500, 1000]
}

scorer = 'f1'
# n_components = 0.95
# classifier = GridSearchCV(RandomForestClassifier(), rf_param_grid, refit=True, verbose=1, scoring=scorer, cv=3)

# classifier = svm.SVC(kernel= 'rbf', C= 1.25, shrinking= True, probability= True, gamma= 1)
# classifier = svm.LinearSVC()
classifier = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 1, scoring=scorer, cv=3)
# classifier = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)

pca_param_grid = {'whiten': [True, False],
              'svd_solver': ['auto', 'full'],
              'copy': [True],
            #   'tol': [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
              'tol': [0.0, 0.1, 0.01, 0.001, 1.0, 0.0001],
              'iterated_power': ['auto', 1, 10, 100, 1000],
              'n_components': [0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]}
            # 'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

# pca = PCA(n_components) # 0.95 => 22.0
# pca = PCA(copy=True, iterated_power='auto', n_components=0.95, random_state=None,
#         svd_solver='auto', tol=0.0, whiten=False)
pca = GridSearchCV(PCA(), pca_param_grid, refit=True, verbose=1)

# features = ['f1_diff', 'f2_slope_cross_dist_1', 'f2_slope_cross_slot_1', 'f2_slope_cross_dist_2', 'f2_slope_cross_slot_2', 'f2_slope_cross_dist_3', 'f2_slope_cross_slot_3', 'f3_slot_diff', 'f4_dom_freq_1', 'f4_dom_freq_2', 'f4_dom_freq_3']

features = ['f1_diff', 
'f2_slope_cross_dist_1', 
'f2_slope_cross_slot_1', 
'f2_slope_cross_dist_2', 
'f2_slope_cross_slot_2', 
'f2_slope_cross_dist_3', 
'f2_slope_cross_slot_3',
'f3_slot_diff',
'f4_dom_freq_1', 
'f4_dom_freq_2',
'f4_dom_freq_3']

# Combine 2 patient data for training
meal_feature_1 = extract_features(meal_train_data_matrix_1, features)
no_meal_feature_1 = extract_features(no_meal_train_data_matrix_1, features)
# meal_feature_1 = extract_features(meal_train_data_matrix_2, features)
# no_meal_feature_1 = extract_features(no_meal_train_data_matrix_2, features)

meal_feature_2 = extract_features(meal_train_data_matrix_2, features)
no_meal_feature_2 = extract_features(no_meal_train_data_matrix_2, features)

meal_feature_1 = pd.concat([meal_feature_1, meal_feature_2])
no_meal_feature_1 = pd.concat([no_meal_feature_1, no_meal_feature_2])

# meal_feature_1 = meal_feature_1[:450]

print("Feature BALANCE:")
print(meal_feature_1.shape)
print(no_meal_feature_1.shape)

target = 'is_meal'
meal_feature_1[target] = 1
no_meal_feature_1[target] = 0

# print(meal_feature_1[:].isna().sum())
# print(no_meal_feature_1[:].isna().sum())

# data_matrix = pd.concat([no_meal_feature_1, no_meal_feature_1])


x_train_1, y_train_1, x_test_1, y_test_1 = split_train_test_tests(meal_feature_1, no_meal_feature_1, features)

# x_train_1, y_train_1 = split_data_label_sets(meal_feature_1, no_meal_feature_1, features, target)

# print(x_train_1)

# Standardization

feature_mean_1 = x_train_1.mean(axis=0)
feature_max_min_diff_1 = x_train_1.max(axis=0) - x_train_1.min(axis=0)
# x_train_1, feature_mean_1, feature_max_min_diff_1 = standardize(x_train_1, get_attributes = True)

std_scalar = StandardScaler()
std_scalar.fit(x_train_1)

# print(np.any(np.isnan(x_train_1)))
# print(np.all(np.isfinite(x_train_1)))
# print(x_train_1)
# for i in x_train_1[:]:
#     if not np.all(np.isfinite(i)):
#         print (i)

x_train_1 = std_scalar.transform(x_train_1)

# print(x_train_1)
# print(np.any(np.isnan(x_train_1)))
# print(np.all(np.isfinite(x_train_1)))
# print(feature_mean_1)
# print(feature_max_min_diff_1)

# PCA
# print(x_train_1)
pca.fit(x_train_1)


# feature_mean_1, feature_max_min_diff_1 = train_model(x_train_1, y_train_1, classifier, pca)
train_model(x_train_1, y_train_1, classifier, pca)

# preds = classifier.predict(X_test)

# rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
# print("RMSE: %f" % (rmse))

# test_model(x_test_1, y_test_1, feature_mean_1, feature_max_min_diff_1, classifier, pca)
test_model(x_test_1, y_test_1, classifier, pca)

# Pickle all needed objects
# print("SVM HP for Scorer: {}".format(scorer))
# print(classifier.best_score_)
# print(classifier.best_params_)
# print("PCA HP")
# print(pca.best_score_)
# print(pca.best_params_)


obj_dict = {'classifier': classifier, 
            'pca': pca, 
            'feature_mean': feature_mean_1, 
            'feature_max_min_diff': feature_max_min_diff_1,
            'std_scalar': std_scalar,
            'features': features}

filename = 'model.pkl'
outfile = open(filename, 'wb')

# pickle.dump(obj_dict, outfile)
outfile.close()
