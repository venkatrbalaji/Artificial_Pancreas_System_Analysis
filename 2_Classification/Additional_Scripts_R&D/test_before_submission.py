# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpy.core.numeric import NaN
from operator import sub
import pandas as pd
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks
import pickle
from sklearn.preprocessing import StandardScaler
import pickle_compat
pickle_compat.patch()



# %%
### Read Input Data

# # Patient 1
# cgm_df = pd.read_csv("data/CGMData.csv", parse_dates=[['Date', 'Time']], low_memory=False)
# insulin_df = pd.read_csv("data/InsulinData.csv", parse_dates=[['Date', 'Time']], low_memory=False)


# Patient 2
# cgm_df = pd.read_excel("data/CGMData670GPatient3.xlsx", parse_dates=[['Date', 'Time']])
# insulin_df = pd.read_excel("data/InsulinAndMealIntake670GPatient3.xlsx", parse_dates=[['Date', 'Time']])


# cgm_df.head()

# %% [markdown]
# FEATURE EXTRACTION - MEAL & NO MEAL

# %%
def calcSlope(series):
    res = np.polyfit(range(len(series)), series, 1)
    # print(f"All Results: {res}")
    return res[0]

def extract_features(data_matrix, features, feature_mean):

    feature_matrix = pd.DataFrame(columns=features)
    # no_meal_feature = pd.DataFrame(columns=features)

    slope_sampling_size = 2
    num_of_zcs = 3
    num_of_fds = 3

    app_ci = True
    # app_ci = False
    app_dist = True
    # app_dist = False

    # for di, dat in enumerate([meal_data, no_meal_data]):
    #     f_idx = 0
    for ind, data in data_matrix.iterrows():
        # Max-Min Distance Feature
        f1_diff = data.max() - data.min()

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

        # if len_zero_cross_dist_df < 3:
        #     # # Skip data that has less than three slope zero crossings
        #     # continue
        #     for i in range(len_zero_cross_dist_df):
        #         f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
        #         f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])

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
        # if len(f4_freq_domain) < 1:
        #     # # Skip data that has less than 3 frequency peaks after FFT
        #     # continue
        #     # for i in range(3-len(f4_freq_domain)):
        #     #     f4_freq_domain.append(0)
        #     while len(f4_freq_domain) < num_of_fds:
        #         f4_freq_domain.append(feature_mean[8+len(f4_freq_domain)])
        if len(f4_freq_domain) < num_of_fds:
            # # Skip data that has less than 3 frequency peaks after FFT
            # continue
            # for i in range(3-len(f4_freq_domain)):
            #     f4_freq_domain.append(0)
            # sub_freq = f4_freq_domain[-1]
            # while len(f4_freq_domain) < num_of_fds:
            #     f4_freq_domain.append(sub_freq)
            while len(f4_freq_domain) < num_of_fds:
                f4_freq_domain.append(0)
                # if (app_dist) and (app_ci):
                #     f4_freq_domain.append(feature_mean[2+(num_of_zcs*2)+len(f4_freq_domain)])
                # elif (app_dist) or (app_ci):
                #     f4_freq_domain.append(feature_mean[2+(num_of_zcs*1)+len(f4_freq_domain)])
        feature_matrix.loc[ind] = [f1_diff] + f2_slope_zero_cross_ordered_dist + [f3_slot_diff] + f4_freq_domain
            
    # print(f"Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix

# def extract_features(data_matrix, features):

#     feature_matrix = pd.DataFrame(columns=features)

#     slope_sampling_size = 2

#     # for di, dat in enumerate([meal_data, no_meal_data]):
#     #     f_idx = 0
#     for ind, data in data_matrix.iterrows():
#         # Max-Min Distance Feature
#         f1_diff = data.max() - data.min()

#         # Slope feature
#         slope_res = data.rolling(slope_sampling_size).apply(calcSlope)
#         zero_crossings = np.where(np.diff(np.sign(slope_res)))[0]   # Zero crossing indexes of slope
#         zero_crossings = np.hstack([zero_crossings, np.array(len(slope_res)-1)])
#         zero_cross_dist_df = pd.DataFrame(columns=['cross_index', 'distance'])
#         zc_idx = 0
#         for idx, slope_idx in enumerate(zero_crossings):
#             if (idx < 2) or (idx == (len(zero_crossings)-1)):
#                 pass
#             else:
#                 # Calculate the dist between Max and Min slopes on either sides of a zero crossing
#                 # Max and Min sides depends on the sign of slope at zero crossing (if '-', the curve is increasing (Max->right, Min->left) and vice versa)
#                 if slope_res[slope_idx] < 0:
#                     dist = max(slope_res[slope_idx:zero_crossings[idx+1]+1]) - min(slope_res[zero_crossings[idx-1]:slope_idx+1])
#                 else:
#                     dist = max(slope_res[zero_crossings[idx-1]:slope_idx+1]) - min(slope_res[slope_idx:zero_crossings[idx+1]+1])
#                 zero_cross_dist_df.loc[zc_idx] = [slope_idx, dist]
#                 zc_idx += 1

#         zero_cross_dist_df.sort_values(['distance'], inplace=True, ascending=False)
#         zero_cross_dist_df.reset_index(inplace=True)

#         f2_slope_zero_cross_ordered_dist = []
#         len_zero_cross_dist_df = len(zero_cross_dist_df)

#         # if len_zero_cross_dist_df < 3:
#         #     # Skip data that has less than three slope zero crossings
#         #     continue

#         # for i in range(3):
#         #     f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
#         #     f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])

#         for i in range(3):
#             if i >= len_zero_cross_dist_df:
#                 f2_slope_zero_cross_ordered_dist.append(NaN)
#                 f2_slope_zero_cross_ordered_dist.append(NaN)
#             else:
#                 f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['distance']].values[0])
#                 f2_slope_zero_cross_ordered_dist.append(zero_cross_dist_df.loc[i, ['cross_index']].values[0])
            
#         # Max-Min Value Index range/distance Feature
#         f3_slot_diff = abs(data.idxmax() - data.idxmin())

#         # Frequency Domain Feature
#         ## Normalize data
#         norm_data = data - data.mean()

#         yf = rfft(norm_data.values)
#         xf = rfftfreq(len(norm_data))
#         yf = np.abs(yf)
        
#         ## Extract peaks
#         peak_idxs, _ = find_peaks(yf)
#         peaks = yf[peak_idxs]
#         peaks.sort()
#         peaks = peaks[::-1]
#         f4_freq_domain = list(peaks[:3])
#         if len(f4_freq_domain) < 3:
#             # Skip data that has less than 3 frequency peaks after FFT
#             # continue
#             # for i in range(3-len(f4_freq_domain)):
#             while len(f4_freq_domain) < 3:
#                 f4_freq_domain.append(NaN)

#         feature_matrix.loc[ind] = [f1_diff] + f2_slope_zero_cross_ordered_dist + [f3_slot_diff] + f4_freq_domain
            
#     # print(f"Feature matrix shape: {feature_matrix.shape}")
#     feature_matrix = feature_matrix.fillna(method='ffill').fillna(method='bfill')
#     return feature_matrix



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

# %% [markdown]
# PCA

# %%

def test_model(x_test, classifier, pca):

    # print(x_test)
    # PCA
    test_pca = pca.transform(x_test)
    y_pred = classifier.predict(test_pca)

    # y_pred = classifier.predict(x_test)

    # print(y_pred)

    # print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    return y_pred

# %% [markdown]
# MAIN FUNCTION

# %%
#FROMAT: [cgm_data, insulin_data]
# training_input_files = [["data/CGMData.csv", "data/InsulinData.csv"], ["data/CGMData670GPatient3.xlsx", "data/InsulinAndMealIntake670GPatient3.xlsx"]]
test_input_files = ["test.csv"]
# test_input_files = ["data/test.csv"]

# meal_train_data_matrix_1, no_meal_train_data_matrix_1 = extract_meal_and_no_meal_instances(training_input_files[0][0], training_input_files[0][1])
# meal_train_data_matrix_2, no_meal_train_data_matrix_2 = extract_meal_and_no_meal_instances(training_input_files[1][0], training_input_files[1][1])

test_data_matrix = pd.read_csv(test_input_files[0], low_memory=False, header=None)

# print(test_data_matrix.head(30))

filename = 'model.pkl'
infile = open(filename, 'rb')
obj_dict = pickle.load(infile)

classifier = obj_dict['classifier']
pca = obj_dict['pca']
feature_mean = obj_dict['feature_mean']
feature_max_min_diff = obj_dict['feature_max_min_diff']
std_scalar = obj_dict['std_scalar']
features = obj_dict['features']

infile.close()

# features = ['f1_diff', 'f2_slope_cross_dist_1', 'f2_slope_cross_slot_1', 'f2_slope_cross_dist_2', 'f2_slope_cross_slot_2', 'f2_slope_cross_dist_3', 'f2_slope_cross_slot_3', 'f3_slot_diff', 'f4_dom_freq_1', 'f4_dom_freq_2', 'f4_dom_freq_3']

# features = ['f1_diff', 'f2_slope_cross_dist_1', 'f2_slope_cross_slot_1', 'f2_slope_cross_dist_2', 'f2_slope_cross_slot_2', 'f3_slot_diff', 'f4_dom_freq_1', 'f4_dom_freq_2']

# features = ['f1_diff', 'f2_slope_cross_dist_1', 'f2_slope_cross_slot_1', 'f3_slot_diff', 'f4_dom_freq_1']

# features = ['f1_diff', 
# # 'f2_slope_cross_dist_1', 
# 'f2_slope_cross_slot_1', 
# # 'f2_slope_cross_dist_2', 
# 'f2_slope_cross_slot_2', 
# # 'f2_slope_cross_dist_3', 
# # 'f2_slope_cross_slot_3',
# 'f3_slot_diff',
# 'f4_dom_freq_1', 
# 'f4_dom_freq_2']
# # 'f4_dom_freq_3']

# Extract Feature
test_feature_matrix = extract_features(test_data_matrix, features, feature_mean)
# test_feature_matrix = extract_features(test_data_matrix, features)


# test_feature_matrix = test_feature_matrix.loc[:, features]

# Standardize
# x_test = standardize(test_feature_matrix, False, feature_mean, feature_max_min_diff)

# x_test, feature_mean_1, feature_max_min_diff_1 = standardize(test_feature_matrix, get_attributes = True)
# x_test = StandardScaler().fit_transform(test_feature_matrix)
x_test = std_scalar.transform(test_feature_matrix)

result = test_model(x_test, classifier, pca)

np.savetxt('Results.csv', result, delimiter=",")


