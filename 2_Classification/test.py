# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks
import pickle
# import pickle_compat
# pickle_compat.patch()


# %% [markdown]
# FEATURE EXTRACTION - MEAL & NO MEAL

def calcSlope(series):
    res = np.polyfit(range(len(series)), series, 1)
    return res[0]

def extract_features(data_matrix, features, feature_mean):

    feature_matrix = pd.DataFrame(columns=features)

    slope_sampling_size = 2
    num_of_zcs = 3
    num_of_fds = 3

    app_ci = True
    app_dist = True

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

        for i in range(num_of_zcs):
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

        while len(f4_freq_domain) < num_of_fds:
            f4_freq_domain.append(0)

        feature_matrix.loc[ind] = [f1_diff] + f2_slope_zero_cross_ordered_dist + [f3_slot_diff] + f4_freq_domain
            
    # print(f"Feature matrix shape: {feature_matrix.shape}")
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

def test_model(x_test, classifier, pca):

    # PCA
    test_pca = pca.transform(x_test)

    y_pred = classifier.predict(test_pca)

    return y_pred

# %% [markdown]
## Entry Point

# Place the test data in ./data folder
test_input_files = ["./data/test.csv"]

test_data_matrix = pd.read_csv(test_input_files[0], low_memory=False, header=None)

# MOdel Pickle file in the current directory
filename = 'classifier_model.pkl'
infile = open(filename, 'rb')
obj_dict = pickle.load(infile)

classifier = obj_dict['classifier']
pca = obj_dict['pca']
feature_mean = obj_dict['feature_mean']
feature_max_min_diff = obj_dict['feature_max_min_diff']
std_scalar = obj_dict['std_scalar']
features = obj_dict['features']

infile.close()

# Extract Feature
test_feature_matrix = extract_features(test_data_matrix, features, feature_mean)

# Standardize
x_test = std_scalar.transform(test_feature_matrix)

result = test_model(x_test, classifier, pca)

np.savetxt('./Results/Classification_Results.csv', result, delimiter=",")


