import pickle
import pandas as pd
import time

from numpy import random
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn import pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, f1_score, precision_recall_curve, plot_precision_recall_curve, average_precision_score, auc
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy import stats
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import plot_confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
from skimage import io, transform
import sys
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score
#import tqdm


t1 = time.time()

old_df_path = sys.argv[1]
cnn_output_path = sys.argv[2]
final_df_path = sys.argv[3]

#2011
household_ind_input = pickle.load(open(old_df_path,'rb'))
#/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/5.Train_Regression_and_Classification_Models/Inputs/Household_Indicators_Input/household_input_2011.pickle

household_ind_input = household_ind_input.iloc[:, :105]
print(len(household_ind_input))

#UNCOMMENT For 2001

mapping = pickle.load(open('/home/cse/mtech/mcs202448/workingDir/arch4/Data/2011_2001_corr_and_nbrs/2001_2011_corr.pickle', 'rb'))
household_ind_input = household_ind_input.rename(index=mapping)

#LOADING FINAL LAYER OUTPUTS

year = '2001'
#year = sys.argv[4]
#folder = year+'_predictions'
folder = cnn_output_path
fc_adv_reg = pickle.load(open(folder+'fc/FC_ADV.pickle', 'rb'))
fc_int_reg = pickle.load(open(folder+'fc/FC_INT.pickle', 'rb'))
fc_rud_reg = pickle.load(open(folder+'fc/FC_RUD.pickle', 'rb'))
bf_adv_reg = pickle.load(open(folder+'bf/BF_ADV.pickle', 'rb'))
bf_int_reg = pickle.load(open(folder+'bf/BF_INT.pickle', 'rb'))
bf_rud_reg = pickle.load(open(folder+'bf/BF_RUD.pickle', 'rb'))
msw_adv_reg = pickle.load(open(folder+'msw/MSW_ADV.pickle', 'rb'))
msw_int_reg = pickle.load(open(folder+'msw/MSW_INT.pickle', 'rb'))
msw_rud_reg = pickle.load(open(folder+'msw/MSW_RUD.pickle', 'rb'))
asset_adv_reg = pickle.load(open(folder+'asset/ASSET_ADV.pickle', 'rb'))
asset_int_reg = pickle.load(open(folder+'asset/ASSET_INT.pickle', 'rb'))
asset_rud_reg = pickle.load(open(folder+'asset/ASSET_RUD.pickle', 'rb'))

# msl_adv_reg = pickle.load(open('/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/pretraining_data/2001_predictions/MSL_ADV.pickle', 'rb'))
# msl_int_reg = pickle.load(open('/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/pretraining_data/2001_predictions/MSL_INT.pickle', 'rb'))
# msl_rud_reg = pickle.load(open('/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/pretraining_data/2001_predictions/MSL_RUD.pickle', 'rb'))

lit_adv_reg = pickle.load(open(folder+'lit/LIT_ADV.pickle', 'rb'))
lit_int_reg = pickle.load(open(folder+'lit/LIT_INT.pickle', 'rb'))
lit_rud_reg = pickle.load(open(folder+'lit/LIT_RUD.pickle', 'rb'))

indicator_name_dict = {'FC_ADV' : fc_adv_reg, 'FC_INT' : fc_int_reg, 'FC_RUD' : fc_rud_reg, 'BF_ADV' : bf_adv_reg, 'BF_INT' : bf_int_reg, 'BF_RUD' : bf_rud_reg, 'MSW_ADV' : msw_adv_reg, 
                        'MSW_INT' : msw_int_reg,'MSW_RUD' : msw_rud_reg,  'ASSET_ADV' : asset_adv_reg, 'ASSET_INT' : asset_int_reg, 'ASSET_RUD' : asset_rud_reg,  
                        'LIT_ADV' : lit_adv_reg, 'LIT_INT' : lit_int_reg, 'LIT_RUD' : lit_rud_reg,}

if list(msw_adv_reg.keys()).sort() == list(lit_rud_reg.keys()).sort():
    print('same village ids (identical list)')
else:
    print('fail')

reqd_vill_id = list(msw_rud_reg.keys())
reqd_vill_id.sort()


final_reg_in = household_ind_input.filter(items=reqd_vill_id, axis=0)

household_ind_reg_input = pd.DataFrame(index=final_reg_in.index.copy(), columns =  [])

neighbours = pickle.load(open('/home/cse/mtech/mcs202448/workingDir/arch4/Data/Village_Neighbors/generated_neighbours.pickle', 'rb'))
#/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/1.Generating_Neighbours/generated_neighbours.pickle'

for indicator_name in indicator_name_dict:
    print(indicator_name)

for indicator_name in indicator_name_dict:
    indicator_column_list = []
    output_number = '0'
    indicator_name_list = indicator_name.split('_')
    
    if indicator_name_list[1] == 'RUD':
        output_number = '1'
    elif indicator_name_list[1] == 'INT' :
        output_number = '2'
    elif indicator_name_list[1] == 'ADV' :
        output_number = '3'
    else :
        output_number = '4'

    for i in range(0, 6):
        indicator_column_list.append(indicator_name_list[0] + '_' + str(i) + '_OUT_' + output_number)
    
    nbr_avg_feature_name = indicator_name_list[0] + '_N_OUT_' + output_number
    # fc_adv_reg_list = ['FC_0_OUT_3', 'FC_1_OUT_3', 'FC_2_OUT_3', 'FC_3_OUT_3', 'FC_4_OUT_3', 'FC_5_OUT_3']
    for nme in indicator_column_list :
        household_ind_reg_input[nme] = np.nan
        
    household_ind_reg_input[nbr_avg_feature_name] = np.nan

    for vill_id in tqdm(household_ind_reg_input.index.tolist()):
        count = 0
        sum = 0
        for nbr in neighbours[int(vill_id)]:
            predicted_output = indicator_name_dict[indicator_name].get(int(nbr), 0)
            if count != 0:
                sum += predicted_output
            
            household_ind_reg_input.loc[vill_id, indicator_column_list[count]] = predicted_output
            count+=1
        household_ind_reg_input.loc[vill_id, nbr_avg_feature_name] = sum/5.0


cols_total = household_ind_reg_input.columns
print("total columns", len(cols_total))

ind_names = ['BF', 'FC', 'MSW', 'ASSET', 'LIT']
num_list = ['0', '1', '2', '3', '4', '5', 'N']
final_col_list = []


for i in num_list:
    for ind in ind_names :
        for j in range(1,4):
            final_col_list.append(ind+'_'+i+'_OUT_'+str(j))

print(final_col_list)
print(len(final_col_list))

IN = household_ind_reg_input.copy()[final_col_list]

# 0:9 -> 9:12
# 12 : 21 -> 21 : 24
# 24 : 33 -> 33 : 36
# 36 : 45 -> 45 : 48
# 48 : 57 -> 57 : 60
# 60 : 69 -> 69 : 72
# 84 : 93
start_range = 0
end_range = 106

X= IN.copy().to_numpy().astype(float)

# SCALE CNN OUTPUTS
for j in range(start_range, end_range-1, 3):
    sum = np.exp(X[:,j])+np.exp(X[:,j+1])+np.exp(X[:,j+2])
    for c in range(3):
        X[:,j+c] = np.exp(X[:,j+c])/sum

X = pd.DataFrame(X, index=IN.index.copy(), columns = IN.columns)
#input_df = household_ind_input.join(X, how = "inner")
pickle.dump(X, open(final_df_path,'wb'))
#/content/drive/MyDrive/Scripts_and_Data_Transfer/3.Arch_2_Procedure/5.Train_Regression_and_Classification_Models/Inputs/Household_Indicators_Input/household_input_2001_new_final.pickle

t2 = time.time()
print("Total time : ", t2-t1)



