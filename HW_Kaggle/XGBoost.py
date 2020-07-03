import numpy
import csv
import time
import sys

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import metrics
import xgboost


# Print full numpy array without truncation
numpy.set_printoptions(threshold=sys.maxsize)


# Calculate program executing time
start = time.time_ns()


# Open input csv file
infptr_1 = open('train.csv', newline='')
infptr_2 = open('test.csv', newline='')


# Open output csv file
now = time.localtime(time.time())
file_name = "{:0>4d}{:0>2d}{:0>2d}".format(now.tm_year, now.tm_mon, now.tm_mday) + \
            "T{:0>2d}{:0>2d}{:0>2d}".format(now.tm_hour, now.tm_min, now.tm_sec) + \
            "_XGBoost_Mode" + str(sys.argv[1]) + ".csv"
outfptr = open(file_name, 'w', newline='')


# Read training data into two matrix 'X_train' and 'y_train'
# 'X_train' : observations' data part
# 'y_train' : observations' label part
rows = list(csv.reader(infptr_1))
X_train = []
y_train = []
for i in range(1, len(rows)):
    X_train.append(rows[i][2:])
    y_train.append(rows[i][1])


# Read testing data into two matrix 'X_test'
rows = list(csv.reader(infptr_2))
X_test = []
for i in range(1, len(rows)):
    X_test.append(rows[i][1:])


# Convert list into numpy array
X_train = numpy.array(X_train, dtype='float')
y_train = numpy.array(y_train, dtype='float')
X_test = numpy.array(X_test, dtype='float')


# (Mode 1)  Delete useless columns by calculating Pearson r
#           Calculate Pearson correlation coefficient of y_train and every X_train's column
#           Drop out columns of X_train which have too small (<0.005) correlation coefficient
# (Mode 2)  Delete useless columns by LB probing 
#           Keep only important features found by LB probing
# (Mode 3)  Do nothing
if sys.argv[1] == 1:  # Mode 1
    delete_column_num = []
    cov_matrix = numpy.cov(y_train, X_train, rowvar=False)
    for i in range(1, cov_matrix.shape[0]):
        r = cov_matrix[0][i]/(numpy.std(y_train)*numpy.std(X_train[:, i-1]))
        if r < 0.01:
            delete_column_num.append(i-1)
    # Delete columns of training data and testing data simultaneously
    X_train = numpy.delete(X_train, delete_column_num, 1)
    X_test = numpy.delete(X_test, delete_column_num, 1)
elif sys.argv[1] == 2:  # Mode 2
    remain_column = [16, 29, 33, 45, 63, 65, 70, 73, 91, 106, 108, 117, 132, 164, 189, 199, 209, 217, 239]
    delete_column = [i for i in range(0, 300) if i not in remain_column]
    X = numpy.delete(X, delete_column, 1)
    X_test = numpy.delete(X_test, delete_column, 1)


# Standardize both of training and testing data part by each feature (i.e., columns of matrix X_train and X_test)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # Apply "the same" standardizing "pattern" on testing data


# Use SMOTE oversampling positive instances (y_train==1) into the same number of negative instances (y_train==0)
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)


# Find optimal parameters of training models using 5-fold cross-validation
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.2)
max_auc = 0
opt_param = 0
param_grid = {
                'Eta_list':         numpy.arange(0.0,  0.5,  0.1).tolist(),  # 50
                'Lambda_list':      numpy.arange(5.0,  10.0, 1.0).tolist(),   # 10
                'Gamma_list':       numpy.arange(20.0, 25.0, 1.0).tolist(),   # 10
                'Child_weight':     numpy.arange(0.0,  5.0, 1.0).tolist()    # 10
}
param_grid = list(ParameterGrid(param_grid))
for i in param_grid:
    param_instance = {
                    'objective':         'reg:logistic',
                    'tree_method':       'exact',
                    'max_depth':         5,
                    'eta':               i['Eta_list'],
                    'lambda':            i['Lambda_list'],
                    'gamma':             i['Gamma_list'],
                    'min_child_weight':  i['Child_weight'],
    }
    print("# of rounds: %d" % param_grid.index(i))
    xgb_train_data = xgboost.DMatrix(X_train_train, label=y_train_train)
    xgb_test_data = xgboost.DMatrix(X_train_test)
    
    print(xgb_train_data.feature_names)
    
    exit()
    xgb = xgboost.train(param_instance, xgb_train_data, num_boost_round=10)
    temp_result = xgb.predict(xgb_test_data)
    fpr, tpr, skip = metrics.roc_curve(y_train_test, temp_result)
    temp_auc = metrics.auc(fpr, tpr)
    if max_auc < temp_auc:
        max_auc = temp_auc
        opt_param = param_instance

print("Optimal parameters:")
print(opt_param)


# Train model by optimal parameters
XGB_train_data = xgboost.DMatrix(X_train, label=y_train)
XGB_test_data = xgboost.DMatrix(X_test)
XGB = xgboost.train(opt_param, XGB_train_data, num_boost_round=10)
target_column = xgb.predict(XGB_test_data)


# Output results into a csv file
title_line = numpy.array(["id", "target"])
title_line = numpy.reshape(title_line, (1, 2))

id_column = numpy.array(range(250, 20000), dtype='U')

numpy.savetxt(outfptr, title_line, '%s', ',')
numpy.savetxt(outfptr, numpy.c_[id_column, target_column], '%s', ',')

infptr_1.close()
infptr_2.close()
outfptr.close()


# Print program executing time
print("CSV file name: " + file_name)
print("ExeTime: %.3f" % ((time.time_ns()-start) / (10**9)))
