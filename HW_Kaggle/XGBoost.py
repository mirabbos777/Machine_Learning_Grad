import numpy
import csv
import time
import sys

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn import metrics
import xgboost


# Calculate program executing time
start = time.time_ns()


# Open input csv file
infptr_1 = open('train.csv', newline='')
infptr_2 = open('test.csv', newline='')


# Open output csv file
now = time.localtime(time.time())
file_name = "submission_{:0>4d}{:0>2d}{:0>2d}".format(now.tm_year, now.tm_mon, now.tm_mday) + \
            "T{:0>2d}{:0>2d}{:0>2d}".format(now.tm_hour, now.tm_min, now.tm_sec) + ".csv"
outfptr = open(file_name, 'w', newline='')
print("CSV file name: " + file_name)


# Read training data into two matrix 'X' and 'y'
# 'X' : observations' data part
# 'y' : observations' label part
rows = list(csv.reader(infptr_1))
X = []
y = []
for i in range(1, len(rows)):
    X.append(rows[i][2:])
    y.append(rows[i][1])


# Read testing data into two matrix 'X_test'
rows = list(csv.reader(infptr_2))
X_test = []
for i in range(1, len(rows)):
    X_test.append(rows[i][1:])


# Convert list into numpy array
X = numpy.array(X, dtype='float')
y = numpy.array(y, dtype='float')
X_test = numpy.array(X_test, dtype='float')


# Keep only important features found by LB probing
remain_column = [16, 29, 33, 45, 63, 65, 70, 73, 91, 106, 108, 117, 132, 164, 189, 199, 209, 217, 239]
delete_column = [i for i in range(0, 300) if i not in remain_column]
X = numpy.delete(X, delete_column, 1)
X_test = numpy.delete(X_test, delete_column, 1)


# Standardize both of training and testing data part by each feature (i.e., columns of matrix X and X_test)
ss = StandardScaler()
X = ss.fit_transform(X)
X_test = ss.transform(X_test)  # Apply "the same" standardizing "pattern" on testing data


# Use SMOTE oversampling positive instances (y==1) into the same number of negative instances (y==0)
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
shuffler = numpy.random.permutation(X.shape[0])
X = X[shuffler]
y = y[shuffler]


# Apply PCA to matrix X (# of components >= 82 would be better)
pca = PCA()
X = pca.fit_transform(X)
X_test = pca.transform(X_test)


# Find optimal C (MAX AUC value) by training models using 5-fold cross-validation
fold_num = 5
max_auc = 0
opt_param = 0
param_grid = {
                'Eta_list':         numpy.arange(0.0,   0.5,    0.01).tolist(),  # 50
                'Lambda_list':      numpy.arange(5.0,   10.0,   0.5).tolist(),   # 10
                'Gamma_list':       numpy.arange(20.0,  30.0,   1.0).tolist(),   # 10
                'Child_weight':     numpy.arange(0.0,   10.0,   1.0).tolist()    # 10
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
                    'max_delta_step':    2,
                    'subsample':         0.5
    }
    print("# of rounds: %d" % param_grid.index(i))
    avg_auc = 0
    for j in range(0, fold_num):
        test_index = numpy.array(range(j*(X.shape[0]//fold_num), (j+1)*(X.shape[0]//fold_num)))
        train_index = numpy.array(range(0, X.shape[0]))
        train_index = numpy.delete(train_index, test_index, None)
        
        data_train = xgboost.DMatrix(X[train_index, :], label=y[train_index])
        data_test = xgboost.DMatrix(X[test_index, :])
        
        XGB = xgboost.train(param_instance, data_train, num_boost_round=10)
        
        y_predict = XGB.predict(data_test)
        fpr, tpr, skip = metrics.roc_curve(y[test_index], y_predict)
        avg_auc = avg_auc + (metrics.auc(fpr,tpr) - avg_auc) / (j + 1)  # Incrementally updating avg.
    if avg_auc > max_auc:
        max_auc = avg_auc
        opt_param = param_instance


data_train = xgboost.DMatrix(X, label=y)
data_test = xgboost.DMatrix(X_test)

print("Optimal parameters:")
print(opt_param)
print(max_auc)

XGB = xgboost.train(opt_param, data_train, num_boost_round=10)
target_column = XGB.predict(data_test)

title_line = numpy.array(["id", "target"])
title_line = numpy.reshape(title_line, (1, 2))

id_column = numpy.array(range(250, 20000), dtype='U')

numpy.savetxt(outfptr, title_line, '%s', ',')
numpy.savetxt(outfptr, numpy.c_[id_column, target_column], '%s', ',')

infptr_1.close()
infptr_2.close()
outfptr.close()

# Print program executing time
print("ExeTime: %.3f"%((time.time_ns()-start) / (10**9)))
#numpy.savetxt(outfptr, numpy.delete(X, delete_column_num, 1), '%s', ',')
