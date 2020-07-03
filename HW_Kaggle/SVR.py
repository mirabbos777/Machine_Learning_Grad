import numpy
import csv
import time
import sys

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVR
from sklearn import metrics


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
            "_SVR_Mode" + str(sys.argv[1]) + ".csv"
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
#           Drop out columns of X_train which have too small (<0.05) correlation coefficient
# (Mode 2)  Delete useless columns by LB probing 
#           Keep only important features found by LB probing
# (Mode 3)  Do nothing
if int(sys.argv[1]) == 1:  # Mode 1
    delete_column_num = []
    cov_matrix = numpy.cov(y_train, X_train, rowvar=False)
    for i in range(1, cov_matrix.shape[0]):
        r = cov_matrix[0][i]/(numpy.std(y_train)*numpy.std(X_train[:, i-1]))
        if abs(r) < 0.05:
            delete_column_num.append(i-1)
    # Delete columns of training data and testing data simultaneously
    X_train = numpy.delete(X_train, delete_column_num, 1)
    X_test = numpy.delete(X_test, delete_column_num, 1)
elif int(sys.argv[1]) == 2:  # Mode 2
    remain_column = [16, 29, 33, 45, 63, 65, 70, 73, 91, 106, 108, 117, 132, 164, 189, 199, 209, 217, 239]
    delete_column = [i for i in range(0, 300) if i not in remain_column]
    X_train = numpy.delete(X_train, delete_column, 1)
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
                'Kernel_list':      ['poly'],
                'Deg_list':         numpy.array(range(1, 3)).tolist(),
                'Gamma_list':       numpy.arange(0.01, 0.5, 0.01).tolist(),
                'C_list':           numpy.arange(0.01, 0.1, 0.01).tolist(),
                'Epsilon_list':     numpy.arange(0.01, 0.5, 0.01).tolist()
}
param_grid = list(ParameterGrid(param_grid))
for i in param_grid:
    print(param_grid.index(i))
    svr = SVR(kernel=i['Kernel_list'], degree=i['Deg_list'], gamma=i['Gamma_list'], C=i['C_list'], max_iter=-1)
    svr.fit(X_train_train, y_train_train)
    temp_result = svr.predict(X_train_test)
    fpr, tpr, skip = metrics.roc_curve(y_train_test, temp_result)
    temp_auc = metrics.auc(fpr, tpr)
    if max_auc < temp_auc:
        max_auc = temp_auc
        opt_param = i

print("Optimal parameters:")
print(opt_param)


# Train model by optimal parameters
svr = SVR(kernel=opt_param['Kernel_list'], degree=opt_param['Deg_list'], gamma=opt_param['Gamma_list'], C=opt_param['C_list'], max_iter=-1)
svr.fit(X_train, y_train)
target_column = svr.predict(X_test)


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
