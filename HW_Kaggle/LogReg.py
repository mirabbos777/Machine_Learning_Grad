import numpy
import csv
import time
import sys

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
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
file_name = "submission_{:0>4d}{:0>2d}{:0>2d}".format(now.tm_year, now.tm_mon, now.tm_mday) + \
            "T{:0>2d}{:0>2d}{:0>2d}".format(now.tm_hour, now.tm_min, now.tm_sec) + ".csv"
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


# Calculate Pearson correlation coefficient of y_train and every X_train's column
# Drop out columns of X_train which have too small (<0.005) correlation coefficient
delete_column_num = []
cov_matrix = numpy.cov(y_train, X_train, rowvar=False)
for i in range(1, cov_matrix.shape[0]):
    r = cov_matrix[0][i]/(numpy.std(y_train)*numpy.std(X_train[:, i-1]))
    if r < 0.005:
        delete_column_num.append(i-1)
# Delete columns of training data and testing data simultaneously
X_train = numpy.delete(X_train, delete_column_num, 1)
X_test = numpy.delete(X_test, delete_column_num, 1)


# Standardize both of training and testing data part by each feature (i.e., columns of matrix X_train and X_test)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # Apply "the same" standardizing "pattern" on testing data

'''
# Use SMOTE oversampling positive instances (y_train==1) into the same number of negative instances (y_train==0)
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)


# Apply PCA to matrix X_train (# of components >= 82 would be better)
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
'''

# Find optimal C (MAX AUC value) by training models using 5-fold cross-validation
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.2)
max_auc = 0
opt_param = 0
param_grid = {
                'C_list': numpy.arange(0.01, 1.0, 0.01).tolist()
}
param_grid = list(ParameterGrid(param_grid))
for i in param_grid:
    print("# of rounds: %d" % param_grid.index(i))
    clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=i['C_list'], max_iter=10000)
    clf.fit(X_train_train, y_train_train)
    fpr, tpr, skip = metrics.roc_curve(y_train_test, clf.predict_proba(X_train_test)[:, 1])
    temp_auc = metrics.auc(fpr, tpr)
    if max_auc < temp_auc:
        max_auc = temp_auc
        opt_param = i['C_list']

print("Optimal parameters: %f" % opt_param)
print("Maximum AUC: %f" % max_auc)

clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=opt_param, max_iter=10000)
clf.fit(X_train, y_train)
target_column = clf.predict_proba(X_test)[:, 1]

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
print("ExeTime: %.3f"%((time.time_ns()-start) / (10**9)))
#numpy.savetxt(outfptr, numpy.delete(X_train, delete_column_num, 1), '%s', ',')
