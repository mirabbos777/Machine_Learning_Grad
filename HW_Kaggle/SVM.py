import numpy
import csv
import time
# import sys
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC, SVR
from sklearn import metrics


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


# Calculate Pearson correlation coefficient of y and every X's column
# Drop out columns of X which have too small (<0.01) corr. coef.
delete_column_num = []
cov_matrix = numpy.cov(y, X, rowvar=False)
for i in range(1, cov_matrix.shape[0]):
    r = cov_matrix[0][i]/(numpy.std(y)*numpy.std(X[:, i-1]))
    if r < 0.005:
        delete_column_num.append(i-1)
# Delete columns of training data and testing data simultaneously  
X = numpy.delete(X, delete_column_num, 1)
X_test = numpy.delete(X_test, delete_column_num, 1)


# Standardize both of training and testing data part by each feature (i.e., columns of matrix X and X_test)
ss = StandardScaler()
X = ss.fit_transform(X)
X_test = ss.transform(X_test) # Apply "the same" standardizing "pattern" on testing data


# Use SMOTE oversampling positive instances (y==1) into the same number of negative instances (y==0)
ROS = RandomOverSampler(random_state=0)
X, y = ROS.fit_resample(X, y)
shuffler = numpy.random.permutation(X.shape[0])
X = X[shuffler]
y = y[shuffler]


# Apply PCA to matrix X (# of compoenets >= 82 would be better)
pca = PCA()
X = pca.fit_transform(X)
X_test = pca.transform(X_test)

'''
# Apply kernel PCA (RBF kernel) to matrix X
kpca = KernelPCA(kernel='rbf', n_jobs=-1)
X_KPCA = kpca.fit_transform(X)
'''


# Find optimal C (MAX AUC value) by training models using 5-fold cross-validation
fold_num = 5
max_auc = 0
opt_C = 0
opt_Eps = 0
C_list = numpy.arange(0.1, 1, 0.1)
Eps_list = numpy.arange(0.1, 1, 0.1)
# Grid search best C and epsilon value of linear SVR
for cur_C in C_list:
    for cur_Eps in Eps_list:
        local_svm = SVR(C=round(cur_C, 3), epsilon=round(cur_Eps, 3), kernel='linear')
        avg_auc = 0

        for j in range(0, fold_num):
            test_index = numpy.array(range(j*(X.shape[0]//fold_num), (j+1)*(X.shape[0]//fold_num)))
            train_index = numpy.array(range(0, X.shape[0]))
            train_index = numpy.delete(train_index, test_index, None)

            local_svm.fit(X[train_index, :], y[train_index])
            y_predict = local_svm.predict(X[test_index, :])
            fpr, tpr, skip = metrics.roc_curve(y[test_index], y_predict)
            avg_auc = avg_auc + (metrics.auc(fpr,tpr) - avg_auc) / (j + 1) # Incrementally updating avg.

        if avg_auc > max_auc:
            max_auc = avg_auc
            opt_C = cur_C
            opt_Eps = cur_Eps


print("%f %f %f"%(opt_C, opt_Eps, max_auc))
# Predict testing data using model which is trained by all trainnig data (not k-fold subset)
svm = SVR(C=opt_C, epsilon=opt_Eps, kernel='linear')
svm.fit(X, y)

title_line = numpy.array(["id", "target"])
title_line = numpy.reshape(title_line, (1, 2))

id_column = numpy.array(range(250, 20000), dtype='U')
target_column = svm.predict(X_test)

numpy.savetxt(outfptr, title_line, '%s', ',')
numpy.savetxt(outfptr, numpy.c_[id_column, target_column], '%s', ',')

infptr_1.close()
infptr_2.close()
outfptr.close()


# Print program executing time
print("ExeTime: %.3f"%((time.time_ns()-start) / (10**9)))
#numpy.savetxt(outfptr, numpy.delete(X, delete_column_num, 1), '%s', ',')