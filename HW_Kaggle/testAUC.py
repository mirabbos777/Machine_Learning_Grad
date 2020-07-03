import numpy
from sklearn import metrics 

y_1 = numpy.array([10, 20, 30, 40, 50])
y_2 = numpy.array([1, 2, 3, 4, 5])
y_true = numpy.array([0, 1, 0, 1, 1])

fpr, tpr, tt = metrics.roc_curve(y_true, y_1)
roc_auc1 = metrics.auc(fpr,tpr)

fpr, tpr, tt = metrics.roc_curve(y_true, y_2)
roc_auc2 = metrics.auc(fpr,tpr)

print(roc_auc1)
print(roc_auc2)