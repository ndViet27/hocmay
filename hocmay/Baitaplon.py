# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pla
import logistic

file_path = r"D:\hocmay\test.xlsx"
data = pd.read_excel(file_path)
df = pd.DataFrame (data)
y_true = df.iloc[:,3].values.tolist()
dem = 0
y=[]
for i in df.iloc[:,2]:
    if i < 7.84 :
        x = np.array([1,df.iloc[dem,2]])
        y.append(logistic.logistic(x))
    else:
        x = np.append(1,df.iloc[dem,:2])
        y.append(pla.perceptron(x))
    dem+=1

def tp1(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] == 1:
            if y[i] == y_true[i]:
                count +=1
    return count
def fp1(y,y_true):
    count = 0
    for i in range (len(y)):
        if y_true[i] != 1:
            if y[i] != y_true[i]:
                count +=1
    return count
def tn1(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] != 1:
            if y[i] == y_true[i]:
                count +=1
    return count
def fn1(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] == 1:
            if y[i] != y_true[i]:
                count +=1
    return count
def tp0(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] == 0:
            if y[i] == y_true[i]:
                count +=1
    return count
def fp0(y,y_true):
    count = 0
    for i in range (len(y)):
        if y_true[i] != 0:
            if y[i] != y_true[i]:
                count +=1
    return count
def tn0(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] != 0:
            if y[i] == y_true[i]:
                count +=1
    return count
def fn0(y,y_true):
    count = 0
    for i in range(len(y)):
        if y_true[i] == 0:
            if y[i] != y_true[i]:
                count +=1
    return count
    
def accuracy(y,y_true):
    count = 0
    for i in range(len(y)):
        if y[i] == y_true[i]:
            count += 1
    return (count/len(y))

def precision(tp,fp):
    return tp/(tp+fp)

def recall(tp,fn):
    return tp/(tp+fn)

def f1(precision,recall):
    return 2/(1/(precision) + 1/(recall))
    
print("accuracy:",accuracy(y,y_true),"\n")
print("\t precision\t         recall\t                 f1")
print("1\t",precision(tp1(y,y_true),fp1(y,y_true)),"\t",recall(tp1(y,y_true),fn1(y,y_true)),"\t",f1(precision(tp1(y,y_true),fp1(y,y_true)),recall(tp1(y,y_true),fn1(y,y_true))))
print("0\t",precision(tp0(y,y_true),fp0(y,y_true)),"\t",recall(tp0(y,y_true),fn0(y,y_true)),"\t",f1(precision(tp0(y,y_true),fp0(y,y_true)),recall(tp0(y,y_true),fn0(y,y_true))))

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_true, y)
import seaborn as sns

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()
