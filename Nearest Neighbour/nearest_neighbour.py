import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from kdtree import kd_tree

# Getting Data Ready
data=pd.read_csv('ads.csv')

training_size=0.7 # define training data size (value b/w 0 and 1)
sample_size=int(len(data)*training_size)

#splitting data into training and testing
train=data.iloc[:sample_size,:]
test=data.iloc[sample_size:,:]

# training data
trainX=train.iloc[:,2:4].values
trainY=train.iloc[:,-1].values

#testing data
testX=test.iloc[:,2:4].values
testY=test.iloc[:,-1].values

# Creating KDTree from training data
tree=kd_tree(dimension=2) # since our training data has 2 dimensions
tree.insert(trainX,trainY)

# Value of K in K nearest Neighbour
k=3

predicted_test=np.array([])
for x in testX:
    predicted_test=np.append(predicted_test,tree.get_point_category(x,k=k))

# Confusion Matrix
def confusion_matrix(pred_cls,actual_cls):
    '''
    true positive |  false positive|
    false negative|  true negative |
    '''
    conf=np.zeros(shape=(2,2))
    conf[0,0]=np.sum((pred_cls==1) & (actual_cls==1))
    conf[0,1]=np.sum((pred_cls==1) & (actual_cls==0))
    conf[1,0]=np.sum((pred_cls==0) & (actual_cls==1))
    conf[1,1]=np.sum((pred_cls==0) & (actual_cls==0))
    return conf

# Sensitivity and Specificity Metrices
def sensitivity(pred_cls,actual_cls): # tells what % of positive classes are correctly identified
    true_positive=np.sum((pred_cls==1) & (actual_cls==1))
    false_negative=np.sum((pred_cls==0) & (actual_cls==1))
    return 100*true_positive/(true_positive+false_negative)

def specificity(pred_cls,actual_cls): # tells what % of negative classes are correctly identified
    false_positive=np.sum((pred_cls==0) & (actual_cls==0))
    true_negative=np.sum((pred_cls==1) & (actual_cls==0))
    return 100*false_positive/(false_positive+true_negative)

print("Confusion Matrix for testing data")
conf_matrix=confusion_matrix(predicted_test,testY)
print(conf_matrix)
print("Sensitivity Metric: ",sensitivity(predicted_test,testY))
print("Specificity Metric: ",specificity(predicted_test,testY))
