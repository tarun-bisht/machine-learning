##AUTHOR: TARUN BISHT https://tarunbisht.herokuapp.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting Data Ready
data=pd.read_csv('ads.csv')
data.insert(loc=2,column="Ones",value=1)

# shuffling data for random training and testing data
data = data.sample(frac=1).reset_index(drop=True)

training_size=0.7 # define training data size (value b/w 0 and 1)
sample_size=int(len(data)*training_size)

#splitting data into training and testing
train=data.iloc[:sample_size,:]
test=data.iloc[sample_size:,:]

# training data
trainX=train.iloc[:,2:5].values
trainY=train.iloc[:,-1].values

#testing data
testX=test.iloc[:,2:5].values
testY=test.iloc[:,-1].values

# Feature Scaling of data (Feature scaling improves the convergence of steepest descent algorithm)
meanX=np.mean(trainX)
stdX=np.std(trainX)
trainX=(trainX-meanX)/stdX
testX=(testX-meanX)/stdX

# Cost function (J=-y*log(p)-(1-y)*log(1-p))
def cost_function(predicted,actual):
    cost_1=-actual*np.log(predicted)
    cost_0=(1-actual)*np.log(1-predicted)
    return np.mean(cost_1-cost_0)

def error(predicted,actual):
    return predicted-actual

# differentiating cost wrt theta
def diff_cost_wrt_theta(predicted,actual):
    err=error(predicted,actual)
    return np.dot(err.T,trainX)

# Logistic Linear Regression Equation: Y=sigmoid(theta.T*X) (where theta.T=Transpose of theta matrix)
theta=np.zeros(shape=trainX.shape[1])
training_epochs=100
learning_rate=0.001
cost=[]

# sigmoid function f(x)=1/1+e^-x
def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(data):
    return sigmoid(np.dot(data,theta.T))

def classify(prediction,db=0.5): #db = decision boundary
    decision=np.vectorize(lambda p: 1 if p >= db else 0)
    return decision(prediction)

# Training Loop
for i in range(training_epochs):
    y_pred=predict(trainX)
    cost.append(cost_function(y_pred,trainY))
    theta=theta-learning_rate*diff_cost_wrt_theta(y_pred,trainY)

predicted_train=predict(trainX)
predicted_class_train=classify(predicted_train)

# plotting cost and iterations
plt.plot(range(1,len(cost)+1),cost,color="red")
plt.title("Cost/ Error in every iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# prediction for test data
predicted_test=predict(testX)
predicted_class_test=classify(predicted_test)

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

print("Confusion Matrix for training data")
conf_matrix=confusion_matrix(predicted_class_train,trainY)
print(conf_matrix)

print("Confusion Matrix for testing data")
conf_matrix=confusion_matrix(predicted_class_test,testY)
print(conf_matrix)
print("Sensitivity Metric: ",sensitivity(predicted_class_test,testY))
print("Specificity Metric: ",specificity(predicted_class_test,testY))
