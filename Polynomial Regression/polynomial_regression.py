##AUTHOR: TARUN BISHT https://tarunbisht.herokuapp.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting Data Ready
data=pd.read_csv('salaries.csv')
data.insert(loc=1,column="Ones",value=1)

# training data
trainX=data.iloc[:,1:3].values
trainY=data.iloc[:,-1].values

# setting data for polynomial regression
polynomial_degree=5
for i in range(polynomial_degree-1):
    temp=(np.power(trainX[0:,1],polynomial_degree-i)).reshape(-1,1)
    trainX=np.concatenate((trainX,temp),axis=1)

# Feature Scaling of data (Feature scaling improves the convergence of steepest descent algorithm)
meanX=np.mean(trainX)
stdX=np.std(trainX)
trainX=(trainX-meanX)/stdX
meanY=np.mean(trainY)
stdY=np.std(trainY)
trainY=(trainY-meanY)/stdY

def cost_function(predicted,actual):
    error=predicted-actual
    return np.mean(np.dot(error.T,error))

def error(predicted,actual):
    return predicted-actual

# differentiating cost wrt theta
def diff_cost_wrt_theta(predicted,actual):
    err=error(predicted,actual)
    return 2*np.dot(err.T,trainX)

theta=np.zeros(shape=trainX.shape[1])
training_epochs=1000
learning_rate=0.001
cost=[]

def predict(data):
    return np.dot(data,theta.T)

# Training Loop
for i in range(training_epochs):
    y_pred=predict(trainX)
    cost.append(cost_function(y_pred,trainY))
    theta=theta-learning_rate*diff_cost_wrt_theta(y_pred,trainY)

predicted_train=predict(trainX)

# plotting cost and iterations
plt.plot(range(1,len(cost)+1),cost,color="red")
plt.title("Cost/ Error in every iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# plotting data after Regression
plt.scatter(trainX[0:,1]*stdX+meanX,trainY*stdY+meanY,color="red")
plt.plot(trainX[0:,1]*stdX+meanX,predicted_train*stdY+meanY,color="blue")
plt.title("Salary Vs Level After Regression Applied ")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# printing predicted and original values
print("PREDICTED\tORIGINAL")
for i in range(len(predicted_train)):
    ## for inverse scaling the values to display scaled values
    print(f'{(predicted_train[i]*stdY)+meanY:.2f}\t{(trainY[i]*stdY)+meanY:.2f}')

# measuring r2 score
def r2_score(original_data,predicted_data):
    data_length=len(original_data)
    original_data_mean=np.mean(original_data)
    numerator=np.sum(np.power(original_data-predicted_data,2))
    denominator=np.sum(np.power(original_data-original_data_mean,2))
    return 1-(numerator/denominator)

print("R2 Score for model is: ",r2_score(trainY,predicted_train))
