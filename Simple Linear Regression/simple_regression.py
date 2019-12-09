##AUTHOR: TARUN BISHT https://tarunbisht.herokuapp.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting Data Ready
data=pd.read_csv('salary.csv')

# shuffling data for random training and testing data
data = data.sample(frac=1).reset_index(drop=True)

training_size=0.7 # define training data size (value b/w 0 and 1)
sample_size=int(len(data)*training_size)

# splitting data into training and testing
train=data.iloc[:sample_size,:]
test=data.iloc[sample_size:,:]
# training data
trainX=train.iloc[:,0].values
trainY=train.iloc[:,1].values
#testing data
testX=test.iloc[:,0].values
testY=test.iloc[:,1].values

# plotting data before Regression (Review dataset)
plt.scatter(trainX,trainY,color="red")
plt.title("Salary Vs Experience Before Regression Applied")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

def cost_function(predicted,actual):
    return np.mean(np.power(predicted-actual,2))

def error(predicted,actual):
    return predicted-actual

# differentiation of cost wrt slope (variable 1)
def diff_cost_wrt_slope(predicted,actual):
    err=error(predicted,actual)
    return 2*np.sum(err*trainX)

# differentiation of cost wrt intercept (variable 2)
def diff_cost_wrt_intercept(predicted,actual):
    err=error(predicted,actual)
    return 2*np.sum(err)

# linear Regression Equation: Y=MX+C we have to find optimal value for M and C using gradient descent
m=0
c=0
training_epochs=5
learning_rate=0.001
cost=[]

def predict(data):
    return np.array([m*i+c for i in data])

# Training
for i in range(training_epochs):
    y_pred=predict(trainX)
    cost.append(cost_function(y_pred,trainY))
    m=m-(diff_cost_wrt_slope(y_pred,trainY)*learning_rate)
    c=c-(diff_cost_wrt_intercept(y_pred,trainY)*learning_rate)

predicted_train=predict(trainX)

# plotting cost and iterations
plt.plot(range(1,len(cost)+1),cost,color="red")
plt.title("Cost/ Error in every iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# plotting data after Regression
plt.scatter(trainX,trainY,color="red")
plt.plot(trainX,predicted_train,color="blue")
plt.title("Salary Vs Experience After Regression Applied ")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# prediction for test data
predicted_test=predict(testX)

# printing predicted and original values
print("PREDICTED\tORIGINAL")
for i in range(len(predicted_test)):
    print(f'{predicted_test[i]:.2f}\t{testY[i]}')

# test Data Visualization
plt.scatter(testX,testY,color="red")
plt.plot(testX,predicted_test,color="blue")
plt.title("Salary Vs Experience for Test Data ")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# measuring r2 score
def r2_score(original_data,predicted_data):
    data_length=len(original_data)
    original_data_mean=np.mean(original_data)
    numerator=np.sum(np.power(original_data-predicted_data,2))
    denominator=np.sum(np.power(original_data-original_data_mean,2))
    return 1-(numerator/denominator)

print("R2 Score for model is: ",r2_score(testY,predicted_test))
