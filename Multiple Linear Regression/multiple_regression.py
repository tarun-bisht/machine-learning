##AUTHOR: TARUN BISHT https://tarunbisht.herokuapp.com
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting Data Ready
data=pd.read_csv('startup.csv')
data.insert(loc=0,column="Ones",value=1)

# shuffling data for random training and testing data
data = data.sample(frac=1).reset_index(drop=True)

training_size=0.7 # define training data size (value b/w 0 and 1)
sample_size=int(len(data)*training_size)

#splitting data into training and testing
train=data.iloc[:sample_size,:]
test=data.iloc[sample_size:,:]

# training data
trainX=train.iloc[:,:4].values
trainY=train.iloc[:,-1].values

#testing data
testX=test.iloc[:,:4].values
testY=test.iloc[:,-1].values

# Feature Scaling of data (Feature scaling improves the convergence of steepest descent algorithm)
meanX=np.mean(trainX)
stdX=np.std(trainX)
trainX=(trainX-meanX)/stdX
testX=(testX-meanX)/stdX

# scaling target variables because its values are very large and it can possible to get a floating point overflow (inf)
meanY=np.mean(trainY)
stdY=np.std(trainY)
trainY=(trainY-meanY)/stdY
testY=(testY-meanY)/stdY

def cost_function(predicted,actual):
    error=predicted-actual
    return np.mean(np.dot(error.T,error))

def error(predicted,actual):
    return predicted-actual

# differentiating cost wrt theta
def diff_cost_wrt_theta(predicted,actual):
    err=error(predicted,actual)
    return 2*np.dot(err.T,trainX)

# Multiple Linear Regression Equation: Y=theta.T*X (where theta.T=Transpose of theta matrix)
theta=np.zeros(shape=trainX.shape[1])
training_epochs=100
learning_rate=0.005
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

# prediction for test data
predicted_test=predict(testX)

# printing predicted and original values
print("PREDICTED\tORIGINAL")
for i in range(len(predicted_test)):
    ## for inverse scaling the values to display scaled values
    print(f'{(predicted_test[i]*stdY)+meanY:.2f}\t{(testY[i]*stdY)+meanY:.2f}')

# measuring r2 score
def r2_score(original_data,predicted_data):
    data_length=len(original_data)
    original_data_mean=np.mean(original_data)
    numerator=np.sum(np.power(original_data-predicted_data,2))
    denominator=np.sum(np.power(original_data-original_data_mean,2))
    return 1-(numerator/denominator)

print("R2 Score for model is: ",r2_score(testY,predicted_test))
