import numpy as np
import pandas as pd
import json

# Make it work for Python 2+3 and with Unicode
import sys
import matplotlib.pyplot as plt

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def sigmoid(sum):
    return 1/(1+np.exp(-sum))

def rmse(predictions, targets):

    differences = predictions - targets                       #the DIFFERENCEs.

    differences_squared = differences ** 2                    #the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^

    return rmse_val

epochs = 10001

trainFileName = sys.argv[1]
data_input = pd.read_csv(trainFileName,names = ["X1","X2","Y"])
training_data = data_input.iloc[:,0:2].values
# print training_data
training_labels = data_input.iloc[:,2:3].values
training_labels = np.subtract(training_labels,1)
training_labels_onehot = np.zeros((training_labels.shape[0], 4)).astype(int)
for i in range (training_labels.shape[0]):
    training_labels_onehot[i, training_labels[i]] = 1




X_train = training_data
n_features = X_train.shape[1]
m = X_train.shape[0]
n_h = 5
n_output = 4

learning_rate = 0.1

W1 = np.random.normal(0, 1,[n_features, n_h])
b1 = np.zeros((1, n_h))
W2 = np.random.normal(0, 1,[n_h, n_output])
b2 = np.zeros((1, n_output))

X = X_train
Y = training_labels_onehot
mses = np.zeros(epochs)
for i in range(epochs):

    Z1 = X.dot(W1) + b1

    # print "shape of bias 1: ",b1.shape
    # print "shape of bias 2: ", b2.shape

    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)

    # record mean square error
    mses[i] = rmse(A2,Y)

    cost = cross_entropy_softmax_loss_array(A2,Y)

    # print cost

    dZ2 = A2-Y
    dW2 = (1./m) * A1.T.dot(dZ2)
    db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)
    # print "shape of db2 : ",db2.shape

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) *  X.T.dot(dZ1)
    db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)
    # print "shape of db1 : ", db1.shape

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (i in [0, 10, 100, 1000, 10000]):
        w1 =W1.tolist()
        bias1 = b1.tolist()
        w2 = W2.tolist()
        bias2 = b2.tolist()
        data = {'W1':w1,'W2':w2,'b1':bias1,'b2':bias2}
        fileName = 'weightFile_'+str(i)+'.json'
        with open(fileName, 'w') as outfile:
            json.dump(data, outfile)






plt.plot(mses)
plt.title("MSE over epochs")
plt.xlabel("epochs #")
plt.ylabel("MSE")
plt.savefig("MSE.png")
plt.show()

