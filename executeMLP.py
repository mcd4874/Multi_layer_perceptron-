import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys



def sigmoid(sum):
    return 1/(1+np.exp(-sum))

def create_confusion_matrix(y,y_pred):
    print('Confusion Matrix')
    print(confusion_matrix(y, y_pred))
    print('Classification Report')
    t = ["Bolt","Nut","ring","Scrap"]
    labels = np.array(t)
    print(classification_report(y, y_pred, target_names=labels))

def evaluate(y,predict):
    accurate = 0
    for i in range(len(predict)):
        if predict[i] == y[i][0]:
            accurate+=1
    print "the accuracy is :",(accurate*1.0/len(predict)),"% "

def calculate_profit(y,predict,profitTable):
    totalProfit = 0
    for i in range(len(predict)):
        totalProfit+= profitTable[y[i][0]-1][int(predict[i])-1]
    print "the total profit is : ",totalProfit


def predict_result(test_input,W):
    # do prediction
    W1 = W['W1']
    W1 = np.array(W1)
    b1 = W['b1']
    b1 = np.array(b1)
    W2 = W['W2']
    W2 = np.array(W2)
    b2 = W['b2']
    b2 = np.array(b2)


    Z1 = test_input.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
    result = np.argmax(A2, axis=1).astype(int)
    result = np.add(result, 1)
    return result


def produce_result_predictions(data,predict,W):
    # generate decision bound graph

    columns1 = ["X1", "X2"]
    X = data[columns1].values
    y = data["Y"].values

    marker = {1: 'x', 2: 'o', 3: '^', 4: 'v'}
    colors = ['red', 'blue', 'lightgreen', 'black', 'cyan']

    cmap = ListedColormap(colors[:len(np.unique(y))])


    # Define our usual decision surface bounding plots
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                         , np.arange(y_min, y_max, h))

    k = np.c_[xx.ravel(), yy.ravel()]

    Z = predict_result(k,W)
    Z = Z.reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    ax_c = f.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    cdict = {1: 'red', 2: 'blue', 3: 'green',4:'yellow'}
    actual_labels = {1:'Bolt actual',2:'Nut actual',3:'Ring actual',4:'Scrap actual'}


    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(test_input['X1'].values[ix], test_input['X2'].values[ix], c=cdict[g], label=actual_labels[g],cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1,marker = marker[g] )


    ax.set(aspect="equal",
           xlim=(x_min, x_max), ylim=(y_min, y_max),
           xlabel="$X_1$", ylabel="$X_2$")
    plt.legend()

    plt.savefig("decision_bound.png")
    plt.show()


profit_tablepro = [[ 20, -7,-7,-7],
          [-7,15,-7,-7],
          [-7,-7,5,-7],
          [-3,-3,-3,-3]
]


# test file
fileName = sys.argv[1]
# weight file for neural network
wFileName = sys.argv[2]
# load test input
test_input = pd.read_csv(fileName,names = ["X1","X2","Y"])

test_data = test_input.iloc[:,0:2].values
test_labels = test_input.iloc[:,2:3].values

weightFile = wFileName
f = open(weightFile,'r')
W = data = json.load(f)
f.close()



# calculate prediction
y_pred = predict_result(test_data,W)

# calculate accuracy
evaluate(test_labels,y_pred)

# make confusion matrix
create_confusion_matrix(test_labels,y_pred)

# calculte profit for prediction
calculate_profit(test_labels,y_pred,profit_tablepro)
# generate decision boundary
produce_result_predictions(test_input,y_pred,W)
