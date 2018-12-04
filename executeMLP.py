import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

profit_tablepro = [[ 20, -7,-7,-7],
          [-7,15,-7,-7],
          [-7,-7,5,-7],
          [-3,-3,-3,-3]
]

def sigmoid(sum):
    return 1/(1+np.exp(-sum))

def create_confusion_matrix(y,y_pred):
    print('Confusion Matrix')
    print(confusion_matrix(y, y_pred))
    print('Classification Report')
    # target_names = ['Cats', 'Dogs', 'Horse']
    t = ["1","2","3","4"]
    labels = np.array(t)
    print(classification_report(y, y_pred, target_names=labels))

def evaluate(y,predict):
    accurate = 0
    for i in range(len(predict)):
        if predict[i] == y[i][0]:
            # print "yes"
            accurate+=1
    print "the accuracy is :",(accurate*1.0/len(predict)),"% "

def calculate_profit(y,predict,profitTable):
    totalProfit = 0
    for i in range(len(predict)):
        # print predict[i]
        # print y[i][0]

        totalProfit+= profitTable[y[i][0]-1][int(predict[i])-1]
    print "the total profit is : ",totalProfit


def generate_plot(X,y,predict):
    cdict = {1: 'red', 2: 'blue', 3: 'green',4:'yellow'}
    predict_labels = {1:'Bolt predict',2:'Nut predict',3:'Ring predict',4:'Scrap predict'}
    actual_labels = {1:'Bolt actual',2:'Nut actual',3:'Ring actual',4:'Scrap actual'}
    marker = {1:'x',2:'o',3:'^',4:'v'}
    fig, ax = plt.subplots()
    # print np.unique(predict)
    y = y.reshape(len(y))
    # print y.shape
    for g in np.unique(predict):
        # print "current G : ",g
        ix = np.where(predict == g)
        # print ix
        ax.scatter(X['X1'].values[ix], X['X2'].values[ix], c=cdict[g], label=predict_labels[g])


    for t in np.unique(y):
        # print "current G : ",t
        ix = np.where(y == t)
        # print ix
        ax.scatter(X['X1'].values[ix], X['X2'].values[ix], marker=marker[t] , c = cdict[t], label=actual_labels[t], s=100)

    ax.legend()
    plt.show()

def predict_result(test_input,W):
    W1 = W['W1']
    W1 = np.array(W1)
    b1 = W['b1']
    b1 = np.array(b1)
    W2 = W['W2']
    W2 = np.array(W2)
    b2 = W['b2']
    b2 = np.array(b2)

    print test_input.shape

    Z1 = test_input.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
    result = np.argmax(A2, axis=1).astype(int)
    result = np.add(result, 1)
    return result


def produce_result_predictions(data,predict,W):
    columns1 = ["X1", "X2"]
    X = data[columns1].values
    y = data["Y"].values
    print y

    markers = ['s', 'x', 'o', '^ ', 'v']
    colors = ['red', 'blue', 'lightgreen', 'lightyellow', 'cyan']

    cmap = ListedColormap(colors[:len(np.unique(y))])


    # Define our usual decision surface bounding plots
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                         , np.arange(y_min, y_max, h))

    k = np.c_[xx.ravel(), yy.ravel()]
    # print k.shape

    Z = predict_result(k,W)
    # print np.unique(Z)
    Z = Z.reshape(xx.shape)

    cm = plt.cm.jet
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    ax_c = f.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    cdict = {1: 'red', 2: 'blue', 3: 'green',4:'yellow'}
    predict_labels = {1:'Bolt predict',2:'Nut predict',3:'Ring predict',4:'Scrap predict'}

    for g in np.unique(y):
        # print "current G : ",g
        ix = np.where(predict == g)
        # print ix
        ax.scatter(test_input['X1'].values[ix], test_input['X2'].values[ix], c=cdict[g], label=predict_labels[g],cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    # ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
    #            cmap="RdBu", vmin=-.2, vmax=1.2,
    #            edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(x_min, x_max), ylim=(y_min, y_max),
           xlabel="$X_1$", ylabel="$X_2$")
    plt.legend()

    plt.savefig("decision_bound.png")
    plt.show()


test_input = pd.read_csv('test_data.csv',names = ["X1","X2","Y"])

test_data = test_input.iloc[:,0:2].values
# print test_data
test_labels = test_input.iloc[:,2:3].values

weightFile = "weightFile_1000.json"
f = open(weightFile,'r')
W = data = json.load(f)
# W1 = W['W1']
# W1 = np.array(W1)
# b1 = W['b1']
# b1 = np.array(b1)
# W2 = W['W2']
# W2 = np.array(W2)
# b2 = W['b2']
# b2 = np.array(b2)
f.close()



# Z1 = test_data.dot(W1) + b1
# A1 = sigmoid(Z1)
# Z2 = A1.dot(W2) + b2
# A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
# predict = np.argmax(A2, axis = 1).astype(int)
# predict = np.add(predict,1)

y_pred = predict_result(test_data,W)

evaluate(test_labels,y_pred)
create_confusion_matrix(test_labels,y_pred)
calculate_profit(test_labels,y_pred,profit_tablepro)
generate_plot(test_input,test_labels,y_pred)
produce_result_predictions(test_input,y_pred,W)
