import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys



class Node:
    def __init__(self, attribute,val):
        self.lower = None
        self.biggerOrEqual = None
        self.currentAttribute = attribute
        self.currentSplit = val
        self.depth = 0

    def _printTree(self, node):
        if not (isinstance(node, (int))):

            print str(node.currentAttribute) + ' < ' + str(node.currentSplit)
            self._printTree(node.lower)
            print str(node.currentAttribute) + ' >=' + str(node.currentSplit)
            self._printTree(node.biggerOrEqual)
        else:
            print str(node) + '\n'


def use_train_model(model,row):
    # make decision prediction for each row by go through the tree
    if (model.currentAttribute == "" and model.lower == None and model.biggerOrEqual == None ):
        return model.currentSplit
    if row[model.currentAttribute]>=model.currentSplit:
        return use_train_model(model.biggerOrEqual,row)
    else:
        return use_train_model(model.lower, row)

def evaluate(y,predict):
    # evaluate the accuracy level
    accurate = 0
    for i in range(len(predict)):
        if predict[i] == y[i]:
            # print "yes"
            accurate+=1
    print "the accuracy is :",(accurate*1.0/len(predict)),"% "




def create_confusion_matrix(y,y_pred):
    # create confusion matrix
    print('Confusion Matrix')
    print(confusion_matrix(y, y_pred))
    print('Classification Report')
    t = ["Bolt", "Nut", "ring", "Scrap"]
    labels = np.array(t)
    print(classification_report(y, y_pred, target_names=labels))



def calculate_profit(y,predict,profitTable):
    # calculate total profit for the prediction
    totalProfit = 0
    for i in range(len(predict)):
        totalProfit+= profitTable[y[i]-1][int(predict[i])-1]
    print "the total profit is : ",totalProfit


def predict_result(test_data,model):
    # calculate prediction from decision tree model and test input
    y_pred = np.zeros(len(test_data))
    for index, row in test_data.iterrows():
        y_pred[index] = int(use_train_model(model, row))
    return y_pred

def produce_result_predictions(data,predict,model):
    # generate decision bound graph

    columns1 = ["X1", "X2"]
    X = data[columns1].values
    y = data["Y"].values

    marker = {1: 'x', 2: 'o', 3: '^', 4: 'v'}
    colors = ['red', 'blue', 'lightgreen', 'lightyellow', 'cyan']

    cmap = ListedColormap(colors[:len(np.unique(y))])


    # Define our usual decision surface bounding plots
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                         , np.arange(y_min, y_max, h))

    k = np.c_[xx.ravel(), yy.ravel()]
    k = pd.DataFrame(k,columns=["X1","X2"])
    Z = predict_result(k,model)
    Z = Z.reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    ax_c = f.colorbar(contour)
    ax_c.set_ticks([0, .25, .5, .75, 1])

    cdict = {1: 'red', 2: 'blue', 3: 'green',4:'black'}
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

def test(test_input,y_pred,profit_table):
    create_confusion_matrix(test_input.Y.values,y_pred)
    evaluate(test_input.Y.values,y_pred)
    calculate_profit(test_input.Y,y_pred,profit_table)

# profit table for 4 targets classes
profit_tablepro = [[ 20, -7,-7,-7],
          [-7,15,-7,-7],
          [-7,-7,5,-7],
          [-3,-3,-3,-3]
]

# test file name
fileName = sys.argv[1]
# model file name
modelFileName = sys.argv[2]
# load test data
test_input = pd.read_csv(fileName,names = ["X1","X2","Y"])

filehandler = open(modelFileName, 'r')

# load model
model = pickle.load(filehandler)

# generate predictions for test input
y_pred = predict_result(test_input,model)

test(test_input,y_pred,profit_tablepro)
# create the decision bound graph
produce_result_predictions(test_input,y_pred,model)