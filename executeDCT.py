import numpy as np
import pandas as pd
import math
import pickle
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

profit_tablepro = [[ 20, -7,-7,-7],
          [-7,15,-7,-7],
          [-7,-7,5,-7],
          [-3,-3,-3,-3]
]

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
    # if (isinstance(model, (int))):
    if (model.currentAttribute == "" and model.lower == None and model.biggerOrEqual == None ):
        # print "result at this predict :",model
        return model.currentSplit
    # print row[model.currentAttribute]
    # print model.currentSplit
    if row[model.currentAttribute]>=model.currentSplit:
        return use_train_model(model.biggerOrEqual,row)
    else:
        return use_train_model(model.lower, row)

def evaluate(y,predict):
    accurate = 0
    for i in range(len(predict)):
        if predict[i] == y[i]:
            # print "yes"
            accurate+=1
    print "the accuracy is :",(accurate*1.0/len(predict)),"% "




def create_confusion_matrix(y,y_pred):
    print('Confusion Matrix')
    print(confusion_matrix(y, y_pred))
    print('Classification Report')
    # target_names = ['Cats', 'Dogs', 'Horse']
    t = ["1","2","3","4"]
    labels = np.array(t)
    print(classification_report(y, y_pred, target_names=labels))



def calculate_profit(y,predict,profitTable):
    totalProfit = 0
    for i in range(len(predict)):
        totalProfit+= profitTable[y[i]-1][int(predict[i])-1]
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

def test(test_input,model,profit_table):
    y_pred = np.zeros(len(test_input))
    for index, row in test_input.iterrows():
        y_pred[index] = int(use_train_model(model,row))

    generate_plot(test_input,test_input.Y.values,y_pred)
    create_confusion_matrix(test_input.Y.values,y_pred)
    evaluate(test_input.Y.values,y_pred)
    calculate_profit(test_input.Y,y_pred,profit_table)

test_input = pd.read_csv('test_data.csv',names = ["X1","X2","Y"])

filename = 'filename_pi.obj'
filehandler = open(filename, 'r')
model = pickle.load(filehandler)



test(test_input,model,profit_tablepro)