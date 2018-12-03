import numpy as np
import pandas as pd
import math
import pickle


data_input = pd.read_csv('train_data.csv',names = ["X1","X2","Y"])


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

def calculate_entropy(dataset):
    n_size = len(dataset)

    if n_size==0:
        return 0

    n_1 = len(dataset[dataset["Y"] == 1]) * 1.0
    n_2 = len(dataset[dataset["Y"] == 2])* 1.0
    n_3 = len(dataset[dataset["Y"] == 3])* 1.0
    n_4 = len(dataset[dataset["Y"] == 4])* 1.0
    Info_1 = 0.0
    Info_2 = 0.0
    Info_3 = 0.0
    Info_4 = 0.0

    if n_1 > 0.0:
        Info_1 = -(n_1/n_size)*math.log(n_1/n_size,2)
    if n_2 > 0.0:
        Info_2 = -(n_2 / n_size) * math.log(n_2 / n_size, 2)
    if n_3 > 0.0:
        Info_3 = -(n_3 / n_size) * math.log(n_3 / n_size, 2)
    if n_4 >0.0:
        Info_4 = -(n_4 / n_size) * math.log(n_4 / n_size, 2)

    entropy = Info_1+Info_2+Info_3+Info_4
    return entropy



def find_max_gain_attribuute(dataset):
    # calculate base entropy
    base_entropy = calculate_entropy(dataset)
    n = len(dataset)

    # best IG
    best_IG = 0

    # best attribute
    best_attribute = ""

    # best split
    best_split_threshold = 0

    attributes = dataset.columns.values
    # go through every feature in the set
    for i in range(len(attributes) - 1):
        unique_values = np.unique(dataset[attributes[i]])
        for threshold in unique_values:
            lowerSet = dataset[dataset[attributes[i]] < threshold]
            upperSet = dataset[(dataset[attributes[i]] > threshold) | (dataset[attributes[i]] == threshold)]

            n_lower = len(lowerSet)
            n_upper = len(upperSet)


            lowerEntropy = calculate_entropy(lowerSet)
            upperEntropy = calculate_entropy(upperSet)



            IG = base_entropy-(n_lower*1.0/n)*lowerEntropy-(n_upper*1.0/n)*upperEntropy

            if IG>best_IG:
                best_IG = IG
                best_attribute = attributes[i]
                best_split_threshold = threshold

    if best_attribute == "":
        print "weird case at for dataset: "
        print dataset

    return [best_attribute, best_split_threshold]

# def terminate_node():
#
# def decision_tree(,max_depth)

def check_pure(dataset):
    n_size = len(dataset)
    # if n_size == 0.0:
    #     return -2
    n_1 = len(dataset[dataset["Y"] == 1])
    n_2 = len(dataset[dataset["Y"]== 2])
    n_3 = len(dataset[dataset["Y"] == 3])
    n_4 = len(dataset[dataset["Y"]== 4])
    n_max = np.max([n_1,n_2,n_3,n_4])
    r = n_max*1.0/n_size
    if n_max == n_1:
        return [1,r]
    if n_max == n_2:
        return [2,r]
    if n_max == n_3:
        return [3,r]
    return [4,r]

def create_terminate_leaf_node(value,depth):
    res = Node("",value)
    res.depth = depth
    res.lower = None
    res.biggerOrEqual = None
    # print "leaf Node : ",1
    return res

def no_more_split(dataset):
    return len(dataset) == 1


def recursiveSplit(dataset,depth,maxDepth):
    if len(dataset) == 0:
        return None

    if no_more_split(dataset):
        return create_terminate_leaf_node(dataset.Y.values[0], depth)

    predict, percentConfidence = check_pure(dataset)

    if depth == maxDepth or percentConfidence ==1.0:
        return create_terminate_leaf_node(predict, depth)


    bestAttribute,bestSplit = find_max_gain_attribuute(dataset)
    lowerSubData = dataset[dataset[bestAttribute]<bestSplit]
    upperSubData = dataset[(dataset[bestAttribute]>bestSplit) | (dataset[bestAttribute]==bestSplit)]


    res = Node(bestAttribute,bestSplit)
    res.depth = depth

    lowerResult = recursiveSplit(lowerSubData,depth+1,maxDepth)
    if lowerResult == None:
        res.lower = create_terminate_leaf_node(predict, depth)
    else:
        res.lower = lowerResult

    upperResult = recursiveSplit(upperSubData,depth+1,maxDepth)
    if upperResult == None:
        res.biggerOrEqual = create_terminate_leaf_node(predict, depth)
    else:
        res.biggerOrEqual = upperResult
    return res



result = recursiveSplit(data_input,0,6)

def findInfo(model):
    # if (isinstance(model, (int))):
    if (model.currentAttribute == "" and model.lower == None and model.biggerOrEqual == None):
        return [model.depth, model.depth, model.depth, 1, 1]

    lower_maxDepth, lower_minDepth, lower_totalDepth, lower_totalNode, lower_totalLeafNode = findInfo(model.lower)
    b_maxDepth, b_minDepth, b_totalDepth, b_totalNode, b_totalLeafNode = findInfo(model.biggerOrEqual)

    return [max(lower_maxDepth,b_maxDepth),min(lower_minDepth,b_minDepth),lower_totalDepth+b_totalDepth,lower_totalNode+b_totalNode+1,lower_totalLeafNode+b_totalLeafNode]

    # return [maxDepth,minDepth,totalDepth,totalNode,totalLeafNode]



file_pi = open('filename_pi.obj', 'w')
pickle.dump(result, file_pi)
max_depth,min_depth,total_depth,total_node,total_leaf_node = findInfo(result)
print "Max depth : ",max_depth
print "Min depth : ",min_depth
print "average depth : ",total_depth*1.0/total_leaf_node
print "total node : ",total_node
print "total leaf node : ",total_leaf_node