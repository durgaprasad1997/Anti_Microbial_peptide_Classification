#Importing packages
import csv
import pandas as pd
import os
import csv
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix


#function to read CSV File
def convertToCSVFile(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent
#function to read Arff file
def readArffData(filename):
    # Main loop for reading and writing files
    with open(filename, "r") as inFile:
        content = inFile.readlines()
        name, ext = os.path.splitext(inFile.name)
        new = convertToCSVFile(content)
        with open(filename[:-5]+".csv", "w") as outFile:
            outFile.writelines(new)

#function reading CSV File and returns DataFrame object
def readCSVData(filename):
    data = pd.read_csv(filename)
    return data



# Data Normalization
def dataNormalization(dataframe):
    print("normalization")
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    dataframe = pd.DataFrame(scaler.transform(dataframe),columns=dataframe.columns)
    dataframe = dataframe.round(5)
    dataframe = dataframe.dropna()

    print("Skewness")
    print(dataframe.skew(axis=0, skipna=True))

    return dataframe

#Splitting the data using Train Test Spit
def train_test(input, output):
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.1, random_state=0)
    return (X_train, X_test, y_train, y_test)




#Ridge CLassifier CV Algorithm
def RidgeClassifierCVAlgo(input, test, output):
    clf = RidgeClassifierCV()
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#Random Forest Algorithm
def RandomForestAlgo(input, test, output):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#Multi Label Classification using Binary Relevance Algorithm
def BinaryRelevanceAlgo(algoObject , input, test, output):
    classifier = BinaryRelevance(algoObject)
    classifier.fit(input, output)
    predictions = classifier.predict(test)
    return predictions
#Multi Label CLassification using Classifier Chain Algorithm
def ClassifierChainAlgo(algoObject , input, test, output):
    classifier = ClassifierChain(algoObject)
    classifier.fit(input, output)
    predictions = classifier.predict(test)
    return predictions

#Print Confusion Matrix
def printConfusionMatrix(actual, predicted):
    print(multilabel_confusion_matrix(actual, predicted))
#Print CLassification Report
def printClassificationReport(actual, predicted):
    print(classification_report(actual, predicted))


#Evaluation Machine Learning Algo
def evaluateMLAlgo(input, output):
    x_train, x_test, y_train, y_test = train_test(input, output)


    print("Random Forest")
    y_output = RandomForestAlgo(x_train, x_test, y_train)
    print(y_output)
    print(y_test)
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)
    print(accuracy_score(y_test, y_output))


    print("BinaryRelevance using GaussianNB")
    y_output = BinaryRelevanceAlgo(GaussianNB(),x_train, x_test, y_train)
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)
    print(accuracy_score(y_test, y_output))


    print("Classification chains for  RidgeClassifierCV")
    y_output = ClassifierChainAlgo(RidgeClassifierCV(), x_train, x_test, y_train)
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)
    print(accuracy_score(y_test, y_output))


    print("Classification chains for  DecisionTreeClassifier")
    y_output = ClassifierChainAlgo(DecisionTreeClassifier(), x_train, x_test, y_train)
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)
    print(accuracy_score(y_test, y_output))




def AlgoEvaluation(dataframe):
    global output_dataframe
    global input_dataframe
    output_dataframe = dataframe[["wound-healing-peptides","spermicidal-peptides","insecticidal-peptides","chemotactic-peptides","antifungal-peptides","anti-protist-peptides","antioxidant-peptides","antibacterial-peptides","antibiofilm-peptides","antimalarial-peptides","Antiparasitic-peptides","antiviral-peptides","anticancer-peptides","anti-HIV-peptides","protease-inhibitors","surface-immobilized-peptides"]]

    input_dataframe = dataframe
    input_dataframe = input_dataframe.drop(columns = ["wound-healing-peptides","spermicidal-peptides","insecticidal-peptides","chemotactic-peptides","antifungal-peptides","anti-protist-peptides","antioxidant-peptides","antibacterial-peptides","antibiofilm-peptides","antimalarial-peptides","Antiparasitic-peptides","antiviral-peptides","anticancer-peptides","anti-HIV-peptides","protease-inhibitors","surface-immobilized-peptides"])



    evaluateMLAlgo(input_dataframe, output_dataframe)



#Main Function
def main():

    readArffData('multilabel_amp_classification.arff')
    dataframe1 = readCSVData('multilabel_amp_classification.csv')
    dataframe2 = dataNormalization(dataframe1)
    AlgoEvaluation(dataframe2)




if __name__ == "__main__":
    main()
