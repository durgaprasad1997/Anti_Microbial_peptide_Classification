#Import statements
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

#function that converts arff file to csv
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

#function which reads arff
def readArffData(filename):
    # Main loop for reading and writing files
    with open(filename, "r") as inFile:
        content = inFile.readlines()
        name, ext = os.path.splitext(inFile.name)
        new = convertToCSVFile(content)
        with open(filename[:-5]+".csv", "w") as outFile:
            outFile.writelines(new)

#function reads given CSV file and returns dataframe object
def readCSVData(filename):
    data = pd.read_csv(filename)
    return data


# Function to graphically represent ouput lables
def plotOutputLabel(dataframe):
    class_dist = dataframe.groupby(188).size()
    class_label = pd.DataFrame(class_dist, columns=['Size'])
    plt.figure(figsize=(10, 2))
    sns.barplot(x=class_label.index, y='Size', data=class_label)
    plt.show()

# Data Normalization
def dataNormalization(dataframe):
    print("normalization")

    scaler = MinMaxScaler()

    scaler.fit(dataframe)
    dataframe = pd.DataFrame(scaler.transform(dataframe))


    dataframe = dataframe.round(5)
    dataframe = dataframe.dropna()

    print("Skewness")
    print(dataframe.skew(axis=0, skipna=True))

    return dataframe
# Function to display data properties
def dataProperties(dataframe):
    print('Data Dimension:')
    print('Number of Records:', dataframe.shape[0])
    print('Number of Features:', dataframe.shape[1])
    print('\n')
    print('Feature Names')
    print(dataframe.columns)
    print('\n')
    print(dataframe.index)
    print('\n')
    print(dataframe.values)
    print('\n')
    print(dataframe.info())
    print('\n')
    print("Skewness")
    print(dataframe.skew(axis=0, skipna=True))

#10 fold cross validation
def kFold(input, output):
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.1, random_state=0)
    return (X_train, X_test, y_train, y_test)

#Decision tree class
def decisionTreeClass(input, test, output):
    clf = tree.DecisionTreeClassifier()
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#naive bayes class
def naiveBayesClass(input, test, output):
    clf = MultinomialNB()
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#k-nn class
def kNNClass(input, test, output):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(input, output)
    y_pred = neigh.predict(test)
    return y_pred

#Rule based algorithm
def ruleBased(input, test, output):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#Random forests algorithms
def randomForest(input, test, output):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#SVM Algorithm
def svm(input,test,output):

    clf = SVC(gamma='auto')
    clf.fit(input, output)
    y_pred = clf.predict(test)
    return y_pred

#Evalation of ML Algorithms
def evaluateMLAlgo(input, output,status):
    x_train, x_test, y_train, y_test = kFold(input, output)
    cross_val_accuracies = pd.DataFrame()

    plt.figure(0).clf()

    if (status == 1):
        plt.title("Raw Data")
    elif (status == 2):
        plt.title("Over sampling")
    elif (status == 3):
        plt.title("under sampling")
    elif (status == 4):
        plt.title("balanced sampling")
    elif (status == 5):
        plt.title("Variance sampling")
    elif (status == 6):
        plt.title("Select K best features sampling")

    print("Decision Tree")
    y_output = decisionTreeClass(x_train, x_test, y_train)
    clf = tree.DecisionTreeClassifier()
    scores = cross_val_score(clf, input, output, cv=10)
    cross_val_accuracies['decisiontree'] = scores
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_output)
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="Decision Tree, auc=" + str(auc))
    plt.legend(loc=0)

    print("Naive Bayes")
    y_output = naiveBayesClass(x_train, x_test, y_train)
    clf = MultinomialNB()
    scores = cross_val_score(clf, input, output, cv=10)
    cross_val_accuracies['naive'] = scores
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_output)
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="Naive Bayes, auc=" + str(auc))
    plt.legend(loc=0)

    print("K-NN Algo")
    y_output = kNNClass(x_train, x_test, y_train)
    clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(clf, input, output, cv=10)
    cross_val_accuracies['knn'] = scores
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_output)
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="K-NN, auc=" + str(auc))
    plt.legend(loc=0)

    print("Random Forests")
    y_output = randomForest(x_train, x_test, y_train)
    clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(clf, input, output, cv=10)
    cross_val_accuracies['random'] = scores
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_output)
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="Random Forests, auc=" + str(auc))
    plt.legend(loc=0)

    print("Support Vector Machine")
    y_output = svm(x_train, x_test, y_train)
    clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(clf, input, output, cv=10)
    cross_val_accuracies['svm'] = scores
    printConfusionMatrix(y_test, y_output)
    printClassificationReport(y_test, y_output)

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_output)
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="Suport Vector Machine, auc=" + str(auc))
    plt.legend(loc=0)


    fpr, tpr, thresh = metrics.roc_curve([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    auc = metrics.roc_auc_score(y_test, y_output)
    plt.plot(fpr, tpr, label="Ideal Algorithm, auc=" + str(auc))
    plt.legend(loc=0)

    plt.show()

    return cross_val_accuracies


#function prints confusion matrix
def printConfusionMatrix(actual, predicted):
    print(confusion_matrix(actual, predicted))

#function prints classification report
def printClassificationReport(actual, predicted):
    print(classification_report(actual, predicted))

#ML Algorithm evaluation
def AlgoEvaluation(dataframe):

    output_dataframe = dataframe[188]
    input_dataframe = dataframe
    input_dataframe = input_dataframe.drop(columns=[188])

    kfoldaccuracies = pd.DataFrame()

    print("Raw Data")
    print(evaluateMLAlgo(input_dataframe.values, output_dataframe.values,1))


    print("Over sampling")
    X_resampled, y_resampled = SMOTE().fit_resample(input_dataframe, output_dataframe)
    re_sample = pd.DataFrame()
    re_sample[188] = y_resampled
    plotOutputLabel(re_sample)

    kfoldaccuracies = evaluateMLAlgo(X_resampled, y_resampled, 2)
    print(kfoldaccuracies)

    print("Under sampling")
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(input_dataframe, output_dataframe)
    re_sample = pd.DataFrame()
    re_sample[188] = y_resampled
    plotOutputLabel(re_sample)
    print(evaluateMLAlgo(X_resampled, y_resampled, 3))

    print("Balanced sampling")
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(input_dataframe, output_dataframe)
    re_sample = pd.DataFrame()
    re_sample[188] = y_resampled
    plotOutputLabel(re_sample)
    print(evaluateMLAlgo(X_resampled, y_resampled, 4))

    # feature extraction -1 Variance Threshold
    print("Variance Threshold")
    sel = VarianceThreshold()
    features = sel.fit_transform(X_resampled)
    print(evaluateMLAlgo(features, y_resampled, 5))
    # features = sel.fit_transform(input_dataframe)
    # print(evaluateMLAlgo(features, output_dataframe,5))

    # feature extraction -2 select K Best
    print("Select K Best")
    X_new = SelectKBest(chi2, k=7).fit_transform(X_resampled, y_resampled)
    print(evaluateMLAlgo(X_new, y_resampled, 6))
    # X_new = SelectKBest(chi2, k=7).fit_transform(input_dataframe, output_dataframe)
    # print(evaluateMLAlgo(X_new, output_dataframe,6))

    return kfoldaccuracies


#friedman  statistical testing
def statisticalTesting(kfoldaccuracies):
    stat, p = friedmanchisquare(kfoldaccuracies['decisiontree'], kfoldaccuracies['naive'], kfoldaccuracies['knn'],kfoldaccuracies['random'],kfoldaccuracies['svm'])
    print('stat=%.3f, p=%.9f' % (stat, p))
    if p > 0.001:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

#plots Heat map for feature correlation
def heatMap(dataframe):
    plt.matshow(dataframe.corr())
    plt.show()


#Main function
def main():
    readArffData('amp_classification.arff')
    readArffData('multilabel_amp_classification.arff')

    dataframe1 = readCSVData("amp_classification.csv")
    heatMap(dataframe1)
    print(dataframe1)


    dataframe1 = dataNormalization(dataframe1)
    heatMap(dataframe1)
    print(dataframe1)

    plotOutputLabel(dataframe1)
    dataProperties(dataframe1)

    kfoldaccuracies = AlgoEvaluation(dataframe1)
    statisticalTesting(kfoldaccuracies)

if __name__ == "__main__":
    main()
