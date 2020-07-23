
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def comp_regressions(csv,dv):
    pass

def comp_classifications(csv,dv):
    df = pd.read_csv(csv)   #; indp_vars = [a,b]
    x = df.iloc[:,[i for i in range(1,dv)]+[i for i in range(dv+1,len(df.columns))]].values
    y = df.iloc[:,dv].values

    '''Splitting dataset into training and testing set'''
    from sklearn.model_selection import train_test_split
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    '''feature scaling'''
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
    x_test = sc_X.transform(x_test)

    #----------------------------------------------------------------------------------------------

    models_accuracies = {}

    '''fitting logictic regression to training set'''
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['LogisticRegression Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting KNNclassifier to training set'''
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')  # metric and p parameters are for choosing eucledian distance...... see ctrl+I
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['KNN Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting KernalSVM to training set'''
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0)   # rbf = gausian kernal
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['KernalSVM Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting SVM to training set'''
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)   # rbf = gausian kernal
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['SVM Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting naive_bayes to training set'''
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()   # no parameters req in this
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['NaiveBayes Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting DecisionTree to training set'''
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['DecisionTree Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    '''fitting RandomForestClassifier to training set'''
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    models_accuracies['RandomForest Classifier'] = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

    # returning the accuracies
    return models_accuracies


def comp_clusterings(csv,dv):
    pass

def runmodels(p_datacsv, p_depvars, p_task):
    if p_task=="1":
        comp_regressions(p_datacsv,p_depvars)
    elif p_task=="2":
        comp_classifications(p_datacsv,p_depvars)
    elif p_task=="3":
        comp_clusterings(p_datacsv,p_depvars)
    else:
        raise Exception("There can be no exception")
