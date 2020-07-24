
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def comp_regressions(csv,dv):
    pass

def comp_classifications(csv,dv):
    #raise Exception("In comp_classifications: ",csv,dv)
    df = pd.read_csv("api/static/api/uploads/"+csv)   #; indp_vars = [a,b]
    x = df.iloc[:,[2,3]].values     #[i for i in range(1,int(dv))]+[i for i in range(int(dv)+1,len(df.columns))]
    y = df.iloc[:,int(dv)].values

    '''Splitting dataset into training and testing set'''
    from sklearn.model_selection import train_test_split
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    '''feature scaling'''
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
    x_test = sc_X.transform(x_test)

    #----------------------------------------------------------------------------------------------

    models_accuracies = {} ; max_accuracy=0 ; maxx_models=[]

    '''fitting logictic regression to training set'''
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['LogisticRegression_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("LogisticRegression_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("LogisticRegression_Classifier")
    else:
        pass


    '''fitting KNNclassifier to training set'''
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')  # metric and p parameters are for choosing eucledian distance...... see ctrl+I
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['KNN_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("KNN_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("KNN_Classifier")
    else:
        pass


    '''fitting KernalSVM to training set'''
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf')   # rbf = gausian kernal
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['KernalSVM_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("KernalSVM_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("KernalSVM_Classifier")
    else:
        pass


    '''fitting SVM to training set'''
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear')   # rbf = gausian kernal
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['SVM_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("SVM_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("SVM_Classifier")
    else:
        pass


    '''fitting naive_bayes to training set'''
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()   # no parameters req in this
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['NaiveBayes_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("NaiveBayes_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("NaiveBayes_Classifier")
    else:
        pass


    '''fitting DecisionTree to training set'''
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion="entropy")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['DecisionTree_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("DecisionTree_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("DecisionTree_Classifier")
    else:
        pass


    '''fitting RandomForestClassifier to training set'''
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    models_accuracies['RandomForest_Classifier'] = acc
    if acc>max_accuracy:
        max_accuracy = acc
        maxx_models.clear() ; maxx_models.append("RandomForest_Classifier")
    elif acc==max_accuracy:
        max_accuracy = acc
        maxx_models.append("RandomForest_Classifier")
    else:
        pass


    # returning the accuracies
    #raise Exception("Accuracies: ",models_accuracies)
    return {'MA':models_accuracies,'max_accuracy':max_accuracy,'maxx_models':maxx_models }


def comp_clusterings(csv,dv):
    pass

def runmodels(p_datacsv, p_depvars, p_task):
    if p_task=="1":
        return comp_regressions(p_datacsv,p_depvars)
    elif p_task=="2":
        return comp_classifications(p_datacsv,p_depvars)
    elif p_task=="3":
        return comp_clusterings(p_datacsv,p_depvars)
    else:
        raise Exception("There can be no exception. "+"Arguments: ", p_datacsv, p_depvars, p_task)
