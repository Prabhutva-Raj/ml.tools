
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def comp_regressions(df,dv):
    pass

def comp_classifications(df,dv):
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

    #models_accuracies = {} ; max_accuracy=0 ; maxx_models=[]
    accuracy_list = {} ; max_score=0 ; max_models=[]

    models = {'LogisticRegression_Classifier':LogisticRegression(), 'NaiveBayes_Classifier':GaussianNB(), 'KNN_Classifier':KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'), 'SVM_Classifier':SVC(kernel='linear'), 'KernalSVM_Classifier':SVC(kernel='rbf'), 'DecisionTree_Classifier':DecisionTreeClassifier(criterion="entropy"),  'RandomForest_Classifier':RandomForestClassifier(n_estimators = 10, criterion='entropy') }
    for key,value in models.items():
        model = value
        model.fit(x_train, y_train)
        accuracy_list.update({key:model.score(x_test, y_test)})
        if accuracy_list[key] > max_score:
            max_score = accuracy_list[key]
            max_models.clear() ; max_models.append(key)
        elif accuracy_list[key] == max_score:
            max_models.append(key)
        else:
            continue

    # returning the accuracies
    #return {'MA':models_accuracies,'max_accuracy':max_accuracy,'maxx_models':maxx_models }
    return {'MA':accuracy_list,'max_accuracy':max_score,'maxx_models':max_models }


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
