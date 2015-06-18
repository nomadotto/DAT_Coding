# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:18:33 2015

@author: Nedd
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import pandas



def city_state_split(string ):
    """ Splits a string into a state_city pair. Assumes of 
    form statecode_city"""
    return(string.split("_"))
    
def bool_mapper(dataframe, Colname):
    """lazy way to map a Y/N column in a dataframe to a 1,0 Column. 
    Takes a dataframe and the name of the column to convert"""
    boolmap = {'Y':1, 'N':0}
    dataframe[Colname] = data[Colname].map(boolmap)
    return dataframe
    


#read in data
path = ("C:\Users\edwardda\Documents\Sample Data\categorization_pull.csv")
llpath = ("C:\Users\edwardda\Documents\Sample Data\Lat_Long.csv")
rawdata = pandas.read_csv(path)
latlong = pandas.read_csv(llpath)

#turn origin and destination into numbers
latlong.columns = ['orglat', 'orglong', 'ORIGIN']

rawdata = pandas.merge(rawdata, latlong)
latlong.columns = ['destlat', 'destlong', 'DESTINATION']
rawdata = pandas.merge(rawdata, latlong)

#make a new, cleaned dataframe
cols = ['EQUIPMENT_CATEGORY','DURATION', 'AVAIL_SAME_DAY','orglat', 'orglong',
'destlat', 'destlong']
data = rawdata[cols]

#turn avail into a 1,0
bool_mapper(data,'AVAIL_SAME_DAY' )


#let's do a plot, 


#time to learn, we have two options, we can shift to using Numpy Arrays or 
#we can import a new package to handle the scikitlearn noise
#for now, we are numpy bound

labels = np.array(data[cols[0]])
vals = np.array(data[cols[1:6]])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(vals, 
                            labels, test_size = .1)


knn = neighbors.KNeighborsClassifier()
svc1 = SVC()
svc2 = SVC()
models = [knn, svc1, svc2]

tuned_parameters = [{'n_neighbors':range(1,10)},
                     {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
i =0
for model in models:
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(model, tuned_parameters[i], 
                       cv=10, scoring = score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    i=i+1







