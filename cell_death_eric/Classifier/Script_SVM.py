## classifier in 2 categories : division or apoptosis 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

bankdata = pd.read_csv('.cvs')

## exploration of data 

bankdata.shape
bankdata.head()

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

## import model and train 
from sklearn.svm import SVC  

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  

## make predictions 
y_pred = svclassifier.predict(X_test)  

## evaluation of algo 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  


## results 