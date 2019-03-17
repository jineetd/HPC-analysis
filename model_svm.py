import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

dataset = pd.read_excel("modelData_Int.xlsx")
dataset = dataset.fillna(value=0)
dataset.columns=['sr_no','benchmark','kernel_name','C/M','reg_thread','smem_pblock','no_of_cmpinst','no_of_globinst','no_of_sharinst','miscellaneous','no_of_blocks', 'threads_pbl','occupancy','power']
X = dataset.iloc[:, 4:13].values
#print X.shape
y = dataset.iloc[:, 13].values


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X.reshape(700,9))
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(700,1))
print (X.shape)
print (y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#Import svm model
from sklearn.svm import SVR

regressor = SVR(gamma='auto')
regressor.fit(X_train,y_train.flatten())

print("Score:",regressor.score(X,y))
#Predict the response for test dataset
y_pred = regressor.predict(X_test)

y_test = sc_y.inverse_transform(y_test)
y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("SVM ACCURACY USING RSQUARE")
# Model Accuracy: how often is the classifier correct?
r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')
evs = explained_variance_score(y_test, y_pred)
print("Rsquare: ",r2,"EVS:",evs)

