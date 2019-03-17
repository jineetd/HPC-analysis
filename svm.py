import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#reading from file
df = pd.read_excel("modelData_Int.xlsx")

df.columns=['sr_no','benchmark','kernel_name','C/M','reg_thread','smem_pblock','no_of_cmpinst','no_of_globinst','no_of_sharinst','miscellaneous','no_of_blocks', 'threads_pbl','occupancy','power']
X = df.iloc[:, 4:13].values
print X.shape
y = df.iloc[:, 13].values
#df=pd.read_csv('Data - Sheet1.csv')
#df.head()

print X
print y

#scaling the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X= scaler.fit_transform(X)
y= scaler.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print("This is y_test")
print(y_test)
from sklearn.svm import SVR  
reg= SVR(gamma='auto')
reg.fit(X_train, y_train) 

#making predictions
y_pred = reg.predict(X_test)  
print("this is y_pred")
print(y_pred)

print("this is y_test after pred")
print(y_test)

y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
plt.plot(range(y_test.shape[0]),y_pred,label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('power')
plt.xlabel('parameters')
plt.title('HPC analysis')
plt.show()

#check accuracy
#print("Score:")
#print(reg.score(X_test,y_pred))
print("R2_score:")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#rmse values
print("RMSE:")
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test,y_pred))
print(rms)