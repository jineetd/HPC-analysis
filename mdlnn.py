import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#Take data input from the csv file
df = pd.read_excel("modelData_Int.xlsx")
#print(df.info())


#df = df.fillna(value=0)

df.columns=['sr_no','benchmark','kernel_name','C/M','reg_thread','smem_pblock','no_of_cmpinst','no_of_globinst','no_of_sharinst','miscellaneous','no_of_blocks', 'threads_pbl','occupancy','power']
X = df.iloc[:, 4:13].values
#print X.shape
y = df.iloc[:, 13].values

#print dataset
#print dataset.count
print X
print y
#X = SelectKBest(chi2, k=2).fit_transform(X, y) #Confirms all features are 
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#X = sel.fit_transform(X)


#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X.reshape(700,9))
#sc_y = StandardScaler()
#y = sc_y.fit_transform(y.reshape(700,1))
print X.shape
print X
print y
print y.shape
scaler=MinMaxScaler()
X= scaler.fit_transform(X)
y= scaler.fit_transform(y.reshape(-1,1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print("This is y_test")
print(y_test)

#denormalize the data
def denormalize(df,norm_data):
    df = df.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new

#nueral net model
def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)


    # layer 1 multiplying and adding bias then activation function

    W_2 = tf.Variable(tf.random_uniform([10,8]))
    b_2 = tf.Variable(tf.zeros([8]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)

    # layer 2 multiplying and adding bias then activation function
    #W_3 = tf.Variable(tf.random_uniform([14,14]))
    #b_3 = tf.Variable(tf.zeros([14]))
    #layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
    #layer_3 = tf.nn.relu(layer_3)
    #Layer 3

    W_O = tf.Variable(tf.random_uniform([8,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)

    # O/p layer multiplying and adding bias then activation function

    # notice output layer has one node only since performing #regression

    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs,9)

#mean square cost function
cost = tf.reduce_mean(tf.square(output-ys))

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
c_t=[]
c_test=[]
with tf.Session() as sess:

    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:X_train[j,:].reshape(1,9), ys:y_train[j]})
            # Run cost and train with each sample

        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])

    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training

    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    #y_test=denormalize(y,y_test)
    #pred=denormalize(y,pred)
    #y_test= y_test.reshape(-1,1)
    #scl = MinMaxScaler()
    #a = scl.fit_transform(y.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test)
    pred = scaler.inverse_transform(pred)
    print("this is y_test")
    print(y_test)
    #Denormalize data     

    plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('power')
    plt.xlabel('parameters')
    plt.title('HPC analysis')
    plt.show()

#calculate r-squared value
from sklearn.metrics import r2_score
a=r2_score(y_test,pred)
print("r-squared score: ")
print(a)

#rmse score
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test,pred))
print(rms)
