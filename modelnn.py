import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#Take data input from the csv file
df = pd.read_csv('Data - Sheet1.csv')
print(df.info())
df=df.drop(['sr_no'],axis=1)
df = df.drop(['benchmark'],axis=1) # Drop Date feature
df = df.drop(['kernel_name'],axis=1) # Drop Date feature
df = df.drop(['C/M'],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.

df_train = df[:1059]    # 60% training data and 40% testing data
df_test = df[1059:]
X_train,X_test
scaler = MinMaxScaler() # For normalizing dataset


#predict power consumed by HPC's
X_train = scaler.fit_transform(df_train.drop(['power'],axis=1).as_matrix())
print(X_train.shape)
#print(df_train['power'].as_matrix())
#y_train=np.array(y_train).reshape(-1,1)
y_train = scaler.fit_transform(df_train['power'].as_matrix())


X_test = scaler.fit_transform(df_test.drop(['power'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['power'].as_matrix())

#denormalize the data
def denormalize(df,norm_data):
    df = df['power'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)

#nueral net model
def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)


    # layer 1 multiplying and adding bias then activation function

    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)

    # layer 2 multiplying and adding bias then activation function

    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)

    # O/p layer multiplying and adding bias then activation function

    # notice output layer has one node only since performing #regression

    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs,3)

#mean square cost function
cost = tf.reduce_mean(tf.square(output-ys))

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:

    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
            # Run cost and train with each sample

        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])

    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training

    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    y_test = denormalize(df_test,y_test)
    pred = denormalize(df_test,pred)
'''
   	plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')
    plt.show()
'''

