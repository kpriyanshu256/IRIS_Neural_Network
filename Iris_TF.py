import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import tensorflow as tf

def get_data(file):
    df=pd.read_csv(file,names=[0,1,2,3,4])
    d={'Iris-setosa':0,'Iris-virginica':2,'Iris-versicolor':1}
    df[4]=df[4].map(d)
    X=df.iloc[:,0:4]
    y=df.iloc[:,4:]
    X=np.array(X)
    y=np.array(y)
    
    ss=StandardScaler().fit(X)
    X=ss.transform(X)
    
    ohe=OneHotEncoder(categorical_features=[0])
    y=ohe.fit_transform(y).toarray()
    
    # Important step else data types won't match
    X=np.array(X,dtype='float32')
    y=np.array(y,dtype='float32')
    return train_test_split(X,y,test_size=0.7,random_state=0)


def model(X_train,X_test,y_train,y_test):
    X_train=X_train.T
    X_test=X_test.T
    y_train=y_train.T
    y_test=y_test.T
    
    
    n_x,m=X_train.shape
    n_h=6
    n_y,_=y_train.shape
    epoch=10000
    alpha=0.5
    
    # Initializing placeholders for data
    X=tf.placeholder(tf.float32)    
    y=tf.placeholder(tf.float32)    
    
    #Initializing neural network
    W1=tf.Variable(tf.random_normal((n_h,n_x)))
    b1=tf.Variable(tf.zeros((n_h,1)))
    W2=tf.Variable(tf.random_normal((n_y,n_h)))
    b2=tf.Variable(tf.zeros((n_y,1)))
    
    # Creating model
    Z1=tf.add(tf.matmul(W1,X),b1)
    A1=tf.sigmoid(Z1)
    Z2=tf.add(tf.matmul(W2,A1),b2)
    A2=tf.sigmoid(Z2)
    
    # Cost function
    error=tf.reduce_sum(tf.square(A2-y))
    
    with tf.Session() as sess:
        # Initializing tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        # Initialzing optimizer
        opt=tf.train.GradientDescentOptimizer(learning_rate=alpha)
        train=opt.minimize(error)
        
        # Starting training
        for i in range(epoch):
            sess.run(train,feed_dict={X:X_train,y:y_train})
            if i%1000==0:
                print('Error= ',sess.run(error,feed_dict={X:X_train,y:y_train}))
        
        
        # Testing
        output=sess.run(A2,feed_dict={X:X_test})
        
    # Checking accuracy
    pred=(np.argmax(output,axis=0)==np.argmax(y_test,axis=0))
    pred=pred*1
    pred=np.sum(pred)
    return (pred/y_test.shape[1])*100

if __name__=='__main__':
    filename='data/IrisData.txt'
    X_train,X_test,y_train,y_test=get_data(filename)
    accuracy=model(X_train,X_test,y_train,y_test)
    print("Accuracy= ",accuracy)
