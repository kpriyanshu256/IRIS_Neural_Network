import numpy as np
import pandas as pd

def sigmoid(x):
    # function to calculate sigmoid
    return (1 / (1 + np.exp(-x)))

def derivative(x):
    # function to calculate derivative of sigmoid
    return sigmoid(x)*(1-sigmoid(x))

def getData(file):
    # processing data
    labels=[0,1,2,3,4]
    df=pd.read_csv(file,names=labels)
    d={'Iris-setosa':0,'Iris-virginica':2,'Iris-versicolor':1}
    df[4]=df[4].map(d)
    df=df.iloc[0:150,:]
    df=normalize(df)
    df=df.sample(frac=1)
    return df

def normalize(X):
    #normalizing data
    for i in range(4):
        X[i]=(X[i]-X[i].min())/(X[i].max()-X[i].min())
    return X

def partition(df,ratio):
    # train-test split
    i=int((ratio/100)*df.shape[0])
    df1=df.iloc[0:i,:]
    df2=df.iloc[i:,:]
    X_train=df1.iloc[:,0:-1]
    y_train=df1.iloc[:,-1]
    X_test=df2.iloc[:,0:-1]
    y_test=df2.iloc[:,-1]
    
    # OneHot encoding of output
    Y=y_train==0
    Y1=Y*1
    Y=y_train==1
    Y2=Y*1
    Y=y_train==2
    Y3=Y*1
    df1=pd.DataFrame(Y1)
    df2=pd.DataFrame(Y2)
    df3=pd.DataFrame(Y3)
    Y=pd.concat([df1, df2], axis=1)
    Y=pd.concat([Y,df3],axis=1)
    
    y_train=Y
    
    Y=y_test==0
    Y1=Y*1
    Y=y_test==1
    Y2=Y*1
    Y=y_test==2
    Y3=Y*1
    df1=pd.DataFrame(Y1)
    df2=pd.DataFrame(Y2)
    df3=pd.DataFrame(Y3)
    Y=pd.concat([df1, df2], axis=1)
    Y=pd.concat([Y,df3],axis=1)
    y_test=Y
    
    
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    
    return X_train,y_train,X_test,y_test
    
    

def training(X_train,y_train):
    # training part
    m,n_x=X_train.shape
    n_y=3 # size of output layer
    n_h=6 # size of hidden layer
    alpha=0.5 # learning rate
    epoch=10000 # no. of iterations
    
    X_train=X_train.T
    y_train=y_train.T
    
    # initialisation of weights and biases
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))
    
    for i in range(epoch):
        
        # feed forward
        Z1=np.dot(W1,X_train)+b1
        A1=sigmoid(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=sigmoid(Z2)
        
        # back-propagation
        dA2=A2-y_train
        dZ2= dA2*derivative(Z2)
        dW2=(1/m)*np.dot(dZ2,A1.T)
        db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
        
        dA1= np.dot(W2.T,dZ2)
        dZ1=dA1*derivative(Z1)
        dW1=(1/m)*np.dot(dZ1,X_train.T)
        db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
        
        # updating weights and biases        
        W1=W1-alpha*dW1
        W2=W2-alpha*dW2
        b1=b1-alpha*db1
        b2=b2-alpha*db2
        
    return W1,b1,W2,b2
    
def predict(X_test,y_test,W1,b1,W2,b2):
    
    # prediction
    X_test=X_test.T
    y_test=y_test.T
    Z1=np.dot(W1,X_test)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    A2=np.round(A2)
    
    pred=(np.argmax(A2,axis=0)==np.argmax(y_test,axis=0))
    pred=pred*1
    print('Accuracy=', (np.sum(pred)/X_test.shape[1])*100)  
    return pred,A2
    
if __name__=='__main__':
    file='data/IrisData.txt'
    df=getData(file)
    X_train,y_train,X_test,y_test=partition(df,30)
    W1,b1,W2,b2=training(X_train,y_train)
    pred,A2=predict(X_test,y_test,W1,b1,W2,b2)
    

