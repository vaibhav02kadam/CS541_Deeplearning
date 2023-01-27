import numpy as np


def batch_former(X,Y,batchsize):
  ind=np.random.permutation(X.shape[0])
  x=X[ind]
  y=Y[ind]
  for i in range(0,X.shape[0],batchsize):
    yield x[i:i+batchsize],y[i:i+batchsize]

def cal_MSE(X, Y, w,b):
  n = Y.shape[0]
  prediction=X.dot(w) + b
  mse = np.sum(np.square(prediction - Y))*(1/(2 * n))
  return mse


def SGD(X,Y,batchsize,epochs,l_r,alpha):
  k,m=X.shape
  k=Y.shape
  # defining random weights and bias
  w = np.random.randn(m)
  b = np.random.randn(1)
  for epoch in range(epochs):
    for batch_x,batch_y in batch_former(X,Y,batchsize):
      prediction=np.dot(batch_x,w)+b #making  prediction
      #performing gradient 
      gradient=(1/batch_y.shape[0])*(np.dot(batch_x.T, (prediction-batch_y) ))+ alpha*w*(1/batch_y.shape[0]) 
      gradient_b=(1/batch_y.shape[0])*np.sum((prediction-batch_y)) 
      # updating weights and bias
      w = w - l_r*gradient
      b = b - l_r*gradient_b
  return w,b
      
#Data Loading
X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
y_tr = np.load("age_regression_ytr.npy")
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
y_te = np.load("age_regression_yte.npy")

#Creating a validation set
k=X_tr.shape[0]
ind=np.arange(k)
vset_size=round(0.2*k) # 20 percent of the data
ind_val=np.random.choice(ind,size=vset_size,replace=False)

X_valid = X_tr[ind_val,:]
Y_valid = y_tr[ind_val]
# creating the training data
idx_tr = np.setxor1d(ind, ind_val)
X_train = X_tr[idx_tr,:]
Y_train = y_tr[idx_tr]


# Tune Hyper parameter
LR_set = [0.001,0.0001, 0.002,0.0015]
EPOCHS_set = [100, 200, 300, 500]
BATCHSIZE_set = [16,32,64,128,256]
ALPHA_set = [0.1, 0.2, 0.3,0.5]
cost = 100000
for l_r in LR_set:
  for epochs in EPOCHS_set:
    for bs in BATCHSIZE_set:
      for alpha in ALPHA_set:
        w,b = SGD(X_train, Y_train,bs,epochs,l_r,alpha)
        MSE = cal_MSE(X_valid, Y_valid, w,b)
        print("alpha, bs, epoch,lr",alpha, bs, epochs,l_r)
        print("MSE: ", MSE)
        
        if(MSE<cost):
          lr_f = l_r
          epochs_f = epochs
          bs_f = bs
          alpha_f = alpha
          cost=MSE
#printing out the best hyper parameters
print(lr_f,epochs_f,bs_f,alpha_f) 

# The best hyperparameters after training are 
# Learning rate: 0.001
# Epochs: 300
# Batchsize: 16
# Alpha:0.5
# lr_f,bs_f,epochs_f,alpha_f=0.001, 16 ,300, 0.5

#Training the entire dataset with tuned hyper parameters
w,b = SGD(X_tr, y_tr,bs_f,epochs_f,lr_f,alpha_f)
#calculating the MSE loss for the testing dataset using the tuned parameters
mse_test_loss=cal_MSE(X_te, y_te, w,b)
print(mse_test_loss)