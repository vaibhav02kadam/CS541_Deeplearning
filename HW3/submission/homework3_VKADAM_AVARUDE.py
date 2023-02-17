import numpy as np

X_tr = np.load("fashion_mnist_train_images.npy")
n = X_tr.shape[0]
# Reshaping the data
X_tr = X_tr.reshape((n,-1))
# Dividing the all the pixel valus of image by 255 value
X_tr=X_tr/255
y_tr = np.load("fashion_mnist_train_labels.npy")

#one hot encoding
new_ytr=np.zeros((y_tr.shape[0],10))
for i in range(y_tr.shape[0]):
  new_ytr[i][y_tr[i]]=1

#loading and processing dataset for testing
X_test = np.load("fashion_mnist_test_images.npy")
n = X_test.shape[0]
X_test = X_test.reshape((n,-1))
X_test=X_test/255
y_test = np.load("fashion_mnist_test_labels.npy")
new_ytest=np.zeros((y_test.shape[0],10))
for i in range(y_test.shape[0]):
  new_ytest[i][y_test[i]]=1

#Creating a validation set
k=X_tr.shape[0]
ind=np.arange(k)
vset_size=round(0.2*k) # 20 percent of the data
ind_val=np.random.choice(ind,size=vset_size,replace=False)

X_valid = X_tr[ind_val,:]
Y_valid = new_ytr[ind_val,:]
# creating the training data
idx_tr = np.setxor1d(ind, ind_val)
X_train = X_tr[idx_tr,:]
Y_train = new_ytr[idx_tr,:]


def batch_former(X,Y,batchsize):
  ind=np.random.permutation(X.shape[0])
  x=X[ind]
  y=Y[ind]
  for i in range(0,X.shape[0],batchsize):
    yield x[i:i+batchsize],y[i:i+batchsize]

def cal_cross_entropy_loss(X,Y,weight_bias):
  new_col=np.ones((X.shape[0],1))
  X=np.hstack((X, new_col))
  n = Y.shape[0]
  prediction=soft_max(np.dot(X,weight_bias))
  loss=np.sum(Y*np.log(prediction))*-1/n
  return loss

def soft_max(mat):
  exp=np.exp(mat)
  b=np.sum(exp,axis=1,keepdims=True)
  # print(b.shape)
  return exp/b

def cal_acc(X,Y,weight_bias):
  new_col=np.ones((X.shape[0],1))
  X=np.hstack((X, new_col))
  n = Y.shape[0]
  prediction=soft_max(np.dot(X,weight_bias))
  # print(prediction.shape)
  a=np.mean(np.argmax(prediction,axis=1)==np.argmax(Y,axis=1))
  return np.mean(a)

def l2reg_softmax(X,Y,bs,epochs,l_r,alpha):
  k,m=X.shape
  k=Y.shape
  nums_classes=10
  new_col=np.ones((X.shape[0],1))
  X=np.hstack((X, new_col))
  print("Data input",X.shape)
  print("label input",Y.shape)
  # defining random weights and bias
  w = np.random.randn(m,nums_classes)
  b = np.random.randn(1,nums_classes)
  batchsize=bs
  weight_bias=np.append(w,b,axis=0)
  for epoch in range(epochs):
    for batch_x,batch_y in batch_former(X,Y,batchsize):
      prediction=soft_max(np.dot(batch_x,weight_bias)) #making  prediction
      #performing gradient 
      gradient=(1/batch_y.shape[0])*(np.dot(batch_x.T, (prediction-batch_y))) + alpha*np.append(w,np.zeros((1,nums_classes)),axis=0)*(1/batch_y.shape[0])
      weight_bias = weight_bias - l_r*gradient
      w=weight_bias[:m,:nums_classes]
    # loss= cal_cross_entropy_loss(X_valid, Y_valid, weight_bias)
    # print("Log loss for each epoch: ", loss)
  return weight_bias

# # Tune Hyper parameter
# LR_set = [0.02,0.01,0.05,0.1]
# EPOCHS_set = [100, 200,300,400]
# BATCHSIZE_set = [32,64,128,256]
# ALPHA_set = [0.001, 0.002, 0.003,0.04]
# cost = 100000
# for l_r in LR_set:
#   for epochs in EPOCHS_set:
#     for bs in BATCHSIZE_set:
#       for alpha in ALPHA_set:
#         weight_bias = l2reg_softmax(X_train, Y_train,bs,epochs,l_r,alpha)
#         loss= cal_cross_entropy_loss(X_valid, Y_valid, weight_bias)
#         print("alpha, bs, epoch,lr",alpha, bs, epochs,l_r)
#         print("Log loss: ", loss)
#         acc=cal_acc(X_valid,Y_valid,weight_bias)
#         print("acc: ", acc)
#         if(loss<cost):
#           lr_f = l_r
#           epochs_f = epochs
#           bs_f = bs
#           alpha_f = alpha
#           cost=loss

#Substituting the tuned hyperparameters and training on the entire dataset
weight_bias=l2reg_softmax(X_tr, new_ytr,32,100,0.02,0.004)
# Calculating loss
loss=cal_cross_entropy_loss(X_test,new_ytest,weight_bias)
print("Loss: ", loss)
#Calculating the acuracy on the testing set 83.8%
acc=cal_acc(X_test,new_ytest,weight_bias)
print("acc: ", acc)