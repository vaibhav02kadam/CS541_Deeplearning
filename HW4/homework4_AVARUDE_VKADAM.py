import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

#best 
def one_hot_encoding(y):
  ny=np.zeros((y.shape[0],10))
  for i in range(y.shape[0]):
    ny[i][y[i]]=1
  return ny


def unpack (weights):
  # Unpack arguments
  Ws = []
  # Weight matrices
  start = 0
  end = NUM_INPUT*NUM_HIDDEN[0]
  W = weights[start:end]
  Ws.append(W)
  # Unpack the weight matrices as vectors
  for i in range(NUM_HIDDEN_LAYERS - 1):
    start = end
    end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
    W = weights[start:end]
    Ws.append(W)
  start = end
  end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
  W = weights[start:end]
  Ws.append(W)
  # Reshape the weight "vectors" into proper matrices
  Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
  for i in range(1, NUM_HIDDEN_LAYERS):
    # Convert from vectors into matrices
    Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
  Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])
  # Bias terms
  bs = []
  start = end
  end = end + NUM_HIDDEN[0]
  b = weights[start:end]
  bs.append(b)
  for i in range(NUM_HIDDEN_LAYERS - 1):
    start = end
    end = end + NUM_HIDDEN[i+1]
    b = weights[start:end]
    bs.append(b)
  start = end
  end = end + NUM_OUTPUT
  b = weights[start:end]
  bs.append(b)
  return Ws, bs


def fCE (X, Y, weights):
  # Ws, bs = unpack(weights)
  n=X.shape[1]
  _,_,prediction=forwardpass(X,Y,weights)
  loss=np.sum(Y*np.log(prediction))*-1/n
  return loss


def cal_acc (X, Y, weights):
  # Ws, bs = unpack(weights)
  _,_,prediction=forwardpass(X,Y,weights)
  acc=np.mean(np.argmax(prediction,axis=0)==np.argmax(Y,axis=0))
  return acc


def relu(z):
  ind=np.where(z<0)
  z[ind[0],ind[1]]=0
  return z


def soft_max(mat):
  exp=np.exp(mat)
  b=np.sum(exp,axis=0,keepdims=True)
  return exp/b


def relu_prime(z):
  ind=np.where(z>0)
  z[ind[0],ind[1]]=1
  ind=np.where(z<0)
  z[ind[0],ind[1]]=0
  return z

def forwardpass(X,Y,weights):
  Ws, bs = unpack(weights)
  all_z=[]
  all_h=[]
  h=X
  all_z.append(np.zeros_like(X))
  all_h.append(h)
  for i in range(NUM_HIDDEN_LAYERS):
    z=np.dot(Ws[i],h)+np.atleast_2d(bs[i]).T
    h=relu(z)
    all_z.append(z)
    all_h.append(h)
  #now the last weight will give the y hat prediction
  prediction=soft_max(np.dot(Ws[-1],h)+np.atleast_2d(bs[-1]).T)

  return all_z,all_h,prediction


def gradCE (X, Y, weights):
  Ws, bs = unpack(weights)
  all_z,all_h,prediction=forwardpass(X,Y,weights)
  # print("len z",len(Ws))
  # print("len h",len(all_h))
  # print("predictions", prediction.shape)
  gradJ_Ws=[]
  gradJ_bias=[]
  g=prediction-Y
  alpha=0.0001
  n=X.shape[1]
  for i in reversed(range(NUM_HIDDEN_LAYERS+1)):
    delta_bias=np.sum(g,axis=1)
    gradJ_bias.append(delta_bias/n)
    if i==NUM_HIDDEN_LAYERS:
      delta_weight=np.dot(g,all_h[i].T)
      gradJ_Ws.append(delta_weight/n)
    else:
      delta_weight=np.dot(g,all_h[i].T)
      gradJ_Ws.append(delta_weight/n)
    g=np.dot(g.T,Ws[i])*relu_prime(all_z[i].T)
    g=g.T
  gradJ_Ws=gradJ_Ws[::-1]    
  gradJ_bias=gradJ_bias[::-1]  
  for i in range(len(Ws)):
    gradJ_Ws[i]=gradJ_Ws[i]+(alpha*Ws[i]/n)

  # gradJ_Ws.reverse()
  # gradJ_bias.reverse()
  grad_weights = np.hstack([ W.flatten() for W in gradJ_Ws ] + [ b.flatten() for b in gradJ_bias ])
  return grad_weights


def batch_former(X,Y,batchsize):
  ind=np.random.permutation(X.shape[1])
  x=X[:,ind]
  y=Y[:,ind]
  for i in range(0,X.shape[1],batchsize):
    yield x[:,i:i+batchsize],y[:,i:i+batchsize]


def train(trainX, trainY, weights, testX, testY, lr):
  epochs=10 #100
  lr=0.01
  batch_size=64 #128
  great_acc=0
  for epoch in range(epochs):
    print("epoch :",epoch)
    for batch_x,batch_y in batch_former(trainX,trainY,batch_size):
      gradient=gradCE(batch_x,batch_y, weights) 
      weights=weights-lr*gradient
    loss=fCE(testX, testY,weights)
    print("testing_loss :",loss)
    acc=cal_acc(testX, testY,weights)
    print("testing_acc:" ,acc)
    if acc>great_acc:
      good_weights=weights
      great_acc=acc
  return weights,good_weights,great_acc


     
def initWeightsAndBiases ():
  Ws = []
  bs = []
  # Strategy:
  # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
  # Initialize biases to small positive number (0.01).
  np.random.seed(0)
  W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
  Ws.append(W)
  b = 0.01 * np.ones(NUM_HIDDEN[0])
  bs.append(b)
  for i in range(NUM_HIDDEN_LAYERS - 1):
    W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[i+1])
    bs.append(b)
  W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5)- 1./NUM_HIDDEN[-1]**0.5
  Ws.append(W)
  b = 0.01 * np.ones(NUM_OUTPUT)
  bs.append(b)
  return Ws, bs

  def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, 
mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()
 

NUM_HIDDEN_LAYERS = 6
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 64 ]
NUM_OUTPUT = 10
print(NUM_HIDDEN)


# Creates an image representing the first layer of weights (W0).

if __name__ == "__main__":
    # Load training data.

    X_tr = np.load("./data/fashion_mnist_train_images.npy")
    n = X_tr.shape[0]
    # Reshaping the data
    X_tr = X_tr.reshape((n,-1))
    # Dividing the all the pixel valus of image by 255 value
    X_tr=(X_tr/255).T
    y_tr = np.load("./data/fashion_mnist_train_labels.npy")

    #one hot encoding
    new_ytr=one_hot_encoding(y_tr).T
    #loading and processing dataset for testing
    X_test = np.load("./data/fashion_mnist_test_images.npy")
    n = X_test.shape[0]
    X_test = X_test.reshape((n,-1))
    X_test=(X_test/255).T
    y_test = np.load("./data/fashion_mnist_test_labels.npy")

    new_ytest=one_hot_encoding(y_test).T

    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    Ws, bs = initWeightsAndBiases()
    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    # print(Ws[0].shape)
    # print(len(bs))
    # print(bs[0].shape)

    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    
    # print(weights)
    # gradCE (X_test, new_ytest, weights)
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).
    # print(np.atleast_2d(X_test[:,0:5]).shape)
    # print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(X_test[:,0:5]), np.atleast_2d(new_ytest[:,0:5]), weights_), \
    #                                 lambda weights_: gradCE(np.atleast_2d(X_test[:,0:5]), np.atleast_2d(new_ytest[:,0:5]), weights_), \
    #                                 weights))
    # a=scipy.optimize.approx_fprime(weights, lambda weights_: 
    #   fCE(np.atleast_2d(X_test[:,0:5]), np.atleast_2d(new_ytest[:,0:5]), weights_), 1e-6)
    # with np.printoptions(threshold=np.inf):
    #   print(weights)
    #   print(a)

    weights,good_weights,good_acc = train(X_tr, new_ytr, weights, X_test, new_ytest, 0.001)
    # show_W0(weights)
