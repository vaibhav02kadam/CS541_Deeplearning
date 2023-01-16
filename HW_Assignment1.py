import numpy as np
import copy
#Question no 1
def problem_1a (A, B):
  return A + B
def problem_1b (A, B, C):
  return np.dot(A,B)-C
def problem_1c (A, B, C):
  return A*B+np.transpose(C)
def problem_1d (x, y):
  return np.dot(np.transpose(x),y)
def problem_1e (A, x):
  return np.linalg.solve(A,x)
def problem_1f (A, i):
  return np.sum(A[i,::2])
def problem_1g (A, c, d):
  return np.mean(A[np.nonzero((A<=d) & (A>=c))])
def problem_1h (A, k):
  w,v=np.linalg.eig(A)
  w=np.abs(w)
  ind=np.argsort(-w)
  return v[:,ind[:k]]
def problem_1i (x, k, m, s):
  z=np.ones([len(x)])
  mean=x+np.dot(m,z)
  covar=s*np.identity(len(x))
  return np.random.multivariate_normal(mean,covar,k).transpose()
def problem_1j (A):
  ind=np.random.permutation(len(A[0]))
  return A[:,ind]
def problem_1k (x):
  mean=np.mean(x)
  std=np.std(x)
  return (x-mean)/std
def problem_1l (x, k):
  return np.repeat(np.atleast_2d(x),k,axis=-1)
def problem_1m (X, Y):
  k,n=X.shape
  k,m=Y.shape
  x=np.repeat(np.atleast_3d(X),m,axis=-1)
  y=np.repeat(np.atleast_3d(Y),n,axis=-1)
  y=np.swapaxes(y, 1, 2)
  D=np.sqrt(np.sum(np.square(x-y), axis=0))
  return D
def problem_1n (matrices):
  nums_mul=0
  for i in range(len(matrices)-1):
    nums_mul+=matrices[i].shape[0]*matrices[i].shape[1]*matrices[i+1].shape[1]
  return nums_mul
### Question No :2
def linear_regression (X_tr, y_tr):
  A=np.dot(np.transpose(X_tr),X_tr)
  X=copy.deepcopy(X_tr)
  b=np.dot(X.transpose(),y_tr)
  w=np.linalg.solve(A,b)
  return w

def train_age_regressor ():
  # Load data
  print("load data")
  X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
  new_col=np.ones((X_tr.shape[0],1))
  X_tr=np.hstack((X_tr, new_col))
  y_tr = np.load("age_regression_ytr.npy")
  w= linear_regression(X_tr, y_tr)
  # print("done")
  return w.reshape(-1,1)
w=train_age_regressor()
print(w)
X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
y_tr = np.load("age_regression_ytr.npy")
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
y_te = np.load("age_regression_yte.npy")
new_col=np.ones((X_tr.shape[0],1))
X_tr=np.hstack((X_tr, new_col))
new_col_2=np.ones((X_te.shape[0],1))
X_te=np.hstack((X_te, new_col_2))
fmse_tr=0
fmse_te=0
n=X_tr.shape[0]
m=X_te.shape[0]
for i in range(n):
  fmse_tr += (1/(2*n))*(np.dot(X_tr[i,:],w)- y_tr[i])*(np.dot(X_tr[i,:],w) - y_tr[i])
print(fmse_tr)
for i in range(m):
	fmse_te += (1/(2*m))*(np.dot(X_te[i,:],w)- y_te[i])*(np.dot(X_te[i,:],w)- y_te[i])
print(fmse_te)