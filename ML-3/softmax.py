import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):

  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################

  xtheta = X@theta # matrix of m x k: the (i,c)th element is theta(c).T @ x(i)
  mlist = np.max(xtheta, axis=1) # vector of m: the ith element is the maximum of theta.T @ x(i)
  xtheta1 = np.exp(xtheta-mlist[:,np.newaxis]) # trick of broadcasting!
  for i in range(m):
    for c in range(theta.shape[1]):
      J += -1/m*(y[i]==c)*(np.log(xtheta1[i,c]/np.sum(xtheta1[i,:]) ))
  J += reg/(2*m)*np.sum(np.power(theta,2))

  for c in range(theta.shape[1]):
    for i in range(m):
      grad[:,c] += -( X[i,:] * ( 1*(y[i]==c) - xtheta1[i,c]/np.sum(xtheta1[i,:]) ) )/m
  grad += reg/m*theta

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################

  xtheta = X@theta # matrix of m x k: the (i,c)th element is theta(c).T @ x(i)
  mlist = np.max(xtheta, axis=1) # vector of m: the ith element is the maximum of theta.T @ x(i)
  xtheta1 = np.exp(xtheta-mlist[:,np.newaxis]) # trick of broadcasting!
  S = np.sum(xtheta1,axis=1) # vector of m: the sum across all c of all exp(theta(c).T @ x(i))
  ind = np.zeros_like(xtheta)
  ind[np.arange(m), y] = 1
  J = -np.sum(ind*np.log(xtheta1/S[:,np.newaxis]))/m +reg/(2*m)*np.sum(np.power(theta,2))

  grad = -1/m*X.T@(ind - xtheta1/S[:,np.newaxis]) + reg/m*theta




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
