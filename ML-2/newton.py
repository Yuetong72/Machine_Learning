import numpy as np

x = np.array([0,3,1,3,0,1,1,1])
x = x.reshape((4,2))
m,d = x.shape
X = np.hstack([np.ones((m,1)),x])
y = np.array([1,1,0,0])
theta = np.array([0,-2,1])
lam = 0.07

def h(theta,x):
    return 1/(1+np.exp(-theta.T@x))

for ind in range(2):
    H0 = np.array([])

    for i in range(m):
        H0 = np.append(H0,h(theta,X[i,:]))
    
    H0 = H0 - y

    S = np.array([])
    for i in range(m):
        S = np.append(S, h(theta,X[i,:])*(1-h(theta,X[i,:])) )
    
    S = np.diag(S)

    theta = theta - np.linalg.inv(X.T @ \
        S @ X + lam* np.diag( np.append([0],np.repeat(1,d)) ) ) \
            @ (X.T @ H0 + lam * np.append([0],theta[1:]).T )
    print(theta)

