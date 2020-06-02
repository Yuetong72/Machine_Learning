import numpy as np
import math

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass

# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        # Box-Muller transform: https://en.wikipedia.org/wiki/Box-Muller_transform
        z0 = math.sqrt( -2.0 * math.log(u1)) * math.cos( 2.0 * math.pi * u2)
        # z1 = math.sqrt( -2.0 * math.log(u1)) * math.sin( 2.0 * math.pi * u2)
        return z0 * self.sigma + self.mu
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu
        self.Sigma = Sigma

    def sample(self):
        n = self.Mu.size
        z = np.array([])
        for i in range(n):
            z = np.append(z,UnivariateNormal(0,1).sample())
        z = z.reshape(n,1)
        A = np.linalg.cholesky(self.Sigma)
        return np.dot( A, z) + self.Mu


# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap        

    def sample(self):
        n = self.ap.size
        u1 = np.random.uniform()
        cum_prob = [0]
        for i in range(n):
            cum_prob.append( cum_prob[i] + self.ap[i] )
        for i in range(n):
            if( (cum_prob[i] <= u1 ) & ( u1 <= cum_prob[i+1])):
                return i+1
                break    


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm

    def sample(self):
        pm_idx = Categorical(self.ap).sample() -1
        return MultiVariateNormal(self.pm[pm_idx][0], self.pm[pm_idx][1]).sample()

        
import matplotlib.pyplot as plt
import numpy as np
Mu = np.array([[1],[1]])
Sigma = np.array([[1, 0.5], [0.5, 1]])

x = MultiVariateNormal(Mu, Sigma).sample()
for i in range(1000):
    x = np.hstack( ( x, MultiVariateNormal(Mu, Sigma).sample()))
    
plt.scatter( x[0,:],x[1,:])
plt.show()