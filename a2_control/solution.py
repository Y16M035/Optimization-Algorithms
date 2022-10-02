import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    """
    Parameters
    K integer
    A in R^{n x n}
    B in R^{n x n}
    Q in R^{n x n} symmetric
    R in R^{n x n} symmetric
    yf in R^n

    Variables
    y[k] in R^n for k=1,...,K
    u[k] in R^n for k=0,...,K-1

    Optimization Problem:
    LQR with terminal state constraint

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=0}^{K-1}      u [k].T R u [k]
    s.t.
    y[1] - Bu[0]  = 0
    y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
    y[K] - yf = 0

    Hint: Use the optimization variable:
    x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

    Use the following features:
    1 - a single feature of types OT.f
    2 - the features of types OT.eq that you need
    """

    def __init__(self, K, A, B, Q, R, yf):
        #based 
        self.K = K
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.yf = yf
        n = len(yf)
        self.n = n
        self.dim = 2*K*n
        self.H_dyn = np.zeros((K*n, self.dim))

        # so that later we can easily compute y[1] - B*u[0] = 0, y[k+1] - A*y[k] - B*u[k] for k = 1,...,K-1
        for t in range(K):
            cur = 2*n*t
            if t> 0: # skipping A[0] so y[1] - B*u[0] = 0
                self.H_dyn[ n*t : n*t + n, cur-n : cur] = -A
            self.H_dyn[ n*t : n*(t + 1), cur : cur+n] = -B
            self.H_dyn[ n*t : n*(t + 1), cur+n : cur+2*n] = np.eye(n)

    def evaluate(self, x):
        
        #so I dont have to re-reshape later (y[k+1] - A*y[k] - B*u[k])
        eq_dyn = self.H_dyn @ x

        #x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]
        x = x.reshape((2*self.K, self.n))
        u = x[::2]
        y = x[1::2]

        #Find phi, J for f
        phi_f = 0
        Jf = np.zeros((self.dim))
        for i in range(self.K):
            #1/2 * sum_{k=1}^{K} * y[k].T * Q y[k] + 1/2 * sum_{k=0}^{K-1}  u [k].T * R * u[k]
            phi_f+= 1/2 *u[i].T @ self.R @ u[i] + 1/2 *y[i].T @ self.Q @ y[i]
            #Compute d(u [k].T R u [k]) = R * y[i] and save in Jf
            Jf[2*i * self.n : (2*i+1) * self.n] = self.R @ u[i]
            #Compute d(y[k].T Q y[k]) = Q * y[i] and save in Jf
            Jf[(2*i+1) * self.n : (2*i+2) * self.n] = self.Q @ y[i]

        #From the minimal_fuel example  
        Jf = np.reshape(Jf,(1,-1))
        #J_dyn = d(H_dyn*x)= H_dyn
        J_dyn = self.H_dyn
        #J_des = d(H_dyn*x)= H_dyn
        J_des = np.zeros((self.n, self.dim))
        J_des[:,-self.n:] = np.eye(self.n)
        
        # y[K] - yf = 0
        eq_des = y[self.K-1,:] - self.yf
        
        # Concat all of the results
        phi = np.concatenate(([phi_f], eq_dyn, eq_des))
        J = np.concatenate([Jf, J_dyn, J_des], axis=0)
        
        return  phi, J


    def getFHessian(self, x):

        H = np.zeros((self.dim, self.dim))

        for i in range(self.K):
            cur = 2*i*self.n
            #Compute d(R * y[i]) = R and save in H
            H[cur : cur+self.n , cur : cur+self.n] = self.R
            #Compute d(Q * y[i]) = Q and save in H
            H[cur + self.n : cur + 2*self.n , cur + self.n: cur + 2*self.n] = self.Q

        return H

    def getDimension(self):
        return self.dim

    def getInitializationSample(self):
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        return [OT.f] + [OT.eq] * (self.n * (self.K+1))
