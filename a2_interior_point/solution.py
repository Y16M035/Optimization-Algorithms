import numpy as np
import sys, time
from a1_unconstrained_solver.solution import SolverUnconstrained
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

    def solve(self):
        
        #Initialize variables for Interior Point Method
        mu = 1
        rho_mu = 0.5
        alpha = 1
        rho_ls = 0.1
        rho_alpha_minus = 0.5
        rho_alpha_plus = 1.2
        lamda = 0.01
        
        #Obtain x0, problem dimension and the feature types
        x = self.problem.getInitializationSample()
        dim = self.problem.getDimension()
        types = self.problem.getFeatureTypes()
        self.f_index = [i for i in range(len(types)) if types[i] == OT.f]
        self.sos_index = [i for i in range(len(types)) if types[i] == OT.sos]
        self.eq_index = [i for i in range(len(types)) if types[i] == OT.eq]
        self.ineq_index = [i for i in range(len(types)) if types[i] == OT.ineq]

        #Method to compute the values for the next step
        def getParameters(x):

            #Initialize Constant, Gradient and Hessian
            C = 0
            G = np.zeros(dim)    
            H = np.zeros((dim,dim))        
            phi, J = self.problem.evaluate(x)

            #Objective function
            if len(self.f_index):
                C += phi[self.f_index]
                G += J[self.f_index][0]
                H += self.problem.getFHessian(x)

            # Gauss-Newton approximation
            if len(self.sos_index):
                C += phi[self.sos_index].T @ phi[self.sos_index]
                G += 2 * J[self.sos_index].T @ phi[self.sos_index]
                H += 2 * J[self.sos_index].T @ J[self.sos_index]
            
            #Log Barrier method for inequalities
            if len(self.ineq_index):
                for i in self.ineq_index:
                    if phi[i] > 0:
                        C = np.inf
                        G = np.zeros(dim)
                        break
                    C -= mu * np.log(-phi[i])
                    G -= mu * J[i] / phi[i]
                    H += mu * (1 / phi[i]**2) * np.outer(J[i],J[i])
            
            #If H is not convex, adjust until it is
            if min(np.linalg.eigvals(H)) <= 0:
                H += (1 - lamda)*np.eye(dim)

            return C, G, H

        #Main of the program: optain optimum X
        # Loop until convergence
        it = 0
        start = time.time()
        
        #Break if it surpasses it max or time limit
        while it < 10000 and time.time() - start <= 1000:

            x2 = x
            mu *= rho_mu
            while time.time() - start <= 1000:

                #Compute to solve delta 
                C, G, H = getParameters(x)
                delta = -np.linalg.solve(H + lamda * np.eye(dim), G)

                #If Gradient * delta does not decrease or the linear system is ill-defined
                if G.T@delta > 0:
                    delta /= np.linalg.norm(delta)

                #Line-Search algorithm until it decreases enough
                while getParameters(x + alpha * delta)[0] > C + alpha * rho_ls * G.T @ delta:
                    alpha *= rho_alpha_minus   
                x = x + alpha * delta
                alpha += rho_alpha_plus

                #If the difference is unsignificant break: we have the next point
                if np.linalg.norm(alpha * delta) < 1e-4:
                    break

            #If the difference is really unsignificant break: we have the solution
            if np.linalg.norm(x2 - x) < 1e-6 :
                break

            it +=1
            
        return x
