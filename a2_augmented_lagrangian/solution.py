import numpy as np
import sys
import time

from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT

sys.path.append("..")


class SolverAugmentedLagrangian(NLPSolver):

    def __init__(self):

       ""


    def solve(self):

        #Obtain x0 and the feature types
        x = self.problem.getInitializationSample()
        phi, J = self.problem.evaluate(x)
        types = self.problem.getFeatureTypes()
        
        #f, φ, h and g
        f_index = [i for i in range(len(types)) if types[i] == OT.f]
        sos_index = [i for i in range(len(types)) if types[i] == OT.sos]
        eq_index = [i for i in range(len(types)) if types[i] == OT.eq]
        ineq_index = np.array([i for i in range(len(types)) if types[i] == OT.ineq], dtype=np.int64)

        #Initialize parameters
        mu = 1
        rho_mu=1.4
        nu = 1
        rho_nu = 1.4
        alpha = 1
        rho_ls = 0.01
        rho_alpha_plus = 1.2
        rho_alpha_minus = 0.5
        lambda2 = 0.1
        lamda = np.zeros(len(ineq_index))
        kappa = np.zeros(len(eq_index))

        # Loop until convergence
        it = 0
        start = time.time()
        
        #Break if it surpasses it max or time limit
        while it < 10000 and time.time() - start <= 1000:

            #Break if it surpasses time limit
            while time.time() - start <= 1000:
                mask = ineq_index[np.logical_or(phi[ineq_index] >= 0, lamda > 0)]

                # φ*φ + mu*g'*g' + λ*g + nu*h*h + k*h (+ f);
                C = phi[sos_index].T @ phi[sos_index] + mu*phi[mask].T @ phi[mask] + lamda.T @ phi[ineq_index] + nu*phi[eq_index].T @ phi[eq_index] + kappa.T @ phi[eq_index]
                # 2dφ*φ + 2mu*dg'*g' + λ*dg + 2nu*dh*h + k (+ df)
                G = 2*J[sos_index].T @ phi[sos_index] + 2*mu*J[mask].T @ phi[mask] + lamda.T @ J[ineq_index] + 2*nu*J[eq_index].T @ phi[eq_index] + kappa.T @ J[eq_index]
                # 2dφ*dφ + 2mu*dg'*dg' + 2nu*dh*dh + λ2 (+ d2f)
                H = 2*J[sos_index].T @ J[sos_index] + 2*mu*J[mask].T @ J[mask] + 2*nu*J[eq_index].T @ J[eq_index] + lambda2 * np.identity(len(x))
                # if f has indices (+ f)
                if len(f_index):
                    C += phi[f_index] 
                    G += J[f_index][0]
                    H += self.problem.getFHessian(x)
                    
                delta = np.linalg.solve(H, -G)
                if delta @ G > 0:
                    delta = -G / np.linalg.norm(G)

                #Line-search
                C_old = C
                while C > C_old + rho_ls * alpha * G.T @ delta or np.isnan(C):
                    phi = self.problem.evaluate(x + alpha * delta)[0]
                    mask = ineq_index[np.logical_or(phi[ineq_index] >= 0, lamda > 0)]

                    # φ*φ + mu*g'*g' + λ*g + nu*h*h + k*h (+ f);
                    C = phi[sos_index].T @ phi[sos_index] + mu*phi[mask].T @ phi[mask] + lamda.T @ phi[ineq_index] + nu*phi[eq_index].T @ phi[eq_index] + kappa.T @ phi[eq_index]
                    # if f has indices (+ f)
                    if len(f_index):
                        C += phi[f_index] 

                    alpha *= rho_alpha_minus

                x2 = x
                x = x + alpha * delta
                alpha *= rho_alpha_plus
                phi, J = self.problem.evaluate(x)
                
                #Found good next x
                if np.linalg.norm(alpha * delta) < 0.01:
                    break

            #Found optimal
            if np.linalg.norm(x-x2) < 0.001:
                break
            
            #Update params for the next iteration
            lamda = np.maximum(lamda + 2*mu*phi[ineq_index], np.zeros(len(lamda)))
            kappa = 2*nu*phi[eq_index]
            mu *= rho_mu
            nu *= rho_nu
            alpha = 1
            it +=1

        return x
