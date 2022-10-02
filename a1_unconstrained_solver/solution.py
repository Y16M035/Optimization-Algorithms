import sys

import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
import time


def Error(i_f, i_sos, phi):
    e = 0
    if len(i_f) > 0:
        e += phi[i_f][0]
    if len(i_sos) > 0:
        e += phi[i_sos].T @ phi[i_sos]
    return e

class SolverUnconstrained(NLPSolver):

    def __init__(self):
        self.alpha = 1  # alpha
        self.lamda = 0.001  # lambda
        self.theta = 0.0001  # tolerance
        self.rho_ap = 1.2  # rho(alpha +)
        self.rho_am = 0.5  # rho(alpha -)
        self.rho_ls = 0.01  # rho(ls)

    def Delta(self, J, H, phi, linear):

        if linear:
            D = H + self.lamda * np.eye(H.shape[0])
        else:
            D = H

        # try to inverse D
        control = 0
        try:
            delta = -np.matmul(np.linalg.inv(D), J.T)
        except:
            control = 0
            delta = np.divide(-J, abs(J + 1e-5))
            pass

        if linear:
            delta = np.matmul(delta, phi)
            J = 2 * np.matmul(J.T, phi)

        if np.matmul(J, delta) > 0:  # Delta is positive definite
            control = 1

        if control:
            delta = np.divide(-J, abs(J + 1e-5))

        return delta

    def solve(self):

        start = time.time()

        linear = 0
        count = 0

        x = self.problem.getInitializationSample()
        ot = self.problem.getFeatureTypes()
        i_f = [i for i, x in enumerate(ot) if x == OT.f]
        i_sos = [i for i, x in enumerate(ot) if x == OT.sos]

        # Lets evaluate the problem
        phi, J = self.problem.evaluate(x)
        count = count + 1

        # Obtain Hessian
        try:
            if len(i_sos) > 0:
                NotImplementedError()
            H = self.problem.getFHessian(x)
        except NotImplementedError:
            # The Hessian is not implemented, we must approximate
            H = 2 * np.matmul(J.T, J)
            linear = 1

        # Obtain J and delta
        if J.shape[0] == 1:
            J = J[0]
        delta = self.Delta(J, H, phi, linear)

        # until the infinite norm of alpha * delta is smaller than the tolerance
        while np.linalg.norm(self.alpha * delta, np.inf) > self.theta:

            # Check the count and time limits
            if count > 1000 or time.time() - start > 1000:
                break

            # obtain new x, phi and J(acobian); update count
            x2 = x + self.alpha * delta
            phi2, J2 = self.problem.evaluate(x2)
            count = count + 1

            # Repeat procedure to obtain Hessian, save new Jacobian
            try:
                if len(i_sos) > 0:
                    NotImplementedError()
                H2 = self.problem.getFHessian(x2)  # if necessary
            except NotImplementedError:
                H2 = 2 * np.matmul(J2.T, J2)
                linear = 1

            if J2.shape[0] == 1:
                J = J2[0]
                J2 = J2[0]
            else:
                J = J2

            e = Error(i_f, i_sos, phi)
            e2 = Error(i_f, i_sos, phi2)
            phi = phi2
            if linear:
                J = 2 * np.matmul(J.T, phi)

            # while the error > old error + rho_ls * J * (alpha + delta) keep decreasing alpha and recomputing
            while e2 > (e + self.rho_ls * np.matmul(J.T, (self.alpha * delta))):
                if count > 1000 or (time.time() - start) > 1000:
                    break

                # update alpha, delta, phi, newer x and get new phi and Jacobian
                self.alpha = self.rho_am * self.alpha
                delta = self.Delta(J2, H2, phi2, linear)
                phi = phi2
                x3 = x2 + self.alpha * delta
                phi2, J2 = self.problem.evaluate(x3)
                count = count + 1

                # Obtain Hessian
                try:
                    if len(i_sos) > 0:
                        NotImplementedError()
                    H2 = self.problem.getFHessian(x3)
                except NotImplementedError:  # Approximated
                    H2 = 2 * np.matmul(J2.T, J2)
                    linear = 1

                if J2.shape[0] == 1:
                    J2 = J2[0]

                # update errors and stablish newer x
                e = e2
                e2 = Error(i_f, i_sos, phi2)
                x2 = x3

            x = x2
            self.alpha = min(self.rho_ap * self.alpha, 1)
            delta = self.Delta(J2, H2, phi2, linear)

        return x
