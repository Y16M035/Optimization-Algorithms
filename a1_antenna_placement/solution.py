import sys

import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):

    def __init__(self, P, w):
        self.P = P
        self.w = w
        self.d = len(w)

    def evaluate(self, x):
        y = 0
        J = np.zeros((2,))

        for i in range(self.d):
            y = y + self.w[i] * np.exp(-np.linalg.norm(x - self.P[i]) ** 2)
            J = J + self.w[i] * np.exp(-np.linalg.norm(x - self.P[i]) ** 2) * 2 * (x - self.P[i])

        return np.array([-y]), np.array([J])

    def getDimension(self):
        return self.d

    def getUpdateHessian(self, x, exp, i, j, k):
        update = -4 * x[k] * (x[j] - self.P[i][j]) * exp + 4 * self.P[i][k] * (x[j] - self.P[i][j]) * exp
        if j == k:
            update += 2 * exp
        return update

    def getFHessian(self, x):
        H = np.zeros((2, 2))

        for i in range(self.d):
            exp = np.exp(-np.linalg.norm(x - self.P[i]) ** 2)
            for j in range(2):
                for k in range(2):
                    H[j, k] += self.w[i] * self.getUpdateHessian(x, exp, i, j, k)

        return H

    def getInitializationSample(self):
        return 1 / self.d * np.sum(self.P, axis=0)

    def getFeatureTypes(self):
        return [OT.f]
