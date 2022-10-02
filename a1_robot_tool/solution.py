import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):

    def __init__(self, q0, pr, l):
        self.q0 = q0
        self.pr = pr
        self.l = l

    def evaluate(self, x):

        # Formulate p1, p2 given in the exercise, P = (p1,p2) [x=q]
        p1 = np.cos(x[0]) + 0.5 * np.cos(x[0] + x[1]) + 1 / 3 * np.cos(x[0] + x[1] + x[2])
        p2 = np.sin(x[0]) + 0.5 * np.sin(x[0] + x[1]) + 1 / 3 * np.sin(x[0] + x[1] + x[2])
        p = np.array([p1, p2])

        # Formulate optimization objective given in the exercise, norm(p(q) -p*) and lambda * (q - q0)
        y = np.zeros((2,))
        y[0] = np.linalg.norm(p - self.pr)
        y[1] = np.sqrt(self.l) * np.linalg.norm(x - self.q0)

        # Derive p1 based on x0, x1, x2
        dp1_dx0 = -np.sin(x[0]) - 0.5 * np.sin(x[0] + x[1]) - 1 / 3 * np.sin(x[0] + x[1] + x[2])
        dp1_dx1 = -0.5 * np.sin(x[0] + x[1]) - 1 / 3 * np.sin(x[0] + x[1] + x[2])
        dp1_dx2 = - 1 / 3 * np.sin(x[0] + x[1] + x[2])

        # Derive p1 based on x0, x1, x2
        dp2_dx0 = np.cos(x[0]) + 0.5 * np.cos(x[0] + x[1]) + 1 / 3 * np.cos(x[0] + x[1] + x[2])
        dp2_dx1 = 0.5 * np.cos(x[0] + x[1]) + 1 / 3 * np.cos(x[0] + x[1] + x[2])
        dp2_dx2 = 1 / 3 * np.cos(x[0] + x[1] + x[2])

        # Get Jacobian
        J = np.zeros((2, 3))
        J[0, 0] = np.matmul((p - self.pr), np.array([dp1_dx0, dp2_dx0])) / (np.linalg.norm(p - self.pr)+1e-10)
        J[0, 1] = np.matmul((p - self.pr), np.array([dp1_dx1, dp2_dx1])) / (np.linalg.norm(p - self.pr)+1e-10)
        J[0, 2] = np.matmul((p - self.pr), np.array([dp1_dx2, dp2_dx2])) / (np.linalg.norm(p - self.pr)+1e-10)
        for i in range(3):
            J[1, i] = np.sqrt(self.l) * (x[i] - self.q0[i]) / (np.linalg.norm(x - self.q0)+1e-10)

        return y, J

    def getDimension(self):
        return len(self.q0)

    def getInitializationSample(self):
        return self.q0

    def getFeatureTypes(self):
        return [OT.sos] * 5
