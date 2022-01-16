import numpy as np
from cvxopt import matrix, solvers
import utilities
import mosek


""" Classifier """
class CSVC:
    def __init__(self, model_info):

        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Store Necessary Data Used Across All Methods #
        self.trainXSV = []
        self.trainZSV = []
        self.alphaSV = []
        self.b = 0

        self.modelNumber = 0
        self.evaluationMetric = dict(F1_Score=[], Precision=[], Recall=[], ConfusionMatrix=[], Accuracy=[])

    def train(self, trainData):
        trainX, trainZ = trainData[:, :-1], trainData[:, -1]
        psi = self.kernelObj.computeGram( trainX, trainX, self.modelInfo['C'], penalize=self.modelInfo['penalize'])

        n_train = len(trainX)

        ineq = {
            'ineqG': {
                'L1': matrix(np.vstack((-np.eye(n_train), np.eye(n_train))), tc='d'),
                'L2': matrix(np.vstack((-np.eye(n_train), -np.eye(n_train))), tc='d')
            },
            'ineqh': {
                'L1': matrix(
                    np.hstack((np.zeros(n_train), np.ones(n_train) * self.modelInfo['C'])).reshape(-1, 1)),
                'L2': matrix(np.zeros((2 * n_train, 1)))
            }
        }

        P = matrix(np.outer(trainZ, trainZ) * psi)
        q = matrix(-1 * np.ones(n_train))
        eqA = matrix(trainZ.reshape(1, n_train))
        eqb = matrix(0.0)
        G = ineq['ineqG'][self.modelInfo['penalize']]
        h = ineq['ineqh'][self.modelInfo['penalize']]

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, eqA, eqb)
        alpha = np.ravel(sol['x'])
        # Lagrange multipliers
        # Non - Zero Alphas are the support vectors
        idxSV = abs(alpha) > 1e-6
        self.alphaSV = alpha[idxSV]

        self.trainXSV = trainX[idxSV]
        self.trainZSV = trainZ[idxSV]
        sv_mid_i = \
            np.where(
                min((abs(abs(alpha) - (self.modelInfo['C'] / 2)))) == (abs(abs(alpha) - (self.modelInfo['C'] / 2))))[
                0][0]

        self.b = trainZ[sv_mid_i] - np.sum([(self.trainZSV[i] * self.alphaSV[i] * self.kernelObj.kernelFxn(self.trainXSV[i], trainX[sv_mid_i])) for i in range(len(self.alphaSV))])

    def predict(self, testX):
        predPsi = self.kernelObj.computeGram( testX, self.trainXSV, self.modelInfo['C'], self.modelInfo['penalize'])
        predicted = np.array(
            [np.sum(self.alphaSV.reshape(-1, 1) * self.trainZSV.reshape(-1, 1) * predPsi[i].reshape(-1, 1)) for i in range(len(testX))]) + self.b
        predSign = np.array([np.sign(i) for i in predicted])
        return predSign

""" \epsilon-SVR : CVXOPT """
class EpsilonSVR:
    def __init__(self, model_info):
        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Store Necessary Data Used Across All Methods #
        self.trainXSV = []
        self.alphaSV = []
        self.b = 0

        self.modelNumber = 0
        # Uneccesary Storage, used to access and asses methods #
        # self.alpha = []
        # self.trainYSV = []
        # self.midSV = []

    def train(self, trainData):
        trainX, trainZ = trainData[:, :-1], trainData[:, -1]
        psi = self.kernelObj.computeGram( trainX, trainX, self.modelInfo['C'], penalize=self.modelInfo['penalize'])

        n_train = len(trainX)

        ineq = {
            'ineqG': {
                'L1': matrix(np.vstack((-np.eye(2 * n_train), np.eye(2 * n_train))), tc='d'),
                'L2': matrix(np.vstack((-np.eye(2 * n_train), -np.eye(2 * n_train))), tc='d')
            },
            'ineqh': {
                'L1': matrix(
                    np.hstack((np.zeros(2 * n_train), np.ones(2 * n_train) * self.modelInfo['C'])).reshape(-1, 1)),
                'L2': matrix(np.zeros((4 * n_train, 1)))
            }
        }

        P = matrix(np.hstack((np.vstack((psi, -psi)), np.vstack((-psi, psi)))))
        q = matrix(np.array([[(self.modelInfo['modelParam'] - trainZ[i]) for i in range(n_train)],
                             [(self.modelInfo['modelParam'] + trainZ[j]) for j in range(n_train)]]).reshape(-1, 1))
        eqA = matrix(np.array([1 if x <= n_train - 1 else -1 for x in range(2 * n_train)]).reshape(1, -1), tc='d')
        eqb = matrix(0.0)
        G = ineq['ineqG'][self.modelInfo['penalize']]
        h = ineq['ineqh'][self.modelInfo['penalize']]
        solvers.options['mosek'] = {mosek.iparam.log: 0}
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, eqA, eqb)
        alpha_vec = np.ravel(np.array(sol['x']))

        # Lagrange multipliers
        alpha = alpha_vec[:n_train] - alpha_vec[n_train:2 * n_train]

        idxSV = abs(alpha) > 1e-6
        self.alphaSV = alpha[idxSV]
        sv_mid_i = \
            np.where(
                min((abs(abs(alpha) - (self.modelInfo['C'] / 2)))) == (abs(abs(alpha) - (self.modelInfo['C'] / 2))))[
                0][0]

        self.trainXSV = trainX[idxSV]
        # trainZSV = trainZ[idxSV] # Unused Variable

        # Calculating b
        self.b = trainZ[sv_mid_i] - (self.modelInfo['modelParam'] * np.sign(alpha[sv_mid_i])) - np.dot(alpha[idxSV],
                                                                                                       psi[
                                                                                                           idxSV, sv_mid_i])

    def predict(self, testX):
        predPsi = self.kernelObj.computeGram( testX, self.trainXSV, self.modelInfo['C'], penalize='L1')
        predicted = np.array(
            [np.sum(self.alphaSV.reshape(-1, 1) * predPsi[i].reshape(-1, 1)) for i in range(len(testX))]) + self.b
        return predicted

""" Least-Squares """
class LSSVR:

    def __init__(self, model_info):
        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Store Necessary Data Used Across All Methods #
        self.trainX = []
        self.bias = []
        self.alpha = []

        self.modelNumber = 0

    def train(self, trainData):
        self.trainX, trainZ = trainData[:, :-1], trainData[:, -1]
        psi = self.kernelObj.computeGram( self.trainX, self.trainX, self.modelInfo['C'], penalize='L2')

        A = np.vstack((np.append(0, np.ones((1, len(self.trainX)))), np.hstack((np.ones((len(self.trainX), 1)), psi))))
        b = np.append(0, trainZ).reshape(-1, 1)
        x_sol = np.linalg.solve(A, b)
        self.bias = x_sol[0]
        self.alpha = x_sol[1:]

    def predict(self, testX):
        predPsi = self.kernelObj.computeGram( testX, self.trainX, self.modelInfo['C'], penalize='L1')
        predicted = np.array(
            [np.sum(self.alpha.reshape(-1, 1) * predPsi[i].reshape(-1, 1)) for i in range(len(testX))]) + self.bias
        return predicted

""" \epsilon-SVR : Sequential Minimization Optimizer """
class EpsilonSMO:

    def __init__(self, model_info):
        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Initialization #
        self.trainX, self.trainZ = [], []
        self.trainN = 0
        self.activeSet = []
        self.activeSize = 0
        self.alpha = []  # Has size (2, N) where N = len(trainData)
        self.alpha_ = []
        self.alphaStatus = []
        self.qLinear = []  # Has size (2, N) where N = len(trainData), this is the linear term in the Objective fxn

        self.delF = []  # Gradient
        self.Gbar = []
        self.QD = []
        self.y = []
        self.rho = 0
        self.obj = 0

        self.shrinking = True
        self.unshrink = False

        self.iter = 0
        self.maxIter = 1000
        self.counter = 0
        self.b = 0
        self.modelNumber = 0

        self.tau = 1e-12
        self.eps = 1e-3

    @staticmethod
    def swap(x, y):
        temp = np.copy(x)
        x = y
        y = float(temp)
        return x, y

    def computeQ(self, i, length):
        Q_i = np.zeros((length))
        for j in range(length):
            Q_i[j] = self.y[i] * self.y[j] * self.kernelObj.kernelFxn(self.trainX[i], self.trainX[j])
        return Q_i

    def updateAlphaStatus(self, idx):
        if self.alpha[idx] >= self.modelInfo['C']:
            """ Upper Bound """
            self.alphaStatus[idx] = 1
        elif self.alpha[idx] <= 0:
            """ Lower Bound """
            self.alphaStatus[idx] = 2
        else:
            """ Free """
            self.alphaStatus[idx] = 3

    def workingSet(self):

        """ LIBSVM WSS routine """
        objDiffMin = np.inf
        Gmax = -np.inf
        Gmax2 = -np.inf
        GmaxIdx = -1;
        GminIdx = -1;
        # Routine to get i (GmaxIdx) and j (GminIdx)
        for t in range(self.trainN):
            if self.y[t] == 1:
                """ if not self.upperBound """
                if self.alphaStatus[t] != 1:
                    if -self.delF[t] >= Gmax:
                        Gmax = -self.delF[t]
                        GmaxIdx = t
            else:
                """ if not self.lowerBound """
                if self.alphaStatus[t] != 2:
                    if self.delF[t] >= Gmax:
                        Gmax = self.delF[t]
                        GmaxIdx = t

        i = GmaxIdx

        if i != -1:
            Q_i = self.computeQ(i, self.trainN)

        for j in range(self.trainN):
            if self.y[j] == 1:
                if self.alphaStatus[j] != 2:
                    gradDiff = Gmax + self.delF[j]
                    if self.delF[j] >= Gmax2:
                        Gmax2 = self.delF[j]
                    if gradDiff > 0:
                        quadCoeff = self.QD[i] + self.QD[j] - (2 * self.y[i] * Q_i[j])
                        if quadCoeff > 0:
                            objDiff = -(gradDiff * gradDiff) / quadCoeff
                        else:
                            objDiff = -(gradDiff * gradDiff) / self.tau
                        if objDiff <= objDiffMin:
                            GminIdx = j
                            objDiffMin = objDiff
            else:
                if self.alphaStatus[j] != 1:
                    gradDiff = Gmax - self.delF[j]
                    if -self.delF[j] >= Gmax2:
                        Gmax2 = -self.delF[j]
                    if gradDiff > 0:
                        quadCoeff = self.QD[i] + self.QD[j] + (2 * self.y[i] * Q_i[j])
                        if quadCoeff > 0:
                            objDiff = -(gradDiff * gradDiff) / quadCoeff
                        else:
                            objDiff = -(gradDiff * gradDiff) / self.tau
                        if objDiff <= objDiffMin:
                            GminIdx = j
                            objDiffMin = objDiff

        if (Gmax + Gmax2 < self.eps) or (GminIdx == -1):
            return []

        return [GmaxIdx, GminIdx]

    def updateAlpha(self, i, j):

        Q_i = self.computeQ(i, self.activeSize)
        Q_j = self.computeQ(j, self.activeSize)
        oldAlphaI = np.copy(self.alpha[i])
        oldAlphaJ = np.copy(self.alpha[j])

        if self.y[i] != self.y[j]:

            quadCoeff = self.QD[i] + self.QD[j] + (2 * Q_i[j])
            if quadCoeff <= 0:
                quadCoeff = self.tau

            delta = (-self.delF[i] - self.delF[j]) / quadCoeff
            diff = self.alpha[i] - self.alpha[j]
            self.alpha[i] += delta
            self.alpha[j] += delta

            if diff > 0:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = diff

                if self.alpha[i] > self.modelInfo['C']:
                    self.alpha[i] = self.modelInfo['C']
                    self.alpha[j] = self.modelInfo['C'] - diff
            else:
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = -diff

                if self.alpha[j] > self.modelInfo['C']:
                    self.alpha[j] = self.modelInfo['C']
                    self.alpha[i] = self.modelInfo['C'] + diff

        else:
            quadCoeff = self.QD[i] + self.QD[j] - (2 * Q_i[j])
            if quadCoeff <= 0:
                quadCoeff = self.tau
            delta = (self.delF[i] - self.delF[j]) / quadCoeff
            Sum = self.alpha[i] + self.alpha[j]
            self.alpha[i] -= delta
            self.alpha[j] += delta

            if Sum > self.modelInfo['C']:
                if self.alpha[i] > self.modelInfo['C']:
                    self.alpha[i] = self.modelInfo['C']
                    self.alpha[j] = Sum - self.modelInfo['C']
                if self.alpha[j] > self.modelInfo['C']:
                    self.alpha[j] = self.modelInfo['C']
                    self.alpha[i] = Sum - self.modelInfo['C']
            else:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = Sum
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = Sum

        deltaAlphaI = self.alpha[i] - oldAlphaI
        deltaAlphaJ = self.alpha[j] - oldAlphaJ

        """ Update Gradient """
        oldF = np.copy(self.delF)
        for k in range(self.activeSize):
            self.delF[k] = oldF[k] + ((Q_i[k] * deltaAlphaI) + (Q_j[k] * deltaAlphaJ))

        """ Update Alpha Status and Gbar """
        if self.alphaStatus[i] == 1:
            oldStatusI = True
        else:
            oldStatusI = False
        if self.alphaStatus[j] == 1:
            oldStatusJ = True
        else:
            oldStatusJ = False

        self.updateAlphaStatus(i)
        self.updateAlphaStatus(j)

        if self.alphaStatus[i] == 1:
            newStatusI = True
        else:
            newStatusI = False
        if self.alphaStatus[j] == 1:
            newStatusJ = True
        else:
            newStatusJ = False

        if oldStatusI != newStatusI:
            Q_i = self.computeQ(i, self.trainN)
            oldGbar = np.copy(self.Gbar)
            if oldStatusJ:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] - (self.modelInfo['C'] * Q_i[k])
            else:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] + (self.modelInfo['C'] * Q_i[k])

        if oldStatusJ != newStatusJ:
            Q_j = self.computeQ(j, self.trainN)
            oldGbar = np.copy(self.Gbar)
            if oldStatusJ:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] - (self.modelInfo['C'] * Q_j[k])
            else:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] + (self.modelInfo['C'] * Q_j[k])

    def updateRho(self):
        ub = np.inf
        lb = -np.inf
        nFree = 0
        sumFree = 0
        for i in range(self.trainN):
            yG = self.y[i] * self.delF[i]

            if self.alphaStatus[i] == 1:
                if self.y[i] == -1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            elif self.alphaStatus[i] == 2:
                if self.y[i] == 1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            else:
                nFree += 1
                sumFree += yG
        if nFree > 0:
            self.rho = sumFree / nFree
        else:
            self.rho = (ub + lb) / 2

    def beShrunk(self, i, Gmax1, Gmax2):
        if self.alphaStatus[i] == 1:
            if self.y[i] == 1:
                if -self.delF[i] > Gmax1:
                    return 1
            else:
                if -self.delF[i] > Gmax2:
                    return 1
        elif self.alphaStatus[i] == 2:
            if self.y[i] == 1:
                if self.delF[i] > Gmax2:
                    return 1
                else:
                    if self.delF[i] > Gmax1:
                        return 1
        else:
            return 0

    def shrink(self):
        Gmax1 = -np.inf
        Gmax2 = -np.inf

        """ Finding Maximal Violating Pair """
        for i in range(self.activeSize):
            if self.y[i] == 1:
                if self.alphaStatus[i] != 1:
                    if -self.delF[i] >= Gmax1:
                        Gmax1 = -self.delF[i]
                if self.alphaStatus[i] != 2:
                    if self.delF[i] >= Gmax2:
                        Gmax2 = self.delF[i]
            else:
                if self.alphaStatus[i] != 1:
                    if -self.delF[i] >= Gmax2:
                        Gmax2 = -self.delF[i]
                if self.alphaStatus[i] != 2:
                    if self.delF[i] >= Gmax1:
                        Gmax1 = self.delF[i]

        if not self.unshrink and (Gmax1 + Gmax2 <= self.eps * 10):
            self.unshrink = True
            self.reconstructGradient()
            self.activeSize = self.trainN

        for i in range(self.activeSize):
            if self.beShrunk(i, Gmax1, Gmax2) == 1:
                self.activeSize -= 1
                while self.activeSize > i:
                    if self.beShrunk(self.activeSize, Gmax1, Gmax2) == 0:
                        self.swap(self.QD[i], self.QD[self.activeSize])
                        self.swap(self.y[i], self.y[self.activeSize])
                        self.swap(self.delF[i], self.delF[self.activeSize])
                        self.swap(self.alphaStatus[i], self.alphaStatus[self.activeSize])
                        self.swap(self.alpha[i], self.alpha[self.activeSize])
                        self.swap(self.qLinear[i], self.qLinear[self.activeSize])
                        self.swap(self.activeSet[i], self.activeSet[self.activeSize])
                        self.swap(self.Gbar[i], self.Gbar[self.activeSize])
                        break
                self.activeSize -= 1

    def reconstructGradient(self):
        if self.activeSize == self.trainN: return

        freeN = 0
        for j in range(self.activeSize, self.trainN):
            self.delF[j] = self.Gbar[j] + self.qLinear[j]

        for j in range(self.activeSize):
            if self.alphaStatus[j] == 3:
                freeN += 1

        if (self.trainN * freeN) > 2 * self.activeSize * (self.trainN - self.activeSize):
            for i in range(self.activeSize, self.trainN):
                Q_i = self.computeQ(i, self.activeSize)
                oldF = np.copy(self.delF)
                for j in range(self.activeSize):
                    if self.alphaStatus[j] == 3:
                        self.delF[j] = oldF[j] + (self.alpha[j] * Q_i[j])
        else:
            for i in range(self.activeSize):
                if self.alphaStatus[i] == 3:
                    Q_i = self.computeQ(i, self.trainN)
                    oldF = np.copy(self.delF)
                    for j in range(self.activeSize, self.trainN):
                        self.delF[j] = oldF[j] + (self.alpha[i] * Q_i[j])

        pass

    def train(self, trainData):
        self.qLinear = np.vstack((self.modelInfo['modelParam'] - trainData[:, -1].reshape(-1, 1),
                                  self.modelInfo['modelParam'] + trainData[:, -1].reshape(-1, 1)))
        self.y = np.hstack((np.ones(int(len(trainData))), -1 * np.ones(int(len(trainData)))))

        wholeData = np.vstack((trainData, trainData))
        self.trainN = len(wholeData)
        self.trainX, self.trainZ = wholeData[:, :-1], wholeData[:, -1]

        self.alpha = np.zeros(self.trainN)

        """ Initialize Alpha Status """
        self.alphaStatus = np.zeros(self.trainN)
        for i in range(len(self.alpha)):
            self.updateAlphaStatus(i)

        """ Initialize Active Set """
        self.activeSet = np.arange(self.trainN)
        self.activeSize = len(self.activeSet)

        """ Initialize Gradient (delF) and Gbar """
        Psi = self.kernelObj.computeGram( self.trainX, self.trainX, self.modelInfo['C'],)
        self.QD = np.diagonal(Psi)
        self.delF = np.copy(self.qLinear).ravel()
        self.Gbar = np.zeros(self.trainN)
        for i in range(self.trainN):
            if self.alphaStatus[i] != 2:
                Q_i = self.computeQ(i, self.trainN)
                oldF = np.copy(self.delF)
                for j in range(self.trainN):
                    self.delF[j] = oldF[j] + (self.alpha[i] * Q_i[j])
                if self.alphaStatus[i] == 1:
                    self.Gbar[j] = self.modelInfo['C'] * Q_i[j]

        self.counter = min(self.trainN, 1000) + 1
        while self.iter < self.maxIter:
            # print('Iteration: {}'.format(self.iter))
            # print('Alpha: {}'.format(self.alpha.ravel()))
            decrement = self.counter - 1
            if decrement == 0:
                self.counter = min(self.trainN, 1000)
                if self.shrinking: self.shrink()

            pair = self.workingSet()
            if not pair:
                self.reconstructGradient()
                self.activeSize = self.trainN
                pair = self.workingSet()
                if not pair:
                    break
                else:
                    self.counter = 1

            i, j = pair[0], pair[1]
            self.updateAlpha(i, j)

            self.iter += 1

        if self.iter >= self.maxIter:
            if self.activeSize < self.trainN:
                self.reconstructGradient()
                self.activeSize = self.trainN
            # print("Max Iterations Reached")

        """ Update rho """
        self.updateRho()

        """ Calculate Objective Value """
        self.obj = np.dot(self.alpha.reshape(1, -1), (self.delF.reshape(-1, 1) + self.qLinear.reshape(-1, 1))) / 2
        self.alphaFinal = self.alpha[:int(self.trainN / 2)] - self.alpha[int(self.trainN / 2):self.trainN]

        self.alpha_ = np.zeros(len(self.activeSet))
        for k in range(len(self.activeSet)):
            self.alpha_[self.activeSet[k]] = self.alpha[k]

        svIdx = abs(self.alphaFinal) > 0
        self.alphaSV = self.alphaFinal[svIdx.ravel()]
        self.trainX = trainData[:, :-1]
        self.trainXSV = self.trainX[svIdx.ravel()]

    def predict(self, testX):
        predPsi = self.kernelObj.computeGram( testX, self.trainXSV, self.modelInfo['C'], self.modelInfo['penalize'])
        predicted = np.array(
            [np.sum(self.alphaSV.reshape(-1, 1) * predPsi[i].reshape(-1, 1)) for i in range(len(testX))]) - self.rho
        return predicted

""" \nu-SVR : Sequential Minimization Optimizer """
class NuSMO:

    def __init__(self, model_info):
        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Initialization #
        self.trainX, self.trainZ = [], []
        self.trainN = 0
        self.activeSet = []
        self.activeSize = 0
        self.alpha = []
        self.alpha_ = []
        self.alphaStatus = []
        self.qLinear = []

        self.delF = []
        self.Gbar = []
        self.QD = []
        self.y = []
        self.rho = 0
        self.obj = 0

        self.shrinking = True
        self.unshrink = False

        self.alphaFinal = []
        self.alphaSV = []
        self.trainXSV = []

        self.iter = 0
        self.maxIter = 500
        self.counter = 0
        self.epsilon = 0
        self.b = 0
        self.modelNumber = 0

        self.tau = 1e-12
        self.eps = 0.001

    @staticmethod
    def swap(x, y):
        temp = np.copy(x)
        x = y
        y = float(temp)
        return x, y

    def computeQ(self, i, length):
        Q_i = np.zeros(length)
        for j in range(length):
            Q_i[j] = self.y[i] * self.y[j] * self.kernelObj.kernelFxn(self.trainX[i], self.trainX[j])
        return Q_i

    def updateAlphaStatus(self, idx):
        if self.alpha[idx] >= self.modelInfo['C']:
            """ Upper Bound """
            self.alphaStatus[idx] = 1
        elif self.alpha[idx] <= 0:
            """ Lower Bound """
            self.alphaStatus[idx] = 2
        else:
            """ Free """
            self.alphaStatus[idx] = 3

    def workingSet(self):

        objDiffMin = np.inf
        GmaxP = -np.inf
        GmaxP2 = -np.inf
        GmaxPidx = -1;

        GmaxN = -np.inf
        GmaxN2 = -np.inf
        GmaxNidx = -1;
        GminIdx = -1;

        for t in range(self.activeSize):
            if self.y[t] == 1:
                """ if not self.upperBound """
                if self.alphaStatus[t] != 1:
                    if -self.delF[t] >= GmaxP:
                        GmaxP = -self.delF[t]
                        GmaxPidx = t
            else:
                """ if not self.lowerBound """
                if self.alphaStatus[t] != 2:
                    if self.delF[t] >= GmaxN:
                        GmaxN = self.delF[t]
                        GmaxNidx = t

        iP = GmaxPidx
        iN = GmaxNidx

        if iP != -1:
            Q_iP = self.computeQ(iP, self.activeSize)
        if iN != -1:
            Q_iN = self.computeQ(iN, self.activeSize)

        for j in range(self.activeSize):
            if self.y[j] == 1:
                if self.alphaStatus[j] != 2:
                    gradDiff = GmaxP + self.delF[j]
                    if self.delF[j] >= GmaxP2:
                        GmaxP2 = self.delF[j]
                    if gradDiff > 0:
                        quadCoeff = self.QD[iP] + self.QD[j] - (2 * self.y[iP] * Q_iP[j])
                        if quadCoeff > 0:
                            objDiff = -(gradDiff * gradDiff) / quadCoeff
                        else:
                            objDiff = -(gradDiff * gradDiff) / self.tau
                        if objDiff <= objDiffMin:
                            GminIdx = j
                            objDiffMin = objDiff
            else:
                if self.alphaStatus[j] != 1:
                    gradDiff = GmaxN - self.delF[j]
                    if -self.delF[j] >= GmaxN2:
                        GmaxN2 = -self.delF[j]
                    if gradDiff > 0:
                        quadCoeff = self.QD[iN] + self.QD[j] + (2 * self.y[iN] * Q_iN[j])
                        if quadCoeff > 0:
                            objDiff = -(gradDiff * gradDiff) / quadCoeff
                        else:
                            objDiff = -(gradDiff * gradDiff) / self.tau
                        if objDiff <= objDiffMin:
                            GminIdx = j
                            objDiffMin = objDiff

        if (max(GmaxP + GmaxP2, GmaxN + GmaxN2) < self.eps) or (GminIdx == -1):
            return []

        if self.y[GminIdx] == 1:
            return [GmaxPidx, GminIdx]
        else:
            return [GmaxNidx, GminIdx]

    def updateAlpha(self, i, j):

        Q_i = self.computeQ(i, self.activeSize)
        Q_j = self.computeQ(j, self.activeSize)
        oldAlphaI = np.copy(self.alpha[i])
        oldAlphaJ = np.copy(self.alpha[j])

        if self.y[i] != self.y[j]:

            quadCoeff = self.QD[i] + self.QD[j] + (2 * Q_i[j])
            if quadCoeff <= 0:
                quadCoeff = self.tau

            delta = (-self.delF[i] - self.delF[j]) / quadCoeff
            diff = self.alpha[i] - self.alpha[j]
            self.alpha[i] += delta
            self.alpha[j] += delta

            if diff > 0:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = diff

                if self.alpha[i] > self.modelInfo['C']:
                    self.alpha[i] = self.modelInfo['C']
                    self.alpha[j] = self.modelInfo['C'] - diff
            else:
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = -diff

                if self.alpha[j] > self.modelInfo['C']:
                    self.alpha[j] = self.modelInfo['C']
                    self.alpha[i] = self.modelInfo['C'] + diff

        else:
            quadCoeff = self.QD[i] + self.QD[j] - (2 * Q_i[j])
            if quadCoeff <= 0:
                quadCoeff = self.tau
            delta = (self.delF[i] - self.delF[j]) / quadCoeff
            Sum = self.alpha[i] + self.alpha[j]
            self.alpha[i] -= delta
            self.alpha[j] += delta

            if Sum > self.modelInfo['C']:
                if self.alpha[i] > self.modelInfo['C']:
                    self.alpha[i] = self.modelInfo['C']
                    self.alpha[j] = Sum - self.modelInfo['C']
                if self.alpha[j] > self.modelInfo['C']:
                    self.alpha[j] = self.modelInfo['C']
                    self.alpha[i] = Sum - self.modelInfo['C']
            else:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = Sum
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = Sum

        deltaAlphaI = self.alpha[i] - oldAlphaI
        deltaAlphaJ = self.alpha[j] - oldAlphaJ

        """ Update Gradient """
        oldF = np.copy(self.delF)
        for k in range(self.activeSize):
            self.delF[k] = oldF[k] + ((Q_i[k] * deltaAlphaI) + (Q_j[k] * deltaAlphaJ))

        """ Update Alpha Status and Gbar """
        if self.alphaStatus[i] == 1:
            oldStatusI = True
        else:
            oldStatusI = False
        if self.alphaStatus[j] == 1:
            oldStatusJ = True
        else:
            oldStatusJ = False

        self.updateAlphaStatus(i)
        self.updateAlphaStatus(j)

        if self.alphaStatus[i] == 1:
            newStatusI = True
        else:
            newStatusI = False
        if self.alphaStatus[j] == 1:
            newStatusJ = True
        else:
            newStatusJ = False

        if oldStatusI != newStatusI:
            Q_i = self.computeQ(i, self.trainN)
            oldGbar = np.copy(self.Gbar)
            if oldStatusJ:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] - (self.modelInfo['C'] * Q_i[k])
            else:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] + (self.modelInfo['C'] * Q_i[k])

        if oldStatusJ != newStatusJ:
            Q_j = self.computeQ(j, self.trainN)
            oldGbar = np.copy(self.Gbar)
            if oldStatusJ:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] - (self.modelInfo['C'] * Q_j[k])
            else:
                for k in range(self.trainN):
                    self.Gbar[k] = oldGbar[k] + (self.modelInfo['C'] * Q_j[k])

    def updateRho(self):
        ub1, ub2 = np.inf, np.inf
        lb1, lb2 = -np.inf, -np.inf
        nFree1, nFree2 = 0, 0
        sumFree1, sumFree2 = 0, 0
        for i in range(self.activeSize):
            if self.y[i] == 1:
                if self.alphaStatus[i] == 1:
                    lb1 = max(lb1, self.delF[i])
                elif self.alphaStatus[i] == 2:
                    ub1 = min(ub1, self.delF[i])
                else:
                    nFree1 += 1
                    sumFree1 += self.delF[i]
            else:
                if self.alphaStatus[i] == 1:
                    lb2 = max(lb2, self.delF[i])
                elif self.alphaStatus[i] == 2:
                    ub2 = min(ub2, self.delF[i])
                else:
                    nFree2 += 1
                    sumFree2 += self.delF[i]

        if nFree1 > 0:
            rho1 = sumFree1 / nFree1
        else:
            rho1 = (ub1 + lb1) / 2
        if nFree2 > 0:
            rho2 = sumFree2 / nFree2
        else:
            rho2 = (ub2 + lb2) / 2

        self.rho = (rho1 + rho2) / 2

    def beShrunk(self, i, Gmax1, Gmax2, Gmax3, Gmax4):
        if self.alphaStatus[i] == 1:
            if self.y[i] == 1:
                if -self.delF[i] > Gmax1:
                    return 1
            else:
                if -self.delF[i] > Gmax4:
                    return 1
        elif self.alphaStatus[i] == 2:
            if self.y[i] == 1:
                if self.delF[i] > Gmax2:
                    return 1
                else:
                    if self.delF[i] > Gmax3:
                        return 1
        else:
            return 0

    def shrink(self):
        Gmax1 = -np.inf
        Gmax2 = -np.inf
        Gmax3 = -np.inf
        Gmax4 = -np.inf

        """ Finding Maximal Violating Pair """
        for i in range(self.activeSize):
            if self.alphaStatus[i] != 1:
                if self.y[i] == 1:
                    if -self.delF[i] > Gmax1:
                        Gmax1 = -self.delF[i]
                else:
                    if -self.delF[i] > Gmax4:
                        Gmax4 = -self.delF[i]
            if self.alphaStatus[i] != 2:
                if self.y[i] == 1:
                    if self.delF[i] > Gmax2:
                        Gmax2 = self.delF[i]
                else:
                    if self.delF[i] > Gmax3:
                        Gmax3 = self.delF[i]

        if not self.unshrink and (max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= self.eps * 10):
            self.unshrink = True
            self.reconstructGradient()
            self.activeSize = self.trainN

        for i in range(self.activeSize):
            if self.beShrunk(i, Gmax1, Gmax2, Gmax3, Gmax4) == 1:
                self.activeSize -= 1
                while self.activeSize > i:
                    if self.beShrunk(self.activeSize, Gmax1, Gmax2, Gmax3, Gmax4) == 0:
                        self.swap(self.QD[i], self.QD[self.activeSize])
                        self.swap(self.y[i], self.y[self.activeSize])
                        self.swap(self.delF[i], self.delF[self.activeSize])
                        self.swap(self.alphaStatus[i], self.alphaStatus[self.activeSize])
                        self.swap(self.alpha[i], self.alpha[self.activeSize])
                        self.swap(self.qLinear[i], self.qLinear[self.activeSize])
                        self.swap(self.activeSet[i], self.activeSet[self.activeSize])
                        self.swap(self.Gbar[i], self.Gbar[self.activeSize])
                        break
                self.activeSize -= 1

    def reconstructGradient(self):
        if self.activeSize == self.trainN: return

        freeN = 0
        for j in range(self.activeSize, self.trainN):
            self.delF[j] = self.Gbar[j] + self.qLinear[j]

        for j in range(self.activeSize):
            if self.alphaStatus[j] == 3:
                freeN += 1

        if (self.trainN * freeN) > 2 * self.activeSize * (self.trainN - self.activeSize):
            for i in range(self.activeSize, self.trainN):
                Q_i = self.computeQ(i, self.activeSize)
                oldF = np.copy(self.delF)
                for j in range(self.activeSize):
                    if self.alphaStatus[j] == 3:
                        self.delF[j] = oldF[j] + (self.alpha[j] * Q_i[j])
        else:
            for i in range(self.activeSize):
                if self.alphaStatus[i] == 3:
                    Q_i = self.computeQ(i, self.trainN)
                    oldF = np.copy(self.delF)
                    for j in range(self.activeSize, self.trainN):
                        self.delF[j] = oldF[j] + (self.alpha[i] * Q_i[j])

        pass

    def train(self, trainData):
        self.qLinear = np.vstack((trainData[:, -1].reshape(-1, 1), trainData[:, -1].reshape(-1, 1)))
        self.y = np.hstack((np.ones(int(len(trainData))), -1 * np.ones(int(len(trainData)))))
        alphaP = np.zeros(len(trainData))
        alphaN = np.zeros(len(trainData))

        Sum = self.modelInfo['C'] * self.modelInfo['modelParam'] * (len(trainData) / 2)
        for i in range(len(trainData)):
            alphaP[i] = min(Sum, self.modelInfo['C'])
            alphaN[i] = alphaP[i]
            Sum = Sum - alphaP[i]

        self.alpha = np.append(alphaP, alphaN)
        wholeData = np.vstack((trainData, trainData))
        self.trainN = len(wholeData)
        self.trainX, self.trainZ = wholeData[:, :-1], wholeData[:, -1]

        """ Initialize Alpha Status """
        self.alphaStatus = np.zeros(self.trainN)
        for i in range(len(self.alpha)):
            self.updateAlphaStatus(i)

        """ Initialize Active Set """
        self.activeSet = np.arange(self.trainN)
        self.activeSize = len(self.activeSet)

        """ Initialize Gradient (delF) and Gbar """
        Psi = self.kernelObj.computeGram( self.trainX, self.trainX, self.modelInfo['C'],)
        self.QD = np.diagonal(Psi)
        self.delF = np.copy(self.qLinear).ravel()
        self.Gbar = np.zeros(self.trainN)
        for i in range(self.trainN):
            if self.alphaStatus[i] != 2:
                Q_i = self.computeQ(i, self.trainN)
                oldF = np.copy(self.delF)
                for j in range(self.trainN):
                    self.delF[j] = oldF[j] + (self.alpha[i] * Q_i[j])
                if self.alphaStatus[i] == 1:
                    self.Gbar[j] = self.modelInfo['C'] * Q_i[j]

        self.counter = min(self.trainN, 1000) + 1
        while self.iter < self.maxIter:
            # print('Iteration: {}'.format(self.iter))
            # print('Alpha: {}'.format(self.alpha.ravel()))
            decrement = self.counter - 1
            if decrement == 0:
                self.counter = min(self.trainN, 1000)
                if self.shrinking: self.shrink()

            pair = self.workingSet()
            if not pair:
                self.reconstructGradient()
                self.activeSize = self.trainN
                pair = self.workingSet()
                if not pair:
                    break
                else:
                    self.counter = 1

            i, j = pair[0], pair[1]
            self.updateAlpha(i, j)

            self.iter += 1

        if self.iter >= self.maxIter:
            if self.activeSize < self.trainN:
                self.reconstructGradient()
                self.activeSize = self.trainN
            # print("Max Iterations Reached")

        """ Update rho """
        self.updateRho()
        self.epsilon = -self.rho
        """ Calculate Objective Value """
        self.obj = np.dot(self.alpha.reshape(1, -1), (self.delF.reshape(-1, 1) + self.qLinear.reshape(-1, 1))) / 2
        self.alphaFinal = self.alpha[:int(self.trainN / 2)] - self.alpha[int(self.trainN / 2):self.trainN]

        self.alpha_ = np.zeros(len(self.activeSet))
        for k in range(len(self.activeSet)):
            self.alpha_[self.activeSet[k]] = self.alpha[k]

        svIdx = abs(self.alphaFinal) > 0
        self.alphaSV = self.alphaFinal[svIdx.ravel()]
        self.trainX = trainData[:, :-1]
        self.trainXSV = self.trainX[svIdx.ravel()]

    def predict(self, testX):
        predPsi = self.kernelObj.computeGram( testX, self.trainXSV, self.modelInfo['C'], self.modelInfo['penalize'])
        predicted = np.array(
            [np.sum(self.alphaSV.reshape(-1, 1) * predPsi[i].reshape(-1, 1)) for i in range(len(testX))]) - self.rho
        return predicted

"""
Parrella, Francesco. "Online support vector regression." Master's Thesis, 
Department of Information Science, University of Genoa, Italy 69 (2007).
"""
class OnlineSVR:
    def __init__(self, modelInfo):
        self.modelInfo = modelInfo
        self.kernelInfo = {
            'kernelType': modelInfo['kernelType'],
            'kernelParam': modelInfo['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        self.trainedSamplesX = {}
        self.trainedSamplesZ = {}
        self.trainX = []
        self.trainZ = []
        self.trainXc = []
        self.trainZc = []
        self.theta = []
        self.bias = 0

        self.idxSV = []
        self.idxES = []
        self.idxRS = []
        self.R = []
        self.Psi = []

        """ Counters """
        self.trainCount = 0
        self.sampleIdx = 0

    def predict(self, testX):
        trainedX = self.getTrainedSet()
        Q = self.kernelObj.computeGram( trainedX, testX, self.modelInfo['C'],)
        return np.dot(np.array(self.theta).reshape(1, -1), Q).reshape(-1, 1) + self.bias

    def getTrainedSet(self, x=True, z=False):
        X = np.array([row for row in self.trainedSamplesX.values()])
        Z = np.array([row for row in self.trainedSamplesZ.values()])
        if x and not z:
            return X
        if z and not x:
            return Z
        if x and z:
            return X, Z

    def computeMargin(self, testX, testZ):
        return self.predict(testX) - testZ.reshape(-1, 1)

    def computeBeta(self, idx):
        if not self.idxSV:
            beta = []
        else:
            trainedX = self.getTrainedSet()
            Qsc = [self.kernelObj.kernelFxn(trainedX[svIdx], trainedX[idx]) for svIdx in self.idxSV]
            Qsc.insert(0, 1)
            # Qsc = self.kernelObj.kernelFxn(trainedX[self.idxSV], trainedX[self.sampleIdx])
            # Qsc = np.insert(np.array([Qsc]), 0, 1)
            beta = np.dot((-1 * self.R), np.array(Qsc).reshape(-1, 1))
        return beta

    def computeGamma(self, beta):
        if not self.idxSV:
            gamma = np.ones((len(self.trainedSamplesX, )))
        else:
            trainedX = self.getTrainedSet()
            Qns = self.kernelObj.computeGram( trainedX, trainedX[self.idxSV], self.modelInfo['C'],)
            Qns = np.hstack((np.ones((len(trainedX), 1)), Qns))
            deltaQnc = np.array([self.kernelObj.kernelFxn(x, trainedX[self.sampleIdx]) for x in trainedX]).reshape(-1,
                                                                                                                   1)
            gamma = deltaQnc + np.dot(Qns, beta)
        return gamma

    def varLc(self, H, gamma):
        theta_c = self.theta[self.sampleIdx]
        H_c = H[self.sampleIdx]
        gamma_c = gamma[self.sampleIdx]
        eps = self.modelInfo['modelParam']
        C = self.modelInfo['C']

        """ Find Lc1 """
        if gamma_c <= 0:
            Lc1 = self.qDir * np.inf

        if H_c > eps and -C < theta_c <= 0:
            Lc1 = (-H_c + eps) / gamma_c
        elif H_c < -eps and 0 <= theta_c < C:
            Lc1 = (-H_c - eps) / gamma_c
        else:
            Lc1 = self.qDir * np.inf

        """ Find Lc2 """
        self.qDir = np.sign(Lc1)
        if not self.idxSV:
            Lc2 = self.qDir * np.inf
        elif self.qDir > 0:
            Lc2 = -theta_c + C
        else:
            Lc2 = -theta_c - C

        return Lc1, Lc2

    def varLs(self, H, beta):
        C = self.modelInfo['C']
        Ls = []
        if self.idxSV:
            for i in range(len(self.idxSV)):
                if beta[i + 1] == 0:
                    Ls.append(self.qDir * np.inf)
                elif self.qDir * beta[i + 1] > 0:
                    if H[self.idxSV[i]] > 0:
                        if self.theta[self.idxSV[i]] < -C:
                            Ls.append((-self.theta[self.idxSV[i]] - C) / beta[i + 1])
                        elif self.theta[self.idxSV[i]] <= 0:
                            Ls.append(-self.theta[self.idxSV[i]] / beta[i + 1])
                        else:
                            Ls.append(self.qDir * np.inf)
                    else:
                        if self.theta[self.idxSV[i]] < 0:
                            Ls.append(-self.theta[self.idxSV[i]] / beta[i + 1])
                        elif self.theta[self.idxSV[i]] <= C:
                            Ls.append((-self.theta[self.idxSV[i]] + C) / beta[i + 1])
                        else:
                            Ls.append(self.qDir * np.inf)

                else:
                    if H[self.idxSV[i]] > 0:
                        if self.theta[self.idxSV[i]] > 0:
                            Ls.append(-self.theta[self.idxSV[i]] / beta[i + 1])
                        elif self.theta[self.idxSV[i]] >= -C:
                            Ls.append((-self.theta[self.idxSV[i]] - C) / beta[i + 1])
                        else:
                            Ls.append(self.qDir * np.inf)
                    else:
                        if self.theta[self.idxSV[i]] > C:
                            Ls.append((-self.theta[self.idxSV[i]] + C) / beta[i + 1])
                        elif self.theta[self.idxSV[i]] >= 0:
                            Ls.append(-self.theta[self.idxSV[i]] / beta[i + 1])
                        else:
                            Ls.append(self.qDir * np.inf)
        else:
            Ls.append(self.qDir * np.inf)
        return np.array(Ls)

    def varLe(self, H, gamma):
        eps = self.modelInfo['modelParam']
        Le = []

        for i in self.idxES:
            if gamma[i] == 0:
                Le.append(self.qDir * np.inf)
            elif self.qDir * gamma[i] > 0:
                if self.theta[i] > 0:
                    if H[i] < -eps:
                        Le.append((-H[i] - eps) / gamma[i])
                    else:
                        Le.append(self.qDir * np.inf)
                else:
                    if H[i] < eps:
                        Le.append((-H[i] + eps) / gamma[i])
                    else:
                        Le.append(self.qDir * np.inf)

            else:
                if self.theta[i] > 0:
                    if H[i] > -eps:
                        Le.append((-H[i] - eps) / gamma[i])
                    else:
                        Le.append(self.qDir * np.inf)
                else:
                    if H[i] > eps:
                        Le.append((-H[i] + eps) / gamma[i])
                    else:
                        Le.append(self.qDir * np.inf)
        if not Le:
            Le.append(self.qDir * np.inf)

        return np.array(Le)

    def varLr(self, H, gamma):
        eps = self.modelInfo['modelParam']
        Lr = []

        for i in self.idxRS:
            if gamma[i] == 0:
                Lr.append(self.qDir * np.inf)
            elif self.qDir * gamma[i] > 0:
                if H[i] < -eps:
                    Lr.append((-H[i] - eps) / gamma[i])
                elif H[i] < eps:
                    Lr.append((-H[i] + eps) / gamma[i])
                else:
                    Lr.append(self.qDir * np.inf)
            else:
                if H[i] > eps:
                    Lr.append((-H[i] + eps) / gamma[i])
                elif H[i] > -eps:
                    Lr.append((-H[i] - eps) / gamma[i])
                else:
                    Lr.append(self.qDir * np.inf)

        if not Lr:
            Lr.append(self.qDir * np.inf)

        return np.array(Lr)

    def minVariation(self, H, beta, gamma):
        theta_c = self.theta[self.sampleIdx]
        H_c = H[self.sampleIdx]

        self.qDir = np.sign(-H_c)
        Lc1, Lc2 = self.varLc(H, gamma)
        Ls = self.varLs(H, beta)
        Le = self.varLe(H, gamma)
        Lr = self.varLr(H, gamma)

        if gamma[self.sampleIdx] < 0:
            for i in range(len(Ls)):
                if Ls[i] == 0:
                    Ls[i] = self.qDir * np.inf
            for i in range(len(Le)):
                if Le[i] == 0:
                    Le[i] = self.qDir * np.inf
            for i in range(len(Lr)):
                if Lr[i] == 0:
                    Lr[i] = self.qDir * np.inf

        Lc1, Ls, Le, Lr = Lc1.ravel(), Ls.ravel(), Le.ravel(), Lr.ravel()
        MinIndices = [None, None, np.abs(Ls).argmin(), np.abs(Le).argmin(), np.abs(Lr).argmin()]
        MinValues = [Lc1, Lc2, Ls[np.abs(Ls).argmin()], Le[np.abs(Le).argmin()], Lr[np.abs(Lr).argmin()]]
        flag = np.abs(np.array(MinValues)).argmin()
        MinVar, MinVarIdx = MinValues[flag], MinIndices[flag]
        return MinVar, MinVarIdx, flag

    def updateThetaBias(self, H, beta, gamma, minVar):
        if self.idxSV:
            """ Update Weights """
            self.theta[self.sampleIdx] += minVar
            """ Update Bias """
            deltaTheta = beta * minVar
            self.bias += deltaTheta[0]
            """ Update Weights in SV Set """
            deltaTheta = deltaTheta[1:]
            for i in range(len(self.idxSV)):
                self.theta[self.idxSV[i]] += deltaTheta[i]
            """ Update H """
            for i in range(len(self.trainedSamplesX)):
                H[i] += (gamma[i] * minVar)
        else:
            self.bias += minVar
            H += minVar

        return H

    def train(self, trainData):
        self.trainX = trainData[:, :-1]
        self.trainZ = trainData[:, -1]

        for i in range(len(trainData)):
            self.sampleIdx = i
            # print('-'*100)
            # print('Sample {}'.format(self.sampleIdx))
            # print('-' * 100)
            self.trainCount += self.learn(self.trainX[i], self.trainZ[i])

    def learn(self, sampleX, sampleZ):
        """ Initialization """
        self.trainXc = sampleX
        self.trainZc = sampleZ
        self.trainedSamplesX['{}'.format(self.sampleIdx)] = self.trainXc
        self.trainedSamplesZ['{}'.format(self.sampleIdx)] = self.trainZc
        self.theta.append(0)
        self.newSample = False
        trainedX, trainedZ = self.getTrainedSet(x=True, z=True)
        H = self.computeMargin(trainedX, trainedZ)

        if abs(H[self.sampleIdx]) <= self.modelInfo['modelParam']:
            # print('Sample {} Within Epsilon'.format(self.sampleIdx))
            self.idxRS.append(self.sampleIdx)
            return 1

        while not self.newSample:
            # print('Number of SV: {}'.format(len(self.idxSV)))
            # print('R Matrix: {}'.format(self.R))
            # Check Iteration Number
            if self.trainCount > len(trainedX):
                break

            """ Find Beta and Gamma """
            beta = self.computeBeta(self.sampleIdx)
            gamma = self.computeGamma(beta)
            # print('Beta: {}'.format(beta))
            # print('Gamma: {}'.format(gamma))

            """ Find Min Variation """
            minVar, minVarIdx, flag = self.minVariation(H, beta, gamma)
            if isinstance(minVar, np.ndarray):
                minVar = minVar[0]
            H = self.updateThetaBias(H, beta, gamma, minVar)

            if flag == 0:
                self.AddSupportSet(H)
                self.newSample = True
            elif flag == 1:
                self.AddErrorSet()
                self.newSample = True
            elif flag == 2:
                self.MoveSVtoESRS(minVarIdx)
            elif flag == 3:
                self.MoveEStoSV(H, minVarIdx)
            elif flag == 4:
                self.MoveRStoSV(H, minVarIdx)

        return 1

    def AddSupportSet(self, H):
        # print('Adding Sample {} to Support Set'.format(self.sampleIdx))
        H[self.sampleIdx] = np.sign(H[self.sampleIdx]) * self.modelInfo['modelParam']
        self.idxSV.append(self.sampleIdx)
        # self.R = self.AddR(self.sampleIdx, beta, gamma, oldSet='SV')
        self.updateR()

    def AddErrorSet(self):
        # print('Adding Sample {} to Error Set'.format(self.sampleIdx))
        self.theta[self.sampleIdx] = np.sign(self.theta[self.sampleIdx]) * self.modelInfo['C']
        self.idxES.append(self.sampleIdx)

    def MoveSVtoESRS(self, minVarIdx):

        idx = self.idxSV[minVarIdx]
        if abs(self.theta[idx]) < abs(self.modelInfo['C'] - abs(self.theta[idx])):
            self.theta[idx] = 0
        else:
            self.theta[idx] = np.sign(self.theta[idx]) * self.modelInfo['C']
        if self.theta[idx] == 0:
            """ Move Sample from SV to RS """
            # print('Moving Sample {} from Support Set to Remaining Set'.format(idx))
            self.idxRS.append(idx)
            self.idxSV = list(np.delete(np.array(self.idxSV), minVarIdx))
            # self.RemoveR(minVarIdx)
            self.updateR()
        else:
            """ Move Sample from SV to ES """
            # print('Moving Sample {} from Support Set to Error Set'.format(idx))
            self.idxES.append(idx)
            self.idxSV = list(np.delete(np.array(self.idxSV), minVarIdx))
            # self.RemoveR(minVarIdx)
            self.updateR()

    def MoveEStoSV(self, H, minVarIdx):
        idx = self.idxES[minVarIdx]
        # print('Moving Sample {} from Error Set to Support Set'.format(idx))
        H[idx] = np.sign(H[idx]) * self.modelInfo['modelParam']
        self.idxSV.append(idx)
        self.idxES = list(np.delete(np.array(self.idxES), minVarIdx))
        # self.R = self.AddR(idx, beta, gamma, oldSet='ES')
        self.updateR()

    def MoveRStoSV(self, H, minVarIdx):
        idx = self.idxRS[minVarIdx]
        # print('Moving Sample {} from Remaining Set to Support Set'.format(idx))
        H[idx] = np.sign(H[idx]) * self.modelInfo['modelParam']
        self.idxSV.append(idx)
        self.idxRS = list(np.delete(np.array(self.idxRS), minVarIdx))
        # self.AddR(idx, beta, gamma, oldSet='RS')
        self.updateR()

    def updateR(self):
        if self.idxSV:
            trainedX = self.getTrainedSet()
            row = np.insert(np.ones((len(self.idxSV, ))), 0, 0).reshape(1, -1)
            col = np.ones((len(self.idxSV), 1))
            newR = np.array(
                [[self.kernelObj.kernelFxn(trainedX[i], trainedX[j]) for i in self.idxSV] for j in self.idxSV])
            temp1 = np.hstack((col, np.copy(newR)))
            newR = np.vstack((row, temp1))
            self.R = np.linalg.inv(newR)
        else:
            self.R = []

    def AddR(self, idx, beta, gamma, oldSet):
        if not len(self.R):
            newR = np.ones((2, 2))
            newR[0, 0] = -self.kernelObj.kernelFxn(idx, idx)
            newR[1, 1] = 0
        else:
            if oldSet == 'ES' or oldSet == 'RS':
                last = self.idxSV[-1]
                self.idxSV = self.idxSV[:-1]
                newBeta = self.computeBeta(idx)
                if not self.idxSV:
                    gamma_i = 1
                else:
                    trainedX = self.getTrainedSet()
                    Qii = self.kernelObj.kernelFxn(trainedX[idx], trainedX[idx])
                    Qsi = np.array([(self.kernelObj.kernelFxn(trainedX[sv], trainedX[idx])) for sv in self.idxSV])
                    Qsi = np.insert(Qsi, 0, 1)
                    gamma_i = Qii + np.dot(Qsi, newBeta)

                gamma[idx] = gamma_i
                newGamma = gamma
                self.idxSV.append(last)
            else:
                newBeta = beta
                newGamma = gamma

            row = np.zeros((1, self.R.shape[1]))
            col = np.zeros((self.R.shape[0] + 1, 1))
            newR = np.hstack((np.vstack((self.R, row)), col))
            if newGamma[idx] != 0:
                oldR = np.copy(self.R)
                newBeta = np.append(beta, 1)
                newR = oldR + ((1 / newGamma[idx]) * np.dot(newBeta, newBeta.T))

        return newR

    def RemoveR(self, idx):
        set = np.arange(idx + 1)
        R_Ii = np.array([self.R[i, idx] for i in set])
        R_iI = np.array([self.R[idx, i] for i in set])
        if self.R[idx, idx] != 0:
            oldR = np.copy(self.R)
            self.R = np.array([[oldR[i, j] for i in set] for j in set]) - np.divide(np.dot(R_Ii, R_iI), oldR[idx, idx])

        if self.R.shape[0] == 1:
            self.R = []

""" Relevance Vector Machine """
class RVM:

    def __init__(self, model_info):
        self.modelInfo = model_info
        self.kernelInfo = {
            'kernelType': model_info['kernelType'],
            'kernelParam': model_info['kernelParam']
        }
        self.kernelObj = utilities.Kernels(self.kernelInfo)

        # Store Necessary Data Used Across All Methods #
        self.history = {}
        self.alpha = 1e-6
        self.beta = 1e-6
        self.threshold = 1e-9
        self.tol = 1e-3
        self.max_iter = 3000
        self.phi = 0
        self.epsilon = 1e-8
        self.relevance_vectors = None
        self.relevance = None
        # Statistics
        self.covariance = 0  # sigma
        self.mean = 0  # Mu
        self.modelNumber = 0

    def prune(self):
        # Condition
        condition = self.alpha < self.threshold
        # If all of conditon is evaluted as false
        # Set first element of condition to true
        if not np.any(condition):
            condition[0] = True
        
        self.relevance_vectors = self.relevance_vectors[condition[1:]]
        self.alpha = self.alpha[condition]
        self.alpha_old = self.alpha_old[condition]
        self.gamma = self.gamma[condition]
        self.phi = self.phi[:, condition]
        self.covariance = self.covariance[np.ix_(condition, condition)]
        self.mean = self.mean[condition]

    def train(self, trainData):
        trainX, trainZ = trainData[:, :-1], trainData[:, -1]
        if self.kernelObj.kernelInfo['kernelParam'] == None:
            self.kernelObj.kernelInfo['kernelParam'] == 1 / trainX.shape[1]
        self.phi = self.kernelObj.computeGram(trainX, trainX, penalize=self.modelInfo['penalize'])
        n_samples = self.phi.shape[0]
        # if self.bias
        self.phi = np.hstack((np.ones((n_samples, 1)), self.phi))
        M = self.phi.shape[1]
        self.alpha0 =  1 / M**2
        self.relevance = np.arange(n_samples)
        self.relevance_vectors = trainX
        sigma_sqrd = (max(self.epsilon, np.std(trainZ) * 0.1) ** 2)
        self.beta = 1 / sigma_sqrd
        self.alpha = self.alpha0 * np.ones(M)
        self.alpha_old = np.copy(self.alpha)

        for i in range(self.max_iter):
            A = np.diag(self.alpha)
            hess = self.beta * (self.phi.T @ self.phi) + A
            # Calculate Sigma (Covariance) & Mu (Mean) Try Cholesky
            self.covariance = np.linalg.inv(hess)
            self.mean = self.beta * (self.covariance @ self.phi.T @ trainZ)
            covariance_diag = np.diag(self.covariance)
             # Well-determinedness parameters (gamma)
            self.gamma = 1 - self.alpha * covariance_diag
            # Alpha Re-estimation
            self.alpha =  np.maximum(self.gamma, self.epsilon) / (self.mean ** 2) + self.epsilon
            # Prediction error
            pred_error = np.sum((trainZ - self.phi @ self.mean) ** 2)
            self.beta = max((n_samples - np.sum(self.gamma)), self.epsilon) / pred_error + self.epsilon
            # Compute Marginal Likelihood, possible with Cholesky Fact
        
        # Add verbose/info

            self.prune()
            delta = np.amax(np.absolute(np.log(self.alpha + self.epsilon) - np.log(self.alpha_old + self.epsilon)))
            if delta < self.tol and i > 1:
                break
            self.alpha_old = np.copy(self.alpha)
                    
    def predict(self, testX):
        n_samples = len(testX)
        kernel = self.kernelObj.computeGram(testX, self.relevance_vectors, penalize=self.modelInfo['penalize'])
        kernel = np.hstack((np.ones((n_samples, 1)), kernel))
        z_mean = kernel @ self.mean
        error = (1 / self.beta) + kernel @ self.covariance @ kernel.T
        z_std = np.sqrt(np.diag(error))
        return z_mean, z_std





