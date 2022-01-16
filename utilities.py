import numpy as np
import random as rd
import pandas as pd
import svm_models
from pathlib import Path
from optuna import create_study, samplers, trial, integration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def train_test_split(data, split=0.8):
    train = list()
    train_size = split * len(data)
    test = list(data)
    while len(train) < train_size:
        index = rd.randrange(len(test))
        train.append(test.pop(index))

    trainData = np.array(train)
    testData = np.array(test)
    return trainData, testData


def split_fxn(data, i):
    train_data = data[0:-i, :]
    test_data = data[-i:, :]
    return train_data, test_data


def rescale(data_set, col=3):
    if col == 3:
        data_set[:, 0] = (data_set[:, 0] * (np.max(data_set[:, 0]) - np.min(data_set[:, 0]))) + np.min(data_set[:, 0])
        data_set[:, 1] = (data_set[:, 1] * (np.max(data_set[:, 1]) - np.min(data_set[:, 1]))) + np.min(data_set[:, 1])
        data_set[:, 2] = (data_set[:, 2] * (np.max(data_set[:, 2]) - np.min(data_set[:, 2]))) + np.min(data_set[:, 2])
    elif col == 2:
        data_set[:, 0] = (data_set[:, 0] * (np.max(data_set[:, 0]) - np.min(data_set[:, 0]))) + np.min(data_set[:, 0])
        data_set[:, 1] = (data_set[:, 1] * (np.max(data_set[:, 1]) - np.min(data_set[:, 1]))) + np.min(data_set[:, 1])

    return data_set


class ConfigData:
    def __init__(self):

        # Accepted Filetypes (as string) - xlsx, csv, dat
        # NOTE: Write replace '\' in filedir with '/'
        self.data = []

        # Stored if train_test_split() is called
        self.trainData = []
        self.testData = []

    def readData(self, readDir, extension):

        if extension == 'csv':
            data_df = pd.read_csv(readDir / 'whole_data.csv')

        data_df = data_df.dropna()
        self.data = data_df.to_numpy()

    def saveData(self, saveDir):
        np.savetxt(saveDir / 'train_data.dat', self.trainData)
        np.savetxt(saveDir / 'test_data.dat', self.testData)
        pass

    def train_test_split(self, split=0.2):

        train = list()
        train_size = split * len(self.data)
        test = list(self.data)
        while len(train) < train_size:
            index = rd.randrange(len(test))
            train.append(test.pop(index))

        self.trainData = np.array(train)
        self.testData = np.array(test)


class ErrorMetric:

    @staticmethod
    def mean_squared_error(y_true, y_predict):
        error = np.average((y_true - y_predict.reshape(-1, 1)) ** 2, axis=0)
        mse = np.average(error)
        return mse

    @staticmethod
    def root_mean_squared_error(y_true, y_predict):
        error = (y_true - y_predict) ** 2
        # error = np.average((y_true - y_predict.reshape(-1, 1)) ** 2, axis=0)
        mse = np.average(error)
        return np.sqrt(mse)

    @staticmethod
    def accuracy(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        yTrue = np.copy(y_true)
        yPred = np.copy(y_pred)
        yTrue[yTrue == -1] = 0
        yPred[yPred == -1] = 0
        score = accuracy_score(yTrue, yPred)
        return score

    @staticmethod
    def fscore(y_true, y_pred):
        from sklearn.metrics import f1_score
        yTrue = np.copy(y_true)
        yPred = np.copy(y_pred)
        yTrue[yTrue == -1] = 0
        yPred[yPred == -1] = 0
        return f1_score(yTrue, yPred, average='binary')

    @staticmethod
    def prec_recall(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        yTrue = np.copy(y_true)
        yPred = np.copy(y_pred)
        yTrue[yTrue == -1] = 0
        yPred[yPred == -1] = 0
        precision = precision_score(yTrue, yPred, average='binary')
        recall = recall_score(yTrue, yPred, average='binary')
        confusion = confusion_matrix(yTrue, yPred)
        return precision, recall, confusion


class TuneParameters:
    def __init__(self, search, wholeData, train_data, test_data, type, cv_fold):
        self.search = search
        self.type = type
        self.modelGrid = {
            'CSVC': {
                'callSVR': svm_models.CSVC,
                'modelParam': None,
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
            'EpsilonSVR': {
                'callSVR': svm_models.EpsilonSVR,
                'modelParam': search['model_param'],
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
            'LSSVR': {
                'callSVR': svm_models.LSSVR,
                'modelParam': None,
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
            'RVR': {
                'callSVR': EMRVR(),
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],

            },
            'EpsilonSMO': {
                'callSVR': svm_models.EpsilonSMO,
                'modelParam': search['model_param'],
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
            'NuSVR': {
                'callSVR': svm_models.NuSVR,
                'modelParam': search['model_param'],
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
            'OnlineSVR': {
                'callSVR': svm_models.OnlineSVR,
                'modelParam': search['model_param'],
                'kernelType': search['kernel_type'],
                'kernelParam': search['kernel_param'],
                'C': search['C']
            },
        }
        self.svrModel = self.modelGrid[search['model_type']]
        self.trialHistory = []
        self.cv_fold = cv_fold
        self.iter = 0
        self.wholeData = wholeData
        self.trainData = train_data
        self.testData = test_data

    def sequence(self, d_seq, new_sample):
        split = np.array_split(d_seq, 2)
        if len(new_sample) == 0:
            self.trainData = np.vstack((self.wholeData, split[0]))
            self.testData = split[1]
        else:
            split[0] = np.vstack((split[0], new_sample[0, :].reshape(1, -1)))
            split[1] = np.vstack((split[1], new_sample[1, :].reshape(1, -1)))
            self.trainData = np.vstack((self.wholeData, split[0]))
            self.testData = split[1]

    def optunaObjective(self, trial):
        if self.search['model_type'] != 'RVR':
            C = trial.suggest_loguniform('C', self.svrModel['C'][0], self.svrModel['C'][1])

            # kernelType = trial.suggest_categorical('Kernel', self.svrModel['kernelType'])
            kernelType = 'rbf'
            kernelParam = trial.suggest_uniform('KernelParam', self.svrModel['kernelParam'][0],
                                                self.svrModel['kernelParam'][1])

            if self.search['model_type'] == 'LSSVR' or self.search['model_type'] == 'CSVC':
                model_info = {'C': C, 'penalize': 'L1', 'kernelType': kernelType, 'kernelParam': kernelParam}
            else:
                modelParam = trial.suggest_uniform('ModelParam', self.svrModel['modelParam'][0],
                                                   self.svrModel['modelParam'][1])
                model_info = {'C': C, 'modelParam': modelParam, 'penalize': 'L1', 'kernelType': kernelType,
                              'kernelParam': kernelParam}

            trainedModel = self.svrModel['callSVR'](model_info)
        elif self.search['model_type'] == 'RVR':
            kernelType = 'rbf'
            kernelParam = trial.suggest_uniform('KernelParam', self.svrModel['kernelParam'][0], self.svrModel['kernelParam'][1])
            trainedModel = self.svrModel['callSVR']
            trainedModel.kernel = kernelType
            trainedModel.gamma = kernelParam

        self.iter += 1
        trainedModel.modelNumber = self.iter

        # No CV
        # trainedModel.train(self.trainData)
        # y_pred = trainedModel.predict(self.testData[:, :-1])
        # wholeData = np.vstack((self.trainData, self.testData))

        # K-Fold CV
        # cvObj = CrossValidate(trainedModel, self.trainData, self.cv_fold)
        # cvError, trainedModel = cvObj.train(kfold=self.cv_fold)
        # y_pred = trainedModel.predict(self.testData[:, :-1])

        # LOO-CV if split is False
        if self.search['model_type'] != 'RVR':
            if self.cv_fold:
                if len(self.trainData) == 0:
                    cvObj = CrossValidate(trainedModel, self.wholeData)
                    cvError, trainedModel = cvObj.train(kfold=self.cv_fold)
                    y_pred = trainedModel.predict(self.wholeData[:, :-1])
                else:
                    cvObj = CrossValidate(trainedModel, self.trainData)
                    cvError, trainedModel = cvObj.train(kfold=self.cv_fold)
                    y_pred = trainedModel.predict(self.testData[:, :-1])
            else:
                trainedModel.train(self.trainData)
                y_pred = trainedModel.predict(self.testData[:, :-1])

            if self.type == 'classification' and len(self.trainData) != 0:
                fscore = ErrorMetric.fscore(self.testData[:, -1], y_pred)
                accuracy = ErrorMetric.accuracy(self.testData[:, -1], y_pred)
                score, recall, confusion_matrix = ErrorMetric.prec_recall(self.testData[:, -1], y_pred)
                trainedModel.evaluationMetric['F1_Score'] = fscore
                trainedModel.evaluationMetric['Precision'] = score
                trainedModel.evaluationMetric['Recall'] = recall
                trainedModel.evaluationMetric['ConfusionMatrix'] = confusion_matrix
                trainedModel.evaluationMetric['Accuracy'] = accuracy
            elif self.type == 'classification' and len(self.trainData) == 0:
                fscore = ErrorMetric.fscore(self.wholeData[:, -1], y_pred)
                accuracy = ErrorMetric.accuracy(self.wholeData[:, -1], y_pred)
                score, recall, confusion_matrix = ErrorMetric.prec_recall(self.wholeData[:, -1], y_pred)
                trainedModel.evaluationMetric['F1_Score'] = fscore
                trainedModel.evaluationMetric['Precision'] = score
                trainedModel.evaluationMetric['Recall'] = recall
                trainedModel.evaluationMetric['ConfusionMatrix'] = confusion_matrix
                trainedModel.evaluationMetric['Accuracy'] = accuracy

        elif self.search['model_type'] == 'RVR':
            if self.cv_fold:
                if len(self.trainData) == 0:
                    cvObj = CrossValidate(trainedModel, self.wholeData)
                    cvError, trainedModel = cvObj.RVfit(kfold=self.cv_fold)
                    y_pred = trainedModel.predict(self.wholeData[:, :-1])
                else:
                    cvObj = CrossValidate(trainedModel, self.trainData)
                    cvError, trainedModel = cvObj.RVfit(kfold=self.cv_fold)
                    y_pred = trainedModel.predict(self.testData[:, :-1])
            else:
                trainedModel.train(self.trainData)
                y_pred = trainedModel.predict(self.testData[:, :-1])

        self.trialHistory.append(trainedModel)
        if self.type == 'regression' and len(self.trainData) != 0:
            score = ErrorMetric.root_mean_squared_error(self.testData[:, -1], y_pred)
        elif self.type == 'regression' and len(self.trainData) == 0:
            score = ErrorMetric.root_mean_squared_error(self.wholeData[:, -1], y_pred)

        return score

    def optuna(self, ntrials=100):
        if self.type == 'classification':
            study = create_study(direction='maximize', sampler=samplers.TPESampler())
        else:
            study = create_study(direction='minimize', sampler=samplers.TPESampler())
        study.optimize(self.optunaObjective, n_trials=ntrials)
        # optuna.visualization.plot_contour(study, params=['C', 'KernelParam'])
        print('Best Score: ' + str(study.best_value))
        print('Best parameter: ' + str(study.best_params))
        return study.best_params, self.trialHistory[study.best_trial.number]


class CrossValidate:

    def __init__(self, svModel, trainData):
        self.svModel = svModel  # Initialized SVM model with params
        self.trainData = trainData
        # np.random.shuffle(self.trainData)

    def train(self, kfold=5):
        folds = np.array_split(self.trainData, kfold)
        testError = []
        for i in range(kfold):
            validArray = folds[i]
            trainArray = np.concatenate([folds[j] for j in range(kfold) if (j != i)])
            self.svModel.train(trainArray)
            z_pred = self.svModel.predict(validArray[:, :-1])
            testError.append(ErrorMetric.root_mean_squared_error(validArray[:, -1], z_pred))

        test_error = (1 / kfold) * np.sum(np.array(testError))
        return test_error, self.svModel

    def RVfit(self, kfold=5):
        folds = np.array_split(self.trainData, kfold)
        testError = []
        for i in range(kfold):
            validArray = folds[i]
            trainArray = np.concatenate([folds[j] for j in range(kfold) if (j != i)])
            self.svModel.fit(trainArray[:, :-1], trainArray[:, -1])
            z_pred, z_std = self.svModel.predict(validArray[:, :-1], return_std=True)
            testError.append(ErrorMetric.root_mean_squared_error(validArray[:, -1], z_pred))

        test_error = (1 / kfold) * np.sum(np.array(testError))
        return test_error, self.svModel


class Kernels:

    def __init__(self, kernel_info):

        self.kernelInfo = kernel_info

        self.kernelDict = {
            'linear': self.linear_kernel,
            'rbf': self.rbf_kernel,
            'poly': self.poly_kernel,
            'anis_rbf': self.rbf_kernel_anis
        }

        self.kernelFxn = self.kernelDict[self.kernelInfo['kernelType']]

    def linear_kernel(self, x, y):
        return np.dot(x, y.T)

    def rbf_kernel(self, x, y):
        return np.exp(-self.kernelInfo['kernelParam'] * np.linalg.norm(x - y) ** 2)

    def rbf_kernel_anis(self, x, y):
        kernelParam = np.array([self.kernelInfo['kernelParam'], self.kernelInfo['kernelParam']])
        k = np.zeros((len(kernelParam), 1))
        for i in range(len(kernelParam)):
            k[i] = np.exp(-kernelParam[i] * np.linalg.norm(x - y) ** 2)
        return np.prod(k)

    def poly_kernel(self, x, y):
        return (1 + np.dot(x, y.T)) ** self.kernelInfo['kernelParam']

    def computeGram(self, x, y, C = None, penalize='L1'):
        K = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                K[i, j] = self.kernelFxn(x[i].reshape(1, -1), y[j].reshape(1, -1))

        if penalize == 'L1':
            return K
        elif penalize == 'L2' and C:
            return K + np.eye(len(K), len(K)) * (1 / C)

    def computeQ(self, x1, x2, y, C=None, penalize='L1'):
        K = np.zeros((len(x1), len(x2)))
        y = y.ravel()
        for i in range(len(x1)):
            for j in range(len(x2)):
                K[i, j] = y[i] * y[j] * self.kernelFxn(x1[i].reshape(1, -1), x2[j].reshape(1, -1))

        if penalize == 'L1':
            return K
        elif penalize == 'L2' and C:
            return K + np.eye(len(K), len(K)) * (1 / C)


class Plot:
    def __init__(self, wholeData, type):
        self.wholeData = wholeData
        self.type = type

    def plot_decision_boundary(self, trained_model):
        stable = self.wholeData[self.wholeData[:, -1] == 1]
        unstable = self.wholeData[self.wholeData[:, -1] == -1]
        xx, yy = np.meshgrid(np.arange(0.7, 0.9005, 0.01), np.arange(0.4, 2.0, 0.01))

        fig, ax = plt.subplots()
        Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        if self.type == 'classification':
            contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('gnuplot'), alpha=0.8)
        else:
            contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig.colorbar(contour)
        cbar.ax.set_ylabel('Damping Coefficient')
        self.cs = ax.contour(xx, yy, Z, [0.0], colors='k', linewidths=2)
        self.dat0 = self.cs.allsegs[0][0]
        plt.plot(self.dat0[:, 0], self.dat0[:, 1])
        if self.type == 'classification':
            ax.scatter(stable[:, 0], stable[:, 1], c='y', s=20, edgecolors='k', label='Stable')
            ax.scatter(unstable[:, 0], unstable[:, 1], c='m', s=20, edgecolors='k', label='Unstable')
        ax.set_ylabel('Flutter Speed Index')
        ax.set_xlabel('Mach Number')
        ax.set_xlim(0.7, 0.9005)
        ax.set_ylim(0.4, 2.0)
        ax.set_title('Flutter Boundary')
        ax.legend(loc=2)
        plt.show()

        if self.type == 'regression':
            fig2 = plt.figure(2)
            ax2 = fig2.gca(projection='3d')
            Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # Pepsilon = Z + trained_model.modelInfo['modelParam']
            # Nepsilon = Z - trained_model.modelInfo['modelParam']
            # PEsurface = ax2.plot_wireframe(xx, yy, Pepsilon, color='black')
            # NEsurface = ax2.plot_wireframe(xx, yy, Nepsilon, color='black')
            surface = ax2.plot_surface(xx, yy, Z, cmap=plt.get_cmap('nipy_spectral'))
            cbar = fig2.colorbar(surface)
            ax2.scatter(self.wholeData[:, 0], self.wholeData[:, 1], self.wholeData[:, 2], c='k', s=20)
            ax2.set_ylabel('Flutter Speed Index')
            ax2.set_xlabel('Mach Number')
            ax2.set_zlabel('Damping Coefficient')
            ax2.set_xlim(0.7, 0.9005)
            ax2.set_ylim(0.4, 2.0)
            ax2.set_title('Damping Surface')
            ax2.legend()
            plt.show()

    def plot_test(self, trained_model):
        if self.type == 'regression':
            MachMin, MachMax = min(self.wholeData[:, 0]), max(self.wholeData[:, 0])
            AoAMin, AoAMax = min(self.wholeData[:, 1]), max(self.wholeData[:, 1])
            xx, yy = np.meshgrid(np.arange(MachMin, MachMax, 0.005), np.arange(AoAMin, AoAMax, 0.005))

            fig2 = plt.figure(2)
            ax2 = fig2.gca(projection='3d')
            dummy = np.c_[xx.ravel(), yy.ravel()]
            Z = trained_model.predict(dummy)

            # Rescale
            rescaled = rescale(np.copy(self.wholeData))
            MachMin, MachMax = min(rescaled[:, 0]), max(rescaled[:, 0])
            AoAMin, AoAMax = min(rescaled[:, 1]), max(rescaled[:, 1])
            xre, yre = np.meshgrid(np.arange(MachMin, MachMax, 0.005), np.arange(AoAMin, AoAMax, 0.005))
            Z_rescaled = (Z * (np.max(Z) - np.min(Z))) + np.min(Z)
            Z = Z.reshape(xre.shape)

            surface = ax2.plot_surface(xre, yre, Z_rescaled, cmap=plt.get_cmap('nipy_spectral'))
            cbar = fig2.colorbar(surface)
            ax2.scatter(self.wholeData[:, 0], self.wholeData[:, 1], self.wholeData[:, 2], c='k', s=20)
            ax2.set_xlabel('Mach Number')
            ax2.set_ylabel('Angle of Attack')
            ax2.set_zlabel('Pressure Coefficient')
            ax2.set_xlim(MachMin, MachMax)
            ax2.set_ylim(AoAMin, AoAMax)
            ax2.legend()
            plt.show()
        else:
            stable = self.wholeData[self.wholeData[:, -1] == 1]
            unstable = self.wholeData[self.wholeData[:, -1] == -1]
            xx, yy = np.meshgrid(np.arange(min(self.wholeData[:, 0]), max(self.wholeData[:, 0]), 0.005),
                                 np.arange(min(self.wholeData[:, 1]), max(self.wholeData[:, 1]), 0.005))

            fig, ax = plt.subplots()
            Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if self.type == 'classification':
                contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('gnuplot'), alpha=0.8)
            else:
                contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
            cbar = fig.colorbar(contour)
            # cbar.ax.set_ylabel('Damping Coefficient')
            self.cs = ax.contour(xx, yy, Z, [0.0], colors='k', linewidths=2)
            self.dat0 = self.cs.allsegs[0][0]
            plt.plot(self.dat0[:, 0], self.dat0[:, 1])
            if self.type == 'classification':
                ax.scatter(stable[:, 0], stable[:, 1], c='y', s=20, edgecolors='k', label='Class 1')
                ax.scatter(unstable[:, 0], unstable[:, 1], c='m', s=20, edgecolors='k', label='Class 2')
            ax.set_ylabel('X2')
            ax.set_xlabel('X1')
            ax.set_xlim(min(self.wholeData[:, 0]), max(self.wholeData[:, 0]))
            ax.set_ylim(min(self.wholeData[:, 1]), max(self.wholeData[:, 1]))
            # ax.set_title('Flutter Boundary')
            ax.legend(loc=2)
            plt.show()

    def plot_rvm(self, trained_model):
        stable = self.wholeData[self.wholeData[:, -1] == 1]
        unstable = self.wholeData[self.wholeData[:, -1] == -1]
        xx, yy = np.meshgrid(np.arange(0.7, 0.9005, 0.01), np.arange(0.4, 2.0, 0.01))

        fig, ax = plt.subplots()
        # Z, self.std_dev = trained_model.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
        Z, self.std_dev = trained_model.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
        Z = Z.reshape(xx.shape)
        if self.type == 'classification':
            contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('gnuplot'), alpha=0.8)
        else:
            contour = ax.contourf(xx, yy, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig.colorbar(contour)
        cbar.ax.set_ylabel('Damping Coefficient')
        self.cs = ax.contour(xx, yy, Z, [0.0], colors='k', linewidths=2)
        self.dat0 = self.cs.allsegs[0][0]
        plt.plot(self.dat0[:, 0], self.dat0[:, 1])
        if self.type == 'classification':
            ax.scatter(stable[:, 0], stable[:, 1], c='y', s=20, edgecolors='k', label='Stable')
            ax.scatter(unstable[:, 0], unstable[:, 1], c='m', s=20, edgecolors='k', label='Unstable')
        ax.set_ylabel('Flutter Speed Index')
        ax.set_xlabel('Mach Number')
        ax.set_xlim(0.7, 0.9005)
        ax.set_ylim(0.4, 2.0)
        ax.set_title('Flutter Boundary')
        ax.legend(loc=2)
        plt.show()

        fig2 = plt.figure(2)
        ax2 = fig2.gca(projection='3d')
        Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Pepsilon = Z + trained_model.modelInfo['modelParam']
        # Nepsilon = Z - trained_model.modelInfo['modelParam']
        # PEsurface = ax2.plot_wireframe(xx, yy, Pepsilon, color='black')
        # NEsurface = ax2.plot_wireframe(xx, yy, Nepsilon, color='black')
        surface = ax2.plot_surface(xx, yy, Z, cmap=plt.get_cmap('nipy_spectral'))
        cbar = fig2.colorbar(surface)
        ax2.scatter(self.wholeData[:, 0], self.wholeData[:, 1], self.wholeData[:, 2], c='k', s=20)
        ax2.set_ylabel('Flutter Speed Index')
        ax2.set_xlabel('Mach Number')
        ax2.set_zlabel('Damping Coefficient')
        ax2.set_xlim(0.7, 0.9005)
        ax2.set_ylim(0.4, 2.0)
        ax2.set_title('Damping Surface')
        ax2.legend()
        plt.show()

""" Create Sobol sequence """
# import chaospy
#
# distribution = chaospy.J(chaospy.Uniform(0.7, 0.9), chaospy.Uniform(0.4, 2))
# samples = distribution.sample(25, rule="sobol")
#
# fig = plt.figure()
#
# plt.scatter(samples[0, :], samples[1, :])

""" CREATE LHS SAMPLES """
# from smt.sampling_methods import LHS
# xlimits = np.array([[0.7, 0.9], [0.4, 2.0]])
# sampling = LHS(xlimits=xlimits)
#
# num = 25
# x = sampling(num)
# np.savetxt('LHS_25.dat', x)
# print(x.shape)
#
# plt.plot(x[:, 0], x[:, 1], "o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
