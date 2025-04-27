import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # early-stop warnings
import pandas as pd
import time
import pickle

class BEMT_NN:

    def __init__(self, initialize, data_extension, num_of_inputs, hidden_layer_size, activator="tanh", solver="adam", mode=0, degree=8):
        self.initialize = initialize
        self.data_ext = data_extension
        self.num_of_inputs = num_of_inputs
        self.RPM_REFERENCE = 10000
        self.T_REFERENCE = 22
        self.P_REFERENCE = 133
        self.V_REFERENCE = 50
        self.mode = mode
        self.activator = activator
        self.solver = solver
        self.hls = hidden_layer_size
        self.poly = PolynomialFeatures(degree=degree)
        if self.initialize:
            self.trainX = []
            self.testX = []
            self.trainY = []
            self.testY = []
            self.trainXSK = []
            self.testXSK = []
            self.trainYSK = []
            self.testYSK = []
            self.GetData()
            self.GetDataSK()
            self.MLPSettings = self.Settings()
            self.Regressor = self.MLP_Initialize()
            self.RegressorSK = self.SK_Initialize()
        else:
            path_pkl = r"C:/Users/Corentin/OneDrive - vki.ac.be/Project/Project/Quadcopter-Simulator-main/Quadcopter-Simulator-main/INVERSE_UPDATED.pkl"
            self.Regressor = self.ImportModel(path_pkl)

    def ReferenceSettings(self, new_values):
        self.RPM_REFERENCE = new_values[0]
        self.T_REFERENCE = new_values[1]
        self.P_REFERENCE = new_values[2]

    def GetData(self, test_size=0.3):

        DATA = pd.read_excel(self.data_ext)
        inps = DATA.iloc[:, [1,2,3]]#[i for i in range(self.num_of_inputs)]]
        outs = DATA.iloc[:, [0,4]]#[i+self.num_of_inputs for i in range(self.num_of_inputs-1)]]
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(inps, outs, test_size=test_size, random_state=4)

    def GetDataSK(self, test_size=0.3):
        DATA = pd.read_excel(self.data_ext)
        inps = DATA.iloc[:, [1,2,3]]#[i for i in range(self.num_of_inputs)]]
        outs = DATA.iloc[:, [0,4]]#[i+self.num_of_inputs for i in range(self.num_of_inputs-1)]]
        poly_inps = self.poly.fit_transform(inps)
        self.trainXSK, self.testXSK, self.trainYSK, self.testYSK = train_test_split(poly_inps, outs, test_size=test_size, random_state=4)

    def Settings(self, alpha=0.1, batch_size = 2048, random_state = 0, tolerance=0.000001, max_iter=100000):

        params = {'hidden_layer_sizes': self.hls ,
                  'activation': self.activator,
                  'solver': self.solver,
                  'alpha': alpha,
                  'batch_size': batch_size,
                  'random_state': random_state,
                  'tol': tolerance,
                  'nesterovs_momentum': False,
                  'learning_rate': 'constant',
                  'learning_rate_init': 0.001,
                  'max_iter': max_iter,
                  'shuffle': True,
                  'n_iter_no_change': 50,
                  'verbose': False}
        return params


    def MLP_Initialize(self):
        print(f"\nInitializing {self.num_of_inputs}-({len(self.hls)}x{self.hls[0]})-2 {self.activator} neural network")
        start_time = time.time()
        MLP = MLPRegressor(**self.MLPSettings)
        MLP.fit(self.trainX, self.trainY)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training complete in {execution_time} s")
        return MLP

    def SK_Initialize(self):
        SK = linear_model.LinearRegression()
        SK.fit(self.trainXSK, self.trainYSK)
        return SK

    def Calculate(self,V_inf, a_disk, T):

        start_time = time.time()
        inp = np.array([[ V_inf/self.V_REFERENCE, a_disk, T/self.T_REFERENCE]])
        var = self.poly.fit_transform(np.array([V_inf/self.V_REFERENCE, a_disk, T/self.T_REFERENCE]).reshape(1, -1))
        RPM = 0
        P = 0
        if self.mode == 1:
            rs = self.Regressor.predict(inp)
            map(float, rs)
            RPM = rs[0][0]
            P = rs[0][1]
        elif self.mode == 0:
            rs = self.RegressorSK.predict(var)
            map(float, rs)
            RPM = rs[0][0]
            P = rs[0][1]
        end_time = time.time()
        execution_time = end_time - start_time
        #print("Calculated values in  %0.15f s" % execution_time)
        return (RPM * self.RPM_REFERENCE, P * self.P_REFERENCE)

    def Diagnostics(self):
        if self.mode == 1:
            res_pred = self.Regressor.predict(self.testX)
            mae = mean_absolute_error(self.testY, res_pred)
            mse = mean_squared_error(self.testY, res_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.testY, res_pred)
        else:
            res_pred = self.RegressorSK.predict(self.testXSK)
            mae = mean_absolute_error(self.testYSK, res_pred)
            mse = mean_squared_error(self.testYSK, res_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.testYSK, res_pred)
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared (R²): {r2}")

    def ImportModel(self, model_extension):
        try:
            with open(model_extension, 'rb') as file:
                model = pickle.load(file)
            return model
        except:
            raise FileNotFoundError

    def ExportModel(self):
        model_pkl_file = "INVERSE_UPDATED.pkl"
        try:
            with open(model_pkl_file, 'wb') as file:
                pickle.dump(self.Regressor, file)
        except:
            raise FileExistsError
