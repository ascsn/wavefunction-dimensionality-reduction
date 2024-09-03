from CAE_NIF import CAE
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input
import numpy as np
import pysindy as ps
import torch
from scipy.interpolate import UnivariateSpline
import math



class Error():

    def __init__(self, tim, ls, dat, deg, thresh, smooth=False, num_turning_points=4, search=False):
        self.tim = tim
        self.lat_size = ls
        self.data = dat
        self.degree = deg
        self.thresh = thresh
        self.smooth = smooth
        self.ntp = num_turning_points
        self.search = search
        self.autoencoder = CAE(self.lat_size,self.data.shape[1])
        self.autoencoder.load_weights(f"./saved_weights/PySINDY-{0}/QHO{self.lat_size}Lat").expect_partial()


    def get_error(self):
        time = np.linspace(0,40,4000)
        latent = self.autoencoder.encode(self.data)
    
        if self.smooth:
            print('You are smoothing')
            new_latent = []
            for lat in range(self.lat_size):
                bad = True
                itera = -0.1
                while bad:
                    itera += 0.1
                    count = 0
                    a = [0]
                    spline = UnivariateSpline(time, latent[:,lat], s=itera)
                    l = spline(time)
            
                    dt = time[1]-time[0]
                    for j in range(0,len(latent)-1):
                        a.append((l[j+1] - l[j])/dt)
            
                    for i in range(5,tim):
                        if a[i-2] < a[i-1] and a[i] < a[i-1]:
                            count += 1
                        elif a[i-2] > a[i-1] and a[i] > a[i-1]:
                            count += 1
                    if count == self.num_turning_points:
                        bad = False
                        new_latent.append(l)
            new = np.array(new_latent)
            new = new.T
            latent = new
            print('Smoothing complete')
        
        ti = self.tim
        lat = latent[0:]
        dt = 0.01
        a = np.zeros([self.data.shape[0],self.lat_size])
        
        for i in range(self.lat_size):
            a[0,i] = 0
        
        for i in range(0,self.data.shape[0]-1):
            for j in range(self.lat_size):
                a[i+1,j] = ((lat[i+1,j] - lat[i,j])/dt)
                
        if self.search:
            deg = []
            thresh = []
            print('You are searching')
            for i in range(self.lat_size):
                X = np.stack((latent[0:ti,i],a[0:ti, i]), axis=-1)
                X_1 = np.stack((latent[0:ti,i],a[0:ti, i]), axis=-1)
                differentiation_method = ps.FiniteDifference(order=1)
                
                # Define parameter grid
                thresholds = np.arange(0.01, 0.11, 0.01)  # Example range for thresholds
                degrees = range(3, 5)  # Polynomial degrees from 3 to 4
                
                # Placeholder for the best parameters and lowest error
                best_threshold = None
                best_degree = None
                lowest_error = np.inf
                # Loop over all combinations of thresholds and degrees
                for threshold in thresholds:
                    for degree in degrees:
                        X_pred=None
                        feature_library = ps.PolynomialLibrary(degree=degree)
                        optimizer = ps.STLSQ(threshold=threshold)
                        
                        model = ps.SINDy(
                            differentiation_method=differentiation_method,
                            feature_library=feature_library,
                            optimizer=optimizer,
                            feature_names=["lat0", "lat1"]
                        )
                        
                        # Fit the model (assuming you have the target variable `y`)
                        model.fit(X, t=0.01)
                        # Predict using the model
                        X_pred = model.simulate([X[0,0], a[0,i]], time, integrator_kws={'atol': 1e-12, 'method': 'RK45', 'rtol': 1e-12}, interpolator_kws={})
                        # Calculate the error (e.g., mean squared error)
                        error = mean_squared_error(X_1[0:len(X_pred)], X_pred)
                        # Update the best parameters if the current error is lower
                        if error < lowest_error:
                            lowest_error = error
                            best_threshold = threshold
                            best_degree = degree
                        
                # Output the best parameters
                print(f"Best threshold: {best_threshold}", i)
                print(f"Best polynomial degree: {best_degree}")
                print(f"Lowest error: {lowest_error}")
                deg.append(best_degree)
                thresh.append(best_threshold)
                if lowest_error > 0.1:
                    print('You may want to try other values of degree and threshold by hand')
    
        
                differentiation_method = ps.FiniteDifference(order=1)
                feature_library = ps.PolynomialLibrary(degree=deg[i])
                optimizer = ps.STLSQ(threshold=thresh[i])
                model = ps.SINDy(differentiation_method=differentiation_method,feature_library=feature_library,
                optimizer=optimizer,feature_names=["lat0","lat1"],)  
                
                
                X = np.stack((latent[0:ti,i],a[0:ti, i]), axis=-1)
                model.fit(X, t=0.01)
                globals()[f"x_simulated{i}"] = model.simulate([X[0,0],a[0,i]], time, integrator_kws={'atol': 1e-12, 'method': 'RK45', 'rtol': 1e-12}, interpolator_kws={})
                
                
        else:
            differentiation_method = ps.FiniteDifference(order=1)
            feature_library = ps.PolynomialLibrary(degree=self.degree)
            optimizer = ps.STLSQ(threshold=self.thresh)
            model = ps.SINDy(differentiation_method=differentiation_method,feature_library=feature_library,
            optimizer=optimizer,feature_names=["lat0","lat1"],) 
            
            for i in range(self.lat_size):
                X = np.stack((latent[0:ti,i],a[0:ti,i]), axis=-1)
                model.fit(X, t=0.01)
                globals()[f"x_simulated{i}"] = model.simulate([X[0,0],a[0,i]], time, integrator_kws={'atol': 1e-12, 'method': 'RK45', 'rtol': 1e-12}, interpolator_kws={})

    
            #plt.plot(globals()[f"x_simulated{i}"][:,0], globals()[f"x_simulated{i}"][:,1], label='Simulated')
            #plt.plot(X[:,0], X[:,1], alpha=0.5, label='Actual')

        variable_names = np.array([f"x_simulated{i}" for i in range(self.lat_size)])
        print(variable_names.shape)
        py = np.column_stack([*(globals()[name][:,0] for name in variable_names)])
        py_dec = np.array(self.autoencoder.decode(py))
        pred = np.array(self.autoencoder.decode(latent))
        
        for i in range(0,len(self.data),40):
            if i == 0:
                error = []
                pred_error = []
            error.append((((self.data[i]-py_dec[i])**2)*0.05).sum())
            pred_error.append((((self.data[i]-pred[i])**2)*0.05).sum())
        
        x = np.arange(0,100)
    
        plt.scatter(x, error, color='red', label='Poly 1', linestyle=':')
        plt.scatter(x, pred_error, color='blue', label='Poly 1', linestyle=':')
        plt.yscale('log')
        return np.array(error).sum()/len(error)