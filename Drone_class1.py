#%%Imports
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:29:06 2025

@author: Corentin
"""
from BEMT_NN_INVERSE_CORRECT import BEMT_NN
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, diff, lambdify
from scipy.optimize import minimize
import statsmodels.api as sm
from scipy.signal import butter, filtfilt
import pandas as pd
import os
from scipy.signal import resample

#%% Class Drone
class Drone:
    def __init__(self, m, l_arm, I, n_rotor):
        #Drone properties
        self.m = m
        self.l_arm = l_arm
        self.I = I
        self.n_rotor = n_rotor
        
        #General parameters
        self.g = 9.81 #Gravity cst
        self.T_initial = self.m * self.g      
        #add fro hexacopter
        
#%%Class Trajectory
       
class Trajectory(Drone):
    
    def __init__(self, x, y, z, t, key, total_steps, drone ,n_order = 4):
        #Importing waypoints for the trajectory
        self.x_waypoint = x
        self.y_waypoint = y
        self.z_waypoint = z
        self.t_waypoint = t
        self.key = key # identifier for the trajectory
        self.total_steps = total_steps # number of interpolation steps between waypoints
        self.n = n_order #order of calculation to size the matrices correctly for the calculation of the trajectory
        
        #Creating a time vector
        self.t_vec = np.linspace(self.t_waypoint[0], self.t_waypoint[-1], self.total_steps)
        
        #Parameters for trajectory generation
        self.theta_initial = 0.1 # initial pitch angle [rad]
        self.phi_initial = 0.1 # initial roll angle [rad]
        self.psi = np.zeros(self.total_steps) # initialize yaw (psi) to zero over the whole trajectory --> not taken into account during the calculations
        self.angle_max = 45
        
        #Bounds of optimization: thrust T must be positive, theta and phi bounded to ±45 deg
        self.bounds = [(0.01, None), 
                       (-self.angle_max*np.pi/180, self.angle_max*np.pi/180), 
                       (-self.angle_max*np.pi/180, self.angle_max*np.pi/180)]  
        
        self.difference_method = 4
        self.filter_para = [0.5, 0.066, 0.066] #LOWES FILTER parameter aka d
        
        print('Total amount of steps: ', self.total_steps)
        #print('Moving average windowsize becomes: ', round(total_steps*0.02))
        print('Difference method used to obtain angular velocities: ', self.difference_method, 'neighbouring points')
        print('')
        
        #Parameters for NN inputs
        self.data_extension = "DATA_18092024.xlsx"
        self.num_of_inputs = 3 # rpm/rpm_ref , V/Vref, a_disk
        self.hidden_layer_size = [32]*5
        self.activator = "tanh"
        self.solver = "adam"
        self.initialize = False
        self.NN = BEMT_NN(self.initialize, self.data_extension, self.num_of_inputs, self.hidden_layer_size, activator=self.activator, solver=self.solver, mode=1)
        self.NN.ExportModel()
        
        #Parameters of the drone
        self.drone = drone #drone object with defined parameters
        self.m = self.drone.m #mass of the drone
        self.g = self.drone.g #gravity
        self.I = self.drone.I #drone's inertia matrix
        self.l_arm = self.drone.l_arm #lenght of the drone's arm
        
        #function to calculate the trajectory of the drone
        self._cal_traj()
        self.individual_thrust = False

            
    def _generate_boundary_conditions(self, BC_endpoints, BC_intermediatepoints):
        """
        Generate the boundary condition vectors for x, y, z trajectories.
    
        Parameters:
        ----------
        BC_endpoints : int
            Number of boundary conditions applied at the start and end of each segment (e.g., position, velocity).
        BC_intermediatepoints : int
            Number of boundary conditions applied at intermediate points (between waypoints).
    
        Returns:
        -------
        b : np.ndarray
            Full vector containing all boundary conditions (shared across x, y, z).
        b_x : np.ndarray
            Boundary condition vector for x-coordinate.
        b_y : np.ndarray
            Boundary condition vector for y-coordinate.
        b_z : np.ndarray
            Boundary condition vector for z-coordinate.
        
        Notes:
        -----
        - This function constructs the required boundary condition vectors that will later be used 
          to solve for the polynomial trajectory coefficients.
        - It ensures the trajectory passes through all waypoints and satisfies derivative constraints.
        """
        pos_matrix = (self.x_waypoint, self.y_waypoint, self.z_waypoint)
        m = len(self.x_waypoint) - 1
        b_x = np.zeros((2*self.n*m, 1))
        b_y = np.zeros((2*self.n*m, 1))
        b_z = np.zeros((2*self.n*m, 1))

        # Fill b vectors (boundary and intermediate conditions)    
        for i in range(0,3):  # x,y,z coordinates
                b = [b_x, b_y, b_z][i]
                b[0] = pos_matrix[i][0]
                b[len(b)-BC_endpoints-1] = pos_matrix[i][-1]
                n_intermediatepoints = len(self.x_waypoint) - 2
                
                for i_b in range(0, n_intermediatepoints):
                    # Complete b vector
                    for j_b in range(BC_endpoints+1, len(b)-BC_endpoints-1):
                        if j_b == ((i_b)*(BC_intermediatepoints+2)) + (BC_endpoints+1):
                            b[j_b] = pos_matrix[i][i_b+1]
                        elif j_b == ((i_b)*(BC_intermediatepoints+2)) + (BC_endpoints+1+BC_intermediatepoints+1):
                            b[j_b] = pos_matrix[i][i_b+1]
        
        return b, b_x, b_y, b_z
    
    def _generate_function_matrix(self, n_poly, BC_intermediatepoints):
        """
        Generate the matrix of polynomial basis functions and their derivatives.
    
        Parameters:
        ----------
        n_poly : int
            Maximum degree of the polynomial (highest exponent).
        BC_intermediatepoints : int
            Number of intermediate derivative orders needed.
    
        Returns:
        -------
        f_matrix : list of lists
            A matrix where f_matrix[i][j] gives the j-th polynomial derivative function at derivative order i.
            (i.e., evaluates t**j after i derivatives.)
        """
        t = Symbol('t')
        f_matrix = [[None for _ in range(n_poly + 1)] for _ in range(BC_intermediatepoints + 1)]
        
        for i in range(n_poly + 1):
            expression = t ** i
            f_matrix[0][i] = lambdify(t, expression)
            for j in range(1, BC_intermediatepoints + 1):
                expression = diff(expression, t)
                f_matrix[j][i] = lambdify(t, expression)
        
        return f_matrix
    
    def _generate_A_matrix(self, m, n_poly, n_coeff, BC_endpoints, BC_intermediatepoints, f_matrix, b):
        """
        Generate the A matrix used in the linear system A * p = b, where p are the polynomial coefficients.
        
        Parameters:
        ----------
        m : int
            Number of trajectory segments (number of waypoints - 1).
        n_poly : int
            Maximum degree of the polynomial.
        n_coeff : int
            Number of polynomial coefficients to solve per segment.
        BC_endpoints : int
            Number of boundary conditions at the endpoints.
        BC_intermediatepoints : int
            Number of intermediate boundary conditions (continuity conditions between segments).
        f_matrix : list
            Matrix of function derivatives (from _generate_function_matrix).
        b : np.ndarray
            Boundary condition vector.
        
        Returns:
        -------
        A : np.ndarray
            Constraint matrix A that relates the polynomial coefficients to the boundary conditions.
        """
        A = np.zeros((2*self.n*m, 2*self.n*m))
        
        
        # Define boundary conditions of x0 and xm+1
        for i in range(BC_endpoints+1):
            A[i, n_coeff-i-1] = f_matrix[i][i](0)   # for t = 0 
        
        width_f_matrix = len(f_matrix[0])
        
        p = -1
        p_index = 1
        diff_order = BC_endpoints + 1
        
        for i in range(len(b)-1, len(b)-BC_endpoints-2, -1):
            for j in range(len(b)-1, len(b)-n_poly-2, -1):
                p += 1
                p_index += 1
                
                if p % width_f_matrix == 0:
                    diff_order -= 1
                    p_index = 1
                
                A[i, j] = f_matrix[diff_order][p_index-1](self.t_waypoint[-1])
                
        
        
        # Define B.C's intermediate point(s)
        
        f_matrix_reorder = [row[::-1] for row in f_matrix]
        
        
        n_intermediatepoints = len(self.t_waypoint) - 2
        #i_index = -1
        diff_order_intermediatepoints = -1
        i_sequence = -1
        i_poly = 1
        
        for i in range(0,n_intermediatepoints):
            for j in range(BC_endpoints+1+i*(BC_intermediatepoints+2), BC_endpoints+i*(BC_intermediatepoints+2)+(BC_intermediatepoints+3)):
                if diff_order_intermediatepoints >= (BC_intermediatepoints + 1):
                   diff_order_intermediatepoints = -1
                diff_order_intermediatepoints += 1
                for k in range(i*n_coeff, i*n_coeff + 2*n_coeff):
                       i_sequence += 1
                       if diff_order_intermediatepoints <= BC_intermediatepoints:
                            if i_poly % 2 == 0:
                                if j == BC_endpoints+1+i*(2+BC_intermediatepoints):
                                    if width_f_matrix - diff_order_intermediatepoints > i_sequence:
                                       A[j, k] = f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])
                                    else:
                                       A[j, k] = f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])
                                if diff_order_intermediatepoints == 0:
                                    A[j, k] = 0
                                elif width_f_matrix - diff_order_intermediatepoints > i_sequence:
                                    A[j, k] = -f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])
                                else:
                                    A[j, k] = -f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])  
                            else:
                                 if width_f_matrix - diff_order_intermediatepoints > i_sequence:
                                     A[j, k] = f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])
                                 else:
                                     A[j, k] = f_matrix_reorder[diff_order_intermediatepoints][i_sequence](self.t_waypoint[i+1])
                       else: 
                           if i_poly % 2 == 0:
                                 if j == BC_endpoints+(i+1)*(2+BC_intermediatepoints):
                                    if i_sequence+1 == width_f_matrix:
                                       A[j, k] = f_matrix_reorder[0][i_sequence](self.t_waypoint[i+1])
                                    else:
                                       A[j, k] = f_matrix_reorder[0][i_sequence](self.t_waypoint[i+1])
                       if (i_sequence+1) % width_f_matrix == 0:
                            i_sequence = -1
                            i_poly += 1
        
        return A
    
    def _trajectory_algorithm(self):
        """
        Solve for the polynomial coefficients that define the full trajectory in x, y, and z directions.
    
        Returns:
        -------
        p_x, p_y, p_z : np.ndarray
            Coefficient vectors for the x, y, and z trajectories respectively.
        """
        
        m = len(self.x_waypoint) - 1  # Number of points - 1
        BC_endpoints = self.n - 1
        BC_intermediatepoints = 2 * (self.n - 1)
        n_poly = self.n * 2 - 1
        n_coeff = n_poly + 1               # number of coefficients that need to be solved for
        
        # Step 1: Generate boundary conditions (b vectors)
        b, b_x, b_y, b_z = self._generate_boundary_conditions(BC_endpoints, BC_intermediatepoints)
        
        # Step 2: Generate function matrix (polynomial derivatives)
        f_matrix = self._generate_function_matrix(n_poly, BC_intermediatepoints)
        
        #print(b)
        # Step 3: Generate A matrix based on boundary conditions and intermediate points
        A = self._generate_A_matrix(m, n_poly ,n_coeff, BC_endpoints, BC_intermediatepoints, f_matrix, b)
        
        
        # Step 4: Solve for polynomial coefficients
        p_x = np.linalg.solve(A, b_x)
        p_y = np.linalg.solve(A, b_y)
        p_z = np.linalg.solve(A, b_z)
        
        return p_x, p_y, p_z
    
    def _cost_function(self, T_theta_phi, a_x_desired, a_y_desired, a_z_desired):
        """
        Compute the cost based on the difference between actual and desired accelerations.
    
        Parameters:
        ----------
        T_theta_phi : list or np.ndarray
            Thrust, pitch (theta), and roll (phi) values [T, theta, phi].
        a_x_desired, a_y_desired, a_z_desired : float
            Desired accelerations along x, y, and z axes.
    
        Returns:
        -------
        cost : float
            Squared error between actual and desired accelerations (cost value).
        """
        T, theta, phi = T_theta_phi
        psi = self.psi[0]
        if T <= 0:
            return np.inf  # Penalize invalid thrust values
        
        # Calculate actual accelerations
        a_x_actual = T * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi) ) / self.drone.m
        a_y_actual = T * (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi) ) / self.drone.m
        a_z_actual = (T * np.cos(theta) * np.cos(phi) - self.drone.m * self.drone.g) / self.drone.m
        
        # Penalize deviations from desired accelerations
        E_x = (a_x_actual - a_x_desired)**2
        E_y = (a_y_actual - a_y_desired)**2
        E_z = (a_z_actual - a_z_desired)**2
        
        return E_x + E_y + E_z
    
    def _gradient_function(self, T_theta_phi, a_x_desired, a_y_desired, a_z_desired):
        """
        Compute the gradient of the cost function with respect to thrust, pitch, and roll.
    
        Parameters:
        ----------
        T_theta_phi : list or np.ndarray
            Thrust, pitch (theta), and roll (phi) values [T, theta, phi].
        a_x_desired, a_y_desired, a_z_desired : float
            Desired accelerations along x, y, and z axes.
    
        Returns:
        -------
        gradient : np.ndarray
            Gradient vector [dE/dT, dE/dtheta, dE/dphi].
        """
        
        T, theta, phi = T_theta_phi
        a_x_actual = (T * np.sin(theta) * np.cos(phi) ) / self.drone.m
        a_y_actual = (T * -np.sin(phi) ) / self.drone.m
        a_z_actual = (T * np.cos(theta) * np.cos(phi) - self.drone.m * self.drone.g) / self.drone.m  
        
       # If psi is a variable, use the following muted comments a_x and a_y:
       # a_x_actual = T * (np.sin(phi) * np.sin(psi) + np.cos(psi) * np.sin(theta) * np.cos(phi) ) / m
       # a_y_actual = T * (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi) ) / m
       
        dE_totaldT =     2 * (a_x_actual - a_x_desired) * ( np.sin(theta) * np.cos(phi) ) / self.m  + 2 * (a_y_actual - a_y_desired) * (-np.sin(phi) ) / self.m  +  2 * (a_z_actual - a_z_desired) * (np.cos(theta) * np.cos(phi)) / self.m  
        dE_totaldtheta = 2 * (a_x_actual - a_x_desired) * T * ( np.cos(theta) * np.cos(phi)) / self.m + 2 * (a_z_actual - a_z_desired) * T * (-np.sin(theta) * np.cos(phi)) / self.m
        dE_totaldphi =   2 * (a_x_actual - a_x_desired) * ( -T * np.sin(theta) * np.sin(phi)) / self.m +  2 * (a_y_actual - a_y_desired) * T * (-np.cos(phi)) / self.m + 2 * (a_z_actual - a_z_desired) * T * (-np.cos(theta) * np.sin(phi)) / self.m

       # if psi is a variable, use the following muted comments for the derivatives:
       # dE_totaldT = 2 * (a_x_actual - a_x_desired) * ( np.sin(phi) * np.sin(psi) + np.cos(psi) * np.sin(theta) * np.cos(phi) ) / m     + 2 * (a_y_actual - a_y_desired) * (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi) ) / m    + 2 * (a_z_actual - a_z_desired) * (np.cos(theta) * np.cos(phi)) / m  
       # dE_totaldphi = 2 * (a_x_actual - a_x_desired) * T * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(theta) * np.sin(phi)) / m + 2 * (a_y_actual - a_y_desired) * T * (-np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(theta) * np.sin(psi)) / m + 2 * (a_z_actual - a_z_desired) * T * (-np.cos(theta) * np.sin(phi)) / m
       # dE_totaldtheta = 2 * (a_x_actual - a_x_desired) * T * (np.cos(psi) * np.cos(theta) * np.cos(phi)) / m + 2 * (a_y_actual - a_y_desired) * T * (np.cos(phi) * np.cos(theta) * np.sin(psi)) / m + 2 * (a_z_actual - a_z_desired) * T * (-np.sin(theta) * np.cos(phi)) / m
       # dE_totaldpsi =   2 * (a_x_actual - a_x_desired) * T * (np.sin(phi) * np.cos(psi) - np.sin(psi) * np.sin(theta) * np.cos(phi)) / m + \
                        # 2 * (a_y_actual - a_y_desired) * T * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.co(psi)) / m 
        
        # Add dE_totaldpsi in the return line too!!
        return np.array([dE_totaldT, dE_totaldtheta,  dE_totaldphi])
    
    def _T_attitude_optimal(self, a_x_desired, a_y_desired, a_z_desired, time_vector_cost):
        """
        Find the optimal thrust and attitude (theta, phi) that produce desired accelerations.
    
        Parameters:
        ----------
        a_x_desired, a_y_desired, a_z_desired : function
            Functions of time returning the desired accelerations.
        time_vector_cost : list or np.ndarray
            Time vector over which the optimization is performed.
    
        Returns:
        -------
        T_optimal : list
            Optimal thrust values over time.
        phi_optimal : list
            Optimal roll angles over time.
        theta_optimal : list
            Optimal pitch angles over time.
        time_vector_cost : list
            Same as input (used for synchronization).
        """
        print('Cost function is running...')
        T_optimal = []
        theta_optimal = []
        phi_optimal = []
        

        for t in time_vector_cost:
            res = minimize(self._cost_function, [self.drone.T_initial, self.theta_initial, self.phi_initial], args=(a_x_desired(t), a_y_desired(t), a_z_desired(t)), method = 'L-BFGS-B',  bounds=self.bounds, jac = self._gradient_function)
            T_optimal.append(res.x[0])
            theta_optimal.append(res.x[1])
            phi_optimal.append(res.x[2])
            
            T_initial, theta_initial, phi_initial = res.x
            
        return T_optimal, phi_optimal, theta_optimal, time_vector_cost
    
    def _explicit_method(self, a_actual, v_0, v_end):
        """
        Integrate acceleration data to obtain velocity using explicit Euler method.
    
        Parameters:
        ----------
        a_actual : array-like
            List of actual accelerations.
        v_0 : float
            Initial velocity.
        v_end : float
            Final velocity (used for end correction).
    
        Returns:
        -------
        v_actual : np.ndarray
            Integrated velocity values.
        """
        n = len(a_actual)
        v_actual = np.zeros(n)
        dt = self.t_vec[1] - self.t_vec[0]
        for i in range(n):   
            if i == 0:
                v_actual[i] = v_0
            elif i == n-1:
                if v_end <= 1e-3:
                    v_end = 0
                else:
                    v_actual[i] = v_end
            else:
                v_actual[i] = v_actual[i-1] + a_actual[i-1]*dt
                if np.abs(v_actual[i]) <= 1e-6:
                    v_actual[i] = 0
        return v_actual 
    
    def _forward_difference(self,theta, i, dt, order):
        """
        Compute the derivative using forward finite difference.
        
        Parameters:
        ----------
        theta : array-like
            Vector of values (e.g., angles).
        i : int
            Index at which to compute the derivative.
        dt : float
            Time step size.
        order : int
            Order of the difference (2 or 4).
        
        Returns:
        -------
        derivative : float
            Approximated derivative at index i.
        """
        if order == 2:
            return (theta[i + 1] - theta[i]) / dt
        elif order == 4:
            return (-11 * theta[i] + 18 * theta[i + 1] - 9 * theta[i + 2] + 2 * theta[i + 3]) / (6 * dt)

    def _backward_difference(self, theta, i, dt, order):
        """
        Compute the derivative using backward finite difference.
    
        Parameters:
        ----------
        theta : array-like
            Vector of values (e.g., angles).
        i : int
            Index at which to compute the derivative.
        dt : float
            Time step size.
        order : int
            Order of the difference (2 or 4).
    
        Returns:
        -------
        derivative : float
            Approximated derivative at index i.
        """
        if order == 2:
            return (theta[i] - theta[i - 1]) / dt
        elif order == 4:
            return (11 * theta[i] - 18 * theta[i - 1] + 9 * theta[i - 2] - 2 * theta[i - 3]) / (6 * dt)

    def _central_difference(self, theta, i, dt, order):
        """
        Compute the derivative using central finite difference.
    
        Parameters:
        ----------
        theta : array-like
            Vector of values (e.g., angles).
        i : int
            Index at which to compute the derivative.
        dt : float
            Time step size.
        order : int
            Order of the difference (2 or 4).
    
        Returns:
        -------
        derivative : float
            Approximated derivative at index i.
        """
        if order == 2:
            return (theta[i + 1] - theta[i - 1]) / (2 * dt)
        elif order == 4:
            return (-theta[i + 2] + 8 * theta[i + 1] - 8 * theta[i - 1] + theta[i - 2]) / (12 * dt)


    # Refactored method (using the new code)
    def _get_eulerdot_difference_method(self, theta):
        """
        Compute the time derivative of Euler angles using finite difference methods.
    
        Parameters:
        ----------
        theta : np.ndarray
            Euler angle values (roll, pitch, or yaw) over time.
    
        Returns:
        -------
        thetadot : np.ndarray
            Approximated time derivative of the Euler angle.
        """
        n = len(theta)
        thetadot = np.zeros(n)
        #delta_t = [0]*n

        for i in range(n):
            if self.difference_method == 4:
                if i < 2:  # Forward difference for first two points
                    dt = self.t_vec[i + 1] - self.t_vec[i]
                    thetadot[i] = self._forward_difference(theta, i, dt, order=4)
                elif i >= n - 2:  # Backward difference for last two points
                    dt = self.t_vec[i] - self.t_vec[i - 1]
                    thetadot[i] = self._backward_difference(theta, i, dt, order=4)
                else:  # Central difference for interior points
                    dt = self.t_vec[i + 1] - self.t_vec[i]
                    thetadot[i] = self._central_difference(theta, i, dt, order=4)

            elif self.difference_method == 2:
                if i == 0:  # Forward difference
                    dt = self.t_vec[i + 1] - self.t_vec[i]
                    thetadot[i] = self._forward_difference(theta, i, dt, order=2)
                elif i == n - 1:  # Backward difference
                    dt = self.t_vec[i] - self.t_vec[i - 1]
                    thetadot[i] = self._backward_difference(theta, i, dt, order=2)
                else:  # Central difference
                    dt = self.t_vec[i + 1] - self.t_vec[i]
                    thetadot[i] = self._central_difference(theta, i, dt, order=2)

            #delta_t[i] = dt  # Store time step size

        return thetadot
    
    def _get_actual_accelerations(self, T, theta, phi):
        """
        Compute actual accelerations (a_x, a_y, a_z) from thrust and attitude.
    
        Parameters:
        ----------
        T : array-like
            Thrust values over time.
        theta, phi : array-like
            Pitch and roll angles over time.
    
        Returns:
        -------
        a_x_actual, a_y_actual, a_z_actual : list
            Lists of computed accelerations in each axis.
        """
        a_x_actual = []
        a_y_actual = []
        a_z_actual = []
        for i in range(len(T)):
            a_x_actual.append(T[i] * (np.sin(phi[i]) * np.sin(self.psi[i]) + np.cos(self.psi[i]) * np.sin(theta[i]) * np.cos(phi[i]) ) / self.m)
            a_y_actual.append(T[i] * (-np.sin(phi[i]) * np.cos(self.psi[i]) + np.cos(phi[i]) * np.sin(theta[i]) * np.sin(self.psi[i]) ) / self.m)
            a_z_actual.append((T[i] * np.cos(theta[i]) * np.cos(phi[i]) - self.m * self.g) / self.m)
        return a_x_actual, a_y_actual, a_z_actual
    
    def _filter_LOWESS_method_vec(self, vector, d):
        """
        Description: Filter to denoise the vector
        Parameters
        ----------
        vector : TYPE
            Data to be smoothed.
        time_vector : TYPE
            Corresponding time values.
        d : TYPE
            Frac parameter, which controls the smoothing (fraction of data used for local regression.

        Returns
        -------
        vector : TYPE
            smoothed version of vector.

        """
        x= self.t_vec
        y = vector
        lowess = sm.nonparametric.lowess(y, x, frac=d)
        vector = lowess[:,1]
        return vector
    
    def _get_bodyrates(self, total_steps, phi_optimal_part, phidot_optimal_part, theta_optimal_part,thetadot_optimal_part, psidot_part):
        """
        Compute body rates (p, q, r) and their derivatives from Euler angles and angular rates.
    
        Parameters:
        ----------
        total_steps : int
            Number of simulation steps.
        phi_optimal_part, phidot_optimal_part : array-like
            Roll angle and roll rate.
        theta_optimal_part, thetadot_optimal_part : array-like
            Pitch angle and pitch rate.
        psidot_part : array-like
            Yaw rate.
    
        Returns:
        -------
        p, q, r : np.ndarray
            Body angular rates.
        pdot, qdot, rdot : np.ndarray
            Derivatives of body angular rates.
        pdot_smooth, qdot_smooth, rdot_smooth : np.ndarray
            Smoothed derivatives of body angular rates.
        """
        phi = np.array(phi_optimal_part)
        theta = np.array(theta_optimal_part)
        phidot = np.array(phidot_optimal_part)
        thetadot = np.array(thetadot_optimal_part)
        psidot = np.array(psidot_part)
        # Compute rotation matrices for all time steps at once
        R_rot = np.array([
            [np.ones(total_steps), np.zeros(total_steps), -np.sin(theta)],
            [np.zeros(total_steps), np.cos(phi), np.sin(phi) * np.cos(theta)],
            [np.zeros(total_steps), -np.sin(phi), np.cos(phi) * np.cos(theta)]
        ]).transpose(2, 0, 1)  # Reshape for batch processing
        
        eulerdot_vec = np.vstack([phidot, thetadot, psidot]).T  # Shape (total_steps, 3)
        bodyrate = np.einsum('ijk,ik->ij', R_rot, eulerdot_vec)  # Vectorized matrix multiplication
        
        p, q, r = bodyrate[:, 0], bodyrate[:, 1], bodyrate[:, 2]
        
        # Compute derivatives and apply smoothing
        pdot = self._get_eulerdot_difference_method(p)
        pdot_smooth = self._filter_LOWESS_method_vec(pdot, self.filter_para[0])
        qdot = self._get_eulerdot_difference_method(q)
        qdot_smooth = self._filter_LOWESS_method_vec(qdot, self.filter_para[1])
        rdot = self._get_eulerdot_difference_method(r)
        rdot_smooth = self._filter_LOWESS_method_vec(rdot, self.filter_para[2])
        
        return p,q,r, pdot, qdot, rdot, pdot_smooth, qdot_smooth, rdot_smooth
    
    def _filter_LOWESS_method_matrix(self, rpm_total_, t_rpm):
        """
        Apply LOWESS smoothing filter to multiple RPM signals (matrix).
    
        Parameters:
        ----------
        rpm_total_ : list of array-like
            List of RPM signals.
        t_rpm : array-like
            Time vector corresponding to RPM signals.
    
        Returns:
        -------
        rpm_total_filtered : list of np.ndarray
            Smoothed RPM signals.
        """
        
        rpm_total_filtered = [None]*len(rpm_total_)
        for i in range(len(rpm_total_)):
            x = t_rpm
            y = rpm_total_[i]
            lowess = sm.nonparametric.lowess(y, x, frac=self.filter_para)
            rpm_total_filtered[i] = lowess[:,1]
            
        return rpm_total_filtered
    
    def _get_trajectory(self):
        #Code has to be extended for n=1,2,3 if needed if not erase case 1,2,3
        """
        Generate the full 3D drone trajectory based on waypoints and polynomials.
    
        Returns:
        -------
        desired_positions : list of np.ndarray
        actual_positions : list of np.ndarray
        desired_velocities : list of np.ndarray
        actual_velocities : list of np.ndarray
        desired_accelerations : list of np.ndarray
        actual_accelerations : list of np.ndarray
        T_optimal : list
        euler_angles : list of np.ndarray
        euler_velocities : list of np.ndarray
        body_rates : list of np.ndarray
        body_rates_dot_smooth : list of np.ndarray
        body_rates_dot : list of np.ndarray
        """
        i_t = 1
        i_func = -1
        x_desired, y_desired, z_desired = [],[],[]
        vx_desired, vy_desired, vz_desired = [],[],[]
        ax_vec, ay_vec, az_vec = [], [],[]
        #vx_actual = []
        #x_sec = []
        t_sec = []
        T_optimal, phi_optimal, theta_optimal = [],[],[]
        timesteps = 1000
        
        
        for i in range(self.total_steps):
            t_bound = self.t_vec[i]
            t_sec += [t_bound]
            n_poly = self.n * 2 - 1
            
            # Check if the time boundary reaches the next waypoint
            if t_bound >= self.t_waypoint[i_t]:
                i_t = i_t + 1
                i_func = i_func + 1
                
                # Define polynomial functions based on the order of minimization (n)
                def create_trajectory_function(coefficients, degree):
                    return lambda t: sum(coefficients[j + i_func * (n_poly + 1)] * t**(degree - j) for j in range(degree + 1))
                
                if self.n == 1:  # Velocity-based trajectory (Linear motion)
                    traj_section_x = create_trajectory_function(self.p_x, 1)
                    traj_section_y = create_trajectory_function(self.p_y, 1)
                    traj_section_z = create_trajectory_function(self.p_z, 1)
                    t_vec_section = np.linspace(self.t_waypoint[i_func], self.t_waypoint[i_func + 1], timesteps)
                    #ax.plot(traj_section_x(self.t_waypoint_section), traj_section_y(self.t_waypoint_section), traj_section_z(self.t_waypoint_section), label=f"Function {i+1}")    
                
                elif self.n == 2:  # Acceleration-based trajectory (Quadratic motion)
                    traj_section_x = create_trajectory_function(self.p_x, 3)
                    traj_section_y = create_trajectory_function(self.p_y, 3)
                    traj_section_z = create_trajectory_function(self.p_z, 3)
                    t_vec_section = np.linspace(self.t_waypoint[i_func], self.t_waypoint[i_func + 1], timesteps)
                    #ax.plot(traj_section_x(self.t_waypoint_section), traj_section_y(self.t_waypoint_section), traj_section_z(self.t_waypoint_section), label=f"Function {i+1}")    
                
                elif self.n == 3:  # Jerk-based trajectory (Quintic motion)
                    traj_section_x = create_trajectory_function(self.p_x, 5)
                    traj_section_y = create_trajectory_function(self.p_y, 5)
                    traj_section_z = create_trajectory_function(self.p_z, 5)
                    t_vec_section = np.linspace(self.t_waypoint[i_func], self.t_waypoint[i_func + 1], timesteps)
                    #ax.plot(traj_section_x(self.t_waypoint_section), traj_section_y(self.t_waypoint_section), traj_section_z(self.t_waypoint_section), label=f"Function {i+1}")                         
                
                elif self.n == 4:  # Snap-based trajectory (Septic motion)
                    traj_section_x = create_trajectory_function(self.p_x, 7)
                    traj_section_y = create_trajectory_function(self.p_y, 7)
                    traj_section_z = create_trajectory_function(self.p_z, 7)
                    
                    x_desired.append(traj_section_x(np.array(t_sec)))
                    y_desired.append(traj_section_y(np.array(t_sec)))
                    z_desired.append(traj_section_z(np.array(t_sec)))
                    
                    # Compute velocity, acceleration based on polynomial derivatives
                    traj_section_vx = lambda t: sum((7-j) * self.p_x[i_func*(n_poly+1) + j] * t**(6-j) for j in range(7))
                    traj_section_vy = lambda t: sum((7-j) * self.p_y[i_func*(n_poly+1) + j] * t**(6-j) for j in range(7))
                    traj_section_vz = lambda t: sum((7-j) * self.p_z[i_func*(n_poly+1) + j] * t**(6-j) for j in range(7))
                    
                    traj_section_ax = lambda t: sum((7-j)*(6-j) * self.p_x[i_func*(n_poly+1) + j] * t**(5-j) for j in range(6))
                    traj_section_ay = lambda t: sum((7-j)*(6-j) * self.p_y[i_func*(n_poly+1) + j] * t**(5-j) for j in range(6))
                    traj_section_az = lambda t: sum((7-j)*(6-j) * self.p_z[i_func*(n_poly+1) + j] * t**(5-j) for j in range(6))
                    
                    # Store computed values
                    vx_desired.append(traj_section_vx(np.array(t_sec)))
                    vy_desired.append(traj_section_vy(np.array(t_sec)))
                    vz_desired.append(traj_section_vz(np.array(t_sec)))
                    
                    ax_vec.append(traj_section_ax(np.array(t_sec)))
                    ay_vec.append(traj_section_ay(np.array(t_sec)))
                    az_vec.append(traj_section_az(np.array(t_sec)))
                    
                    T_optimal_part, phi_optimal_part, theta_optimal_part, time_vec_optimal_part = self._T_attitude_optimal(traj_section_ax, traj_section_ay, traj_section_az, t_sec)
                    T_optimal += T_optimal_part
                    phi_optimal += phi_optimal_part
                    theta_optimal += theta_optimal_part
                    
                t_sec = []
        desired_positions = [np.concatenate(x_desired), np.concatenate(y_desired), np.concatenate(z_desired)]
        desired_velocities = [np.concatenate(vx_desired), np.concatenate(vy_desired), np.concatenate(vz_desired)]
        desired_accelerations = [np.concatenate(ax_vec), np.concatenate(ay_vec), np.concatenate(az_vec)]
        
        phidot_optimal = self._get_eulerdot_difference_method(phi_optimal)
        thetadot_optimal = self._get_eulerdot_difference_method(theta_optimal)
        
        psi = [0]*self.total_steps
        psidot = self._get_eulerdot_difference_method(psi)
        
        ax_actual, ay_actual, az_actual = self._get_actual_accelerations(T_optimal, theta_optimal, phi_optimal)
        actual_accelerations = [ax_actual, ay_actual, az_actual]
        
        actual_vx = self._explicit_method(actual_accelerations[0], desired_velocities[0][0], desired_velocities[0][-1])
        actual_vy = self._explicit_method(actual_accelerations[1], desired_velocities[1][0], desired_velocities[1][-1])
        actual_vz = self._explicit_method(actual_accelerations[2], desired_velocities[2][0], desired_velocities[2][-1])
        actual_velocities = [actual_vx, actual_vy, actual_vz]
        
        actual_x = self._explicit_method(actual_velocities[0], desired_positions[0][0], desired_positions[0][-1])
        actual_y = self._explicit_method(actual_velocities[1], desired_positions[1][0], desired_positions[1][-1])
        actual_z = self._explicit_method(actual_velocities[2], desired_positions[2][0], desired_positions[2][-1])
        actual_positions = [actual_x, actual_y, actual_z]
        
        euler_angles = [phi_optimal, theta_optimal, psi]
        euler_velocities = [phidot_optimal, thetadot_optimal, psidot]
        
        p, q, r, pdot, qdot, rdot, pdot_smooth, qdot_smooth, rdot_smooth = self._get_bodyrates(len(phi_optimal), phi_optimal, phidot_optimal, theta_optimal, thetadot_optimal, psidot)
        body_rates = [p, q, r]
        body_rates_dot_smooth = [pdot_smooth, qdot_smooth, rdot_smooth]
        body_rates_dot = [pdot, qdot, rdot]
        
        return desired_positions, actual_positions, desired_velocities, actual_velocities, desired_accelerations, actual_accelerations, T_optimal, euler_angles, euler_velocities, body_rates, body_rates_dot_smooth, body_rates_dot     
    
    def _obtain_Moments(self, body_rates_dot):
        """
        Compute required moments (torques) from body angular rates and their derivatives.
    
        Parameters:
        ----------
        body_rates_dot : np.ndarray
            Angular rate derivatives.
    
        Returns:
        -------
        M_total : np.ndarray
            Computed moments (torques) over time.
        """
        body_rates = np.array(self.body_rates)  # Ensure it's a NumPy array
        body_rates_dot = np.array(body_rates_dot)  # Ensure it's a NumPy array
        
        
        # Determine number of time steps (columns)
        if body_rates.ndim == 1:  # Single time step case
            return np.matmul(self.I,body_rates_dot) + np.cross(body_rates, np.matmul(self.I,body_rates))
        
        amount = body_rates.shape[1]  # Get number of columns (time steps)
        M_total = np.zeros((3, amount))  # Preallocate a (3, N) array
        
        for i in range(amount):
            omega = body_rates[:, i]
            omega_dot = body_rates_dot[:, i]
            M_total[:, i] = np.matmul(self.I,omega_dot) + np.cross(omega, np.matmul(self.I, omega))

        return M_total
    
    def _compute_individual_Thrust_prop(self, T_total, Moments):
        """
        Compute individual propeller thrusts to produce total thrust and desired roll/pitch moments.
    
        Parameters:
        ----------
        T_total : array-like
            Total thrust values.
        Moments : array-like
            Moments (roll and pitch) to be achieved.
    
        Returns:
        -------
        T_tot_check : np.ndarray
            Recomputed total thrust (sanity check).
        [M_phi_check, M_theta_check] : list
            Recomputed roll and pitch moments (sanity check).
        T_prop : np.ndarray
            Thrust per propeller over time.
        """
        T_total = np.array(T_total)  # Ensure T_total is an array
        Moments = np.array(Moments)  # Ensure Moments is an array
        Moments = Moments[0:2,:]
        
        T_tot_prop = T_total/4
        b = (np.sqrt(2)/2) * self.l_arm
        # Compute roll (phi) and pitch (theta) moment contributions
        M_phi, M_theta = Moments  # Extract moment components
        
        T_phi =  M_phi / (4 * b)  # Roll contributions
        T_theta =  M_theta / (4 * b)  # Pitch contributions
            
        # Compute individual propeller thrusts (vectorized)
        T1_tot = T_tot_prop - T_phi - T_theta
        T2_tot = T_tot_prop + T_phi - T_theta
        T3_tot = T_tot_prop + T_phi + T_theta
        T4_tot = T_tot_prop - T_phi + T_theta
        
        # Verify total thrust per time step
        T_tot_check = T1_tot + T2_tot + T3_tot + T4_tot

        # Verify roll and pitch moments (should match input Moments)
        M_phi_check = (T1_tot * b + T4_tot * b - T3_tot * b - T2_tot * b)
        M_theta_check = -(T1_tot * b + T2_tot * b - T3_tot * b - T4_tot * b)
        #Diverge from thesis***
        #When M_theta_check and M_phi_check are plotted  M_phi mirroed compare to M_phi_smooth

        # Stack thrust values per propeller
        T_prop = np.vstack([T1_tot, T2_tot, T3_tot, T4_tot])
        
        if T_total.ndim == 0:
            print('T1 is: ', round(T1_tot[0],4))
            print('T2 is: ', round(T2_tot[0],4))
            print('T3 is: ', round(T3_tot[0],4))
            print('T4 is: ', round(T4_tot[0],4))
            print('Total thrust is: ', round(T_tot_check[0],4), 'and the predetermined: ', T_total)
            print('Moment phi is: ', round(M_phi_check[0],4), 'and actual moment: ', M_phi[0])
            print('Moment theta is: ', round(M_theta_check[0],4), 'and actual moment: ', M_theta[0])
                
        return T_tot_check, [M_phi_check, M_theta_check], T_prop
    
    
    def _get_rpm_and_Q_net_from_NN(self, V_inflow, inflow_angle, T_prop):
        """
        Compute RPM and net torque from Neural Network based on inflow conditions and thrust.
        
        Parameters:
        ----------
        V_inflow : array-like
            Inflow velocities.
        inflow_angle : array-like
            Disk inflow angles.
        T_prop : array-like
            Thrust per propeller.
        
        Returns:
        -------
        prop_dict : dict
            Dictionary containing RPM, torque (Q), and net torque (Q_net) for each propeller.
        """
        # Convert inputs to NumPy arrays for efficiency
        V_inflow = np.asarray(V_inflow, dtype=np.float64)
        inflow_angle = np.asarray(inflow_angle, dtype=np.float64)
        T_prop = np.asarray(T_prop, dtype=np.float64)
        
        num_props, num_steps = T_prop.shape  # 4 propellers, N time steps
        
        # Preallocate arrays
        rpm_values = np.zeros((num_props, num_steps))
        P_values = np.zeros((num_props, num_steps))
        Q_values = np.zeros((num_props, num_steps))
        Q_net_values = np.zeros(num_steps)
        
        # Process the Neural Network for each propeller and each time step separately
        for i in range(num_props):
            for j in range(num_steps):
                rpm_values[i, j], P_values[i, j] = self.NN.Calculate(V_inflow[j], inflow_angle[j], T_prop[i, j])
        
        # Compute torque for all propellers
        Q_values = P_values / (rpm_values * 0.1407)
        
        # Compute net torque with alternating sign
        Q_net_values = np.sum(((-1) ** np.arange(num_props))[:, None] * Q_values, axis=0)
        
        # Create output dictionary
        prop_dict = {f'rpm_{i+1}': rpm_values[i].tolist() for i in range(num_props)}
        prop_dict.update({f'Q_{i+1}': Q_values[i].tolist() for i in range(num_props)})
        prop_dict['Net Torque'] = Q_net_values.tolist()
                
        return prop_dict
        
    
    def _get_inflow_angle_alfadisc_and_velocity(self, velocities_body, convert):
        """
        Compute inflow angle and magnitude based on body frame velocities.
    
        Parameters:
        ----------
        velocities_body : array-like
            3D velocity components in body frame.
        convert : bool
            If True, convert angle to degrees.
    
        Returns:
        -------
        alfa_disk_total : np.ndarray
            Inflow angles.
        V_magnitude : np.ndarray
            Velocity magnitudes.
        """
        velocities_body = np.array(velocities_body)  # Ensure it's a NumPy array
        V_magnitude = np.linalg.norm(velocities_body, axis=0)  # Compute velocity magnitude

        # Create a mask for small velocities
        small_velocity_mask = V_magnitude <= 1e-6

        # Initialize inflow angle array with default 90 degrees (in radians)
        alfa_disk_total = np.full_like(V_magnitude, 90 * (np.pi / 180))

        # Compute inflow angle where velocity is nonzero
        valid_mask = ~small_velocity_mask
        alfa_disk_total[valid_mask] = np.arccos(velocities_body[2, valid_mask] / V_magnitude[valid_mask])

        # Convert to degrees if required
        if convert:
            alfa_disk_total *= 180 / np.pi

        return alfa_disk_total, V_magnitude
    
    def _rotation_matrix(self, phi, theta, psi):
        """
        Compute rotation matrix from inertial frame to body frame.
    
        Parameters:
        ----------
        phi, theta, psi : float
            Roll, pitch, yaw angles.
    
        Returns:
        -------
        R : np.ndarray
            3x3 rotation matrix.
        """
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)

        return np.array([
            [cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
            [sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, sin_phi * cos_theta],
            [cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, cos_phi * cos_theta]
        ])
    
    def _inertial_to_body_frame(self, velocities, angles):
        """
        Convert velocities from inertial frame to body frame.
    
        Parameters:
        ----------
        velocities : array-like
            Inertial frame velocities.
        angles : array-like
            Euler angles.
    
        Returns:
        -------
        velocities_body : np.ndarray
            Body frame velocities.
        """
        velocities = np.asarray(velocities)
        angles = np.asarray(angles)

        if velocities.ndim == 1:  # Single time step case
            R = self._rotation_matrix(*angles)
            return np.matmul(R, velocities)

        # Multiple time steps case
        steps = velocities.shape[1]
        velocities_body = np.zeros_like(velocities)

        for i in range(steps):
            R = self._rotation_matrix(angles[0, i], angles[1, i], angles[2, i])
            velocities_body[:, i] = np.matmul(R, velocities[:, i]) 

        return velocities_body
    
    def _cal_traj(self):
        """
        Compute the trajectory polynomials and all physical quantities for the drone motion.
        """
        self.p_x, self.p_y, self.p_z = self._trajectory_algorithm()
        self.desired_positions, self.actual_positions, self.desired_velocities, self.actual_velocities, self.desired_accelerations, self.actual_accelerations, self.T_optimal, self.euler_angles, self.euler_velocities, self.body_rates, self.body_rates_dot_smooth, self.body_rates_dot = self._get_trajectory()
        
        print('Trajectories are created')
    
    def get_desired_positions(self):
        """
        Get the desired 3D positions along the trajectory.
    
        Returns:
        -------
        desired_positions : list
            x, y, z desired positions over time.
        """
        return self.desired_positions
    
    
    def calc_individual_Thrust(self):
        """
        Calculate and store individual thrusts per propeller if not already computed.
        """
        if self.individual_thrust == False:
            self.Moments_smoothed = self._obtain_Moments(self.body_rates_dot_smooth)
            self.T_totcheck, self.M_check, self.T_propeller = self._compute_individual_Thrust_prop(self.T_optimal, self.Moments_smoothed)
            self.individual_thrust = True
        else:
            pass
    def _calc_rpm(self):
        """
        Compute propeller RPMs based on calculated thrusts and inflow conditions.
        """
        self.calc_individual_Thrust()
        actual_velocities_bodyframe = self._inertial_to_body_frame(self.actual_velocities, self.euler_angles)
        alfa_disk_rad, V_inflow = self._get_inflow_angle_alfadisc_and_velocity(actual_velocities_bodyframe, False)
        self.propeller_dictionary = self._get_rpm_and_Q_net_from_NN(V_inflow, alfa_disk_rad, self.T_propeller)
        
        
    def add_noise_thrust(self):
        """
        Add random Gaussian noise to individual thrust signals.
    
        Returns:
        -------
        noised_signal : np.ndarray
            Thrust signals with added noise.
        """
        self.calc_individual_Thrust()
        noised_signal = []
        for i in range(4):
            noisy_signal = self.T_propeller[i][:] + np.random.normal(0, 1, size=self.T_propeller[i].shape)  # Add noise
            noised_signal.append(noisy_signal)
        return np.array(noised_signal)
    
    def add_noise_rpm(self):
        """
        Add random Gaussian noise to individual propeller RPM signals.
    
        Returns:
        -------
        noised_signal : np.ndarray
            RPM signals with added noise.
        """
        self._calc_rpm()
        noised_signal = []
        
        #print(self.propeller_dictionary['rpm_1'])
        columns=['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']
        df = pd.DataFrame(zip(self.propeller_dictionary['rpm_1'],
        self.propeller_dictionary['rpm_2'],
        self.propeller_dictionary['rpm_3'],
        self.propeller_dictionary['rpm_4']),
        columns=columns)
        #print(df)
        #print(df[columns[0]])
        
        for i in range(4):
            noisy_signal = df[columns[i]] + 0.02*np.max(df[columns[i]])*np.random.normal(0, 3, size=df[columns[i]].shape)  # Add noise
            noised_signal.append(noisy_signal)
        return np.array(noised_signal)
        
    def plot_noised_thrust(self,d=0.09,freq=60):
        """
        Plot noisy and filtered thrust signals for all propellers.
    
        Parameters:
        ----------
        d : float
            LOWESS smoothing fraction.
        freq : float
            Cutoff frequency for optional Butterworth filter.
        """
        noised_prop = self.add_noise_thrust()
        filtered_prop1 = self.filter_noise(noised_prop[0][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_prop2 = self.filter_noise(noised_prop[1][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_prop3 = self.filter_noise(noised_prop[2][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_prop4 = self.filter_noise(noised_prop[3][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        
        plt.figure()
        #plt.plot(self.t_vec,noised_prop[0][:],label='noised',alpha=0.5)
        plt.plot(self.t_vec,filtered_prop1,label='filtered prop1')
        plt.plot(self.t_vec,self.T_propeller[0][:],label='prop1',linestyle=':')
        plt.plot(self.t_vec,filtered_prop2,label='filtered prop2')
        plt.plot(self.t_vec,self.T_propeller[1][:],label='prop2',linestyle=':')
        plt.plot(self.t_vec,filtered_prop3,label='filtered prop3')
        plt.plot(self.t_vec,self.T_propeller[2][:],label='prop3',linestyle=':')
        plt.plot(self.t_vec,filtered_prop4,label='filtered prop4')
        plt.plot(self.t_vec,self.T_propeller[3][:],label='prop4',linestyle=':')
        plt.legend()
        
        plt.xlabel('time [s]')
        plt.ylabel('Force [N]')

        print(np.corrcoef(self.T_propeller[0][:], filtered_prop1)[0, 1])
        print(np.corrcoef(self.T_propeller[1][:], filtered_prop2)[0, 1])
        print(np.corrcoef(self.T_propeller[2][:], filtered_prop3)[0, 1])
        print(np.corrcoef(self.T_propeller[3][:], filtered_prop4)[0, 1])
    
    def plot_noised_rpm(self,d=0.09,freq=60):
        """
        Plot noisy and filtered RPM signals for a selected propeller.
    
        Parameters:
        ----------
        d : float
            LOWESS smoothing fraction.
        freq : float
            Cutoff frequency for optional Butterworth filter.
        """
        self._calc_rpm()
        
        noised_rpm = self.add_noise_rpm()
        filtered_rpm1 = self.filter_noise(noised_rpm[0][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_rpm2 = self.filter_noise(noised_rpm[1][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_rpm3 = self.filter_noise(noised_rpm[2][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        filtered_rpm4 = self.filter_noise(noised_rpm[3][:], d=d,cutoff_freq=freq,sampling_rate=1000)
        
        
        plt.figure()
        plt.plot(self.t_vec,noised_rpm[0][:],alpha=0.5, label='noised')
        plt.plot(self.t_vec,filtered_rpm1,label='filtered prop1')
        plt.plot(self.t_vec,self.propeller_dictionary['rpm_1'][:],label='prop1',linestyle=':')
        
        #plt.plot(self.t_vec,filtered_rpm2,label='filtered prop2')
        #plt.plot(self.t_vec,self.propeller_dictionary['rpm_2'][:],label='prop2',linestyle=':')
        #plt.plot(self.t_vec,filtered_rpm3,label='filtered prop3')
        #plt.plot(self.t_vec,self.propeller_dictionary['rpm_3'][:],label='prop3',linestyle=':')
        #plt.plot(self.t_vec,filtered_rpm4,label='filtered prop4')
        #plt.plot(self.t_vec,self.propeller_dictionary['rpm_4'][:],label='prop4',linestyle=':')
        plt.legend()
        
        plt.xlabel('time [s]')
        plt.ylabel('rpm')
        plt.title('RPM after filtering compared to computed RPM')

        print(np.corrcoef(self.propeller_dictionary['rpm_1'][:], filtered_rpm1)[0, 1])
        print(np.corrcoef(self.propeller_dictionary['rpm_2'][:], filtered_rpm2)[0, 1])
        print(np.corrcoef(self.propeller_dictionary['rpm_3'][:], filtered_rpm3)[0, 1])
        print(np.corrcoef(self.propeller_dictionary['rpm_4'][:], filtered_rpm4)[0, 1])
    
    def plot_individual_rpm(self):
        """
        Plot the RPM evolution of each propeller over time.
        """
        self._calc_rpm()
        
        plt.figure()
        plt.title('RPM for each propeller',fontsize = 36)
        plt.plot(self.t_vec, self.propeller_dictionary['rpm_1'][:], label=r'Prop$_1$ FR', marker='.', linestyle='None', markerfacecolor='r', markeredgecolor='None')
        plt.plot(self.t_vec, self.propeller_dictionary['rpm_2'][:], label=r'Prop$_2$ FL', marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='b')
        plt.plot(self.t_vec, self.propeller_dictionary['rpm_3'][:], label=r'Prop$_3$ RL', marker='s', linestyle='None', markerfacecolor='None', markeredgecolor='green')
        plt.plot(self.t_vec, self.propeller_dictionary['rpm_4'][:], label=r'Prop$_4$ RR', marker='d', linestyle='None', markerfacecolor='purple', markeredgecolor='None')
        plt.xlabel('time [s]',fontsize = 28)
        plt.ylabel('Rotational speed [RPM]',fontsize = 28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.legend(fontsize = 21)
        
    def filter_noise(self, vector, d, cutoff_freq=None, sampling_rate=None):
        """
        Filter a signal using optional Butterworth filter + LOWESS smoothing.
    
        Parameters:
        ----------
        vector : array-like
            Data to be smoothed.
        d : float
            LOWESS smoothing fraction.
        cutoff_freq : float, optional
            Butterworth cutoff frequency.
        sampling_rate : float, optional
            Data sampling rate.
    
        Returns:
        -------
        smoothed_vector : np.ndarray
            Smoothed signal.
        """
        if cutoff_freq and sampling_rate:
            nyquist = 0.5 * sampling_rate  # Nyquist frequency
            normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
            b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)  # Butterworth filter
            vector = filtfilt(b, a, vector)  # Apply zero-phase filter

        # Apply LOWESS smoothing
        lowess = sm.nonparametric.lowess(vector, self.t_vec, frac=d)
        smoothed_vector = lowess[:, 1]
        return smoothed_vector

    def plot_trajectory(self,data):
        #desired_positions, actual_positions, desired_velocities, actual_velocities, desired_accelerations, actual_accelerations, T_optimal, euler_angles, euler_velocities, body_rates, body_rates_dot_smooth, body_rates_dot = self._get_trajectory()
        """
        Plot the 3D desired trajectory alongside experimental data (e.g., Qualisys).
    
        Parameters:
        ----------
        data : array-like
            Measured 3D positions to overlay.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.desired_positions[0], self.desired_positions[1], self.desired_positions[2], color='r', marker='.', s=30, label="trajectory")
        ax.scatter(self.x_waypoint, self.y_waypoint, self.z_waypoint, color = 'k', marker = 's', s = 80, label = 'waypoints')
        ax.set_xlabel('x [m]',fontsize = 22, labelpad=15)
        ax.set_ylabel('y [m]',fontsize = 22, labelpad=15)
        ax.set_zlabel('z [m]',fontsize = 22, labelpad=15)
        ax.tick_params(axis='x', which='major', labelsize=18)  # X-axis tick font size
        ax.tick_params(axis='y', which='major', labelsize=18)  # Y-axis tick font size
        ax.tick_params(axis='z', which='major', labelsize=18)  # Z-axis tick font size
        ax.plot(data[1175:1500,(7*3)+2],data[1175:1500,(7*3)+3],data[1175:1500,(7*3)+4], label = 'Qualisys')
        plt.legend(fontsize = 18)
        ax.set_title('Desired Trajectory',fontsize = 26)
        plt.show()
        
    def plot_trajectory_2D(self):
        """
        Plot desired x, y, z positions separately over time.
        """
        plt.figure()
        plt.title('Desired Positions')
        plt.plot(self.t_vec, self.desired_positions[0], label = 'x', color = 'r')
        plt.plot(self.t_vec, self.desired_positions[1], label = 'y', color = 'green', linestyle = 'dashed')
        plt.plot(self.t_vec, self.desired_positions[2], label = 'z', color = 'c')
        plt.plot(self.t_waypoint, self.x_waypoint, marker = 's', linestyle = 'None', markerfacecolor = 'gold', markeredgecolor = 'Orange', label = 'x waypoint', ms = 12)
        plt.plot(self.t_waypoint, self.y_waypoint, marker = 'x', linestyle = 'None', markerfacecolor = 'none', markeredgecolor = 'black', label = 'y waypoint', ms = 14)
        plt.plot(self.t_waypoint, self.z_waypoint, marker = 'o', linestyle = 'None', markerfacecolor = 'blue', markeredgecolor = 'None', label = 'z waypoint', ms = 12)
        plt.grid()
        plt.legend()
        plt.xlabel('time [s]')
        plt.ylabel('position [m]')
        
    def plot_desired_velocities(self):
        """
        Plot desired velocity components over time.
        """
        plt.figure()
        plt.title('Desired Velocity')
        plt.plot(self.t_vec, self.desired_velocities[1], label=r'$v_{y,\text{desired}}$', color='blue')
        plt.plot(self.t_vec, self.desired_velocities[0], label=r'$v_{x,\text{desired}}$', color='r')
        plt.plot(self.t_vec, self.desired_velocities[2], label=r'$v_{z,\text{desired}}$', color='gold', linestyle='dashed')
        plt.legend(loc = 'upper left')
        plt.xlabel('time [s]')
        plt.ylabel(r'velocity $[\frac{m}{s}]$') 
        
    def plot_desired_accelerations(self):
        """
        Plot desired acceleration components over time.
        """
        plt.figure()
        plt.title('Desired accelerations',fontsize = 20)
        plt.plot(self.t_vec, self.desired_accelerations[1], label=r'$a_{y,\text{desired}}$', color='blue')
        plt.plot(self.t_vec, self.desired_accelerations[0], label=r'$a_{x,\text{desired}}$', color='r')
        plt.plot(self.t_vec, self.desired_accelerations[2], label=r'$a_{z,\text{desired}}$', color='gold', linestyle='dashed')
        plt.grid()
        plt.legend()
        plt.xlabel('time [s]',fontsize = 16)
        plt.ylabel(r'acceleration $[\frac{m}{s^2}]$',fontsize = 16)
        
    def plot_individual_Thrust(self):
        """
        Plot thrust profiles for each propeller over time.
        """
        plt.figure()
        plt.title('Thrust for each propeller')
        plt.plot(self.t_vec, self.T_propeller[0][:], label=r'$T_1$', marker='.', linestyle='None', markerfacecolor='r', markeredgecolor='None')
        plt.plot(self.t_vec, self.T_propeller[1][:], label=r'$T_2$', marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='b')
        plt.plot(self.t_vec, self.T_propeller[2][:], label=r'$T_3$', marker='s', linestyle='None', markerfacecolor='None', markeredgecolor='green')
        plt.plot(self.t_vec, self.T_propeller[3][:], label=r'$T_4$', marker='d', linestyle='None', markerfacecolor='purple', markeredgecolor='None')
        plt.xlabel('time [s]')
        plt.ylabel('Force [N]')
        plt.grid()
        plt.legend()

#%% Class Data_rpm
class Data_rpm:
    """
    Class to load, filter, and plot RPM data from CSV files.
    
    Attributes:
    ----------
    paths : list
        List of file paths to RPM CSV files.
    fs : int
        Sampling frequency of the RPM data [Hz].
    """
    def __init__(self, paths):
        """
        Initialize the Data_rpm object.
    
        Parameters:
        ----------
        paths : list
            List of paths to the CSV files containing RPM data.
        """
        self.paths = paths
        self.fs = 100
    
    def load_rpm_data(self):
        """
        Load all RPM CSV files and organize them into a dictionary.
    
        Returns:
        -------
        data : dict
            Dictionary with labels as keys and corresponding RPM DataFrames as values.
            Each DataFrame has 'Time' and 'RPM' columns.
        """
        data = {}
        for path in self.paths:
            filename = os.path.basename(path)
            label = filename.split("-hover_")[-1].replace(".csv", "")
            
            df = pd.read_csv(path , names=["Time", "RPM"])
            data[label] = df
        return data
    
    def plot_rpm_data(self, data_dict):
        """
        Plot full RPM time series for each dataset.
    
        Parameters:
        ----------
        data_dict : dict
            Dictionary of DataFrames containing 'Time' and 'RPM' columns to plot.
        """
        plt.figure(figsize=(12, 8))

        for label, df in data_dict.items():
            if 'Time' in df.columns and 'RPM' in df.columns:
                plt.plot(df['Time'], df['RPM'], label=label)
            else:
                print(f"Warning: {label} missing 'Time' or 'RPM' columns.")

        plt.xlabel('Time [s]', fontsize = 28)
        plt.ylabel('RPM', fontsize = 28)
        plt.title('Motor RPM vs Time', fontsize = 36)
        plt.legend(fontsize = 21)
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()
        
    def plot_rpm_data_small_part(self, data_dict, index):
        """
        Plot a specific part of the RPM time series.
    
        Parameters:
        ----------
        data_dict : dict
            Dictionary of DataFrames containing 'Time' and 'RPM' columns to plot.
        index : list or tuple
            Two-element list/tuple [start_index, end_index] to slice the time series.
        """
        plt.figure(figsize=(12, 8))

        for label, df in data_dict.items():
            if 'Time' in df.columns and 'RPM' in df.columns:
                plt.plot(df['Time'][index[0]:index[1]], df['RPM'][index[0]:index[1]], label=label)
            else:
                print(f"Warning: {label} missing 'Time' or 'RPM' columns.")

        plt.xlabel('Time [s]', fontsize = 28)
        plt.ylabel('RPM', fontsize = 28)
        plt.title('Motor RPM vs Time', fontsize = 36)
        plt.legend(fontsize = 21)
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()
    
    def lowess(self, data_vec, d, cutoff_freq=None, sampling_rate=None):
        """
        Apply optional Butterworth filter followed by LOWESS smoothing to RPM data.
    
        Parameters:
        ----------
        data_vec : DataFrame
            DataFrame containing 'Time' and 'RPM' columns.
        d : float
            LOWESS fraction parameter (defines smoothing strength).
        cutoff_freq : float, optional
            Cutoff frequency for optional Butterworth filter (Hz).
        sampling_rate : float, optional
            Sampling rate of the signal (Hz).
    
        Returns:
        -------
        smoothed_vector : np.ndarray
            Smoothed RPM values.
        """
        time_vector = data_vec["Time"].values
        data = data_vec["RPM"].values
        if cutoff_freq and sampling_rate:
            nyquist = 0.5 * sampling_rate  # Nyquist frequency
            normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
            b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)  # Butterworth filter
            data = filtfilt(b, a, data)  # Apply zero-phase filter

        # Apply LOWESS smoothing
        lowess = sm.nonparametric.lowess(data, time_vector, frac=d)
        smoothed_vector = lowess[:, 1]

        return smoothed_vector
    
    def load_and_filter_rpm_data(self, d=0.02, cutoff_freq=None):
        """
        Load RPM data and apply filtering (optional Butterworth + LOWESS).
    
        Parameters:
        ----------
        d : float, optional
            LOWESS fraction parameter for smoothing (default is 0.02).
        cutoff_freq : float, optional
            Cutoff frequency for optional Butterworth filter (Hz).
    
        Returns:
        -------
        filtered_data : dict
            Dictionary containing filtered RPM DataFrames with 'Time' and 'RPM' columns.
        """
        raw_data = self.load_rpm_data()
        self.filtered_data = {}
        for label, df in raw_data.items():
            smoothed_rpm = self.lowess(df, d, cutoff_freq=cutoff_freq, sampling_rate=self.fs)
            filtered_df = pd.DataFrame({
                "Time": df["Time"],
                "RPM": smoothed_rpm
            })
            self.filtered_data[label] = filtered_df
        return self.filtered_data
        
        
    
#%%
# --- Load flown trajectory data (from Qualisys/experimental recording) ---
hov_path4 = r'D:\VKI\Project\Test flight\20250409\Flights\export-tsv\20250904-1558-04-hover.tsv'
df = pd.read_csv(hov_path4, sep='\t', skiprows = 12, header=None)/1000
data_h4 = df.to_numpy()

# --- Define file paths for propeller RPM measurements ---
folder = r"D:\VKI\Project\Test flight\20250409\Flights\hover\drone\20250409-1558-04-hover_"  # Change this path if needed
patterns = np.array(["RPM_aft_left", "RPM_aft_right", "RPM_front_left", "RPM_front_right"])
file_paths = [folder + pattern + ".csv" for pattern in patterns]

# --- Load and filter RPM data ---
rpm = Data_rpm(file_paths)
data_rpm = rpm.load_rpm_data()
indices = [78,102]
data_rpm_filtered = rpm.load_and_filter_rpm_data(d=0.02, cutoff_freq=40)
rpm.plot_rpm_data_small_part(data_rpm_filtered,indices)

#rpm.plot_rpm_data(data_rpm)
n = 4  # velocity is 1st order, acceleration is 2nd order, jerk is 3rd order, snap is 4th order

# Waypoints obtained from the flown trajectory (position in meters)
x_waypoint = [0.078582,  0.066121,  0.054091,  0.040123,  0.025547,  
              0.011525, -0.002027, -0.013824, -0.024317, -0.034134]  #[0, 8, 0, 8]
y_waypoint = [-1.041588, -1.065625, -1.06986 , -1.052896, -1.016325, 
              -0.98067, -0.971433, -0.983934, -1.015618, -1.034557]   #[0, 4, 0, 4]
z_waypoint = [0.496714, 0.639687, 0.809938, 0.987289, 1.166493, 
              1.351557, 1.549719, 1.753027, 1.946396, 2.072185]  #[10, 10, 10, 10]
# Time at each waypoint (seconds)
t_waypoint = [9.79167, 10.09167, 10.39167, 10.69167, 10.99167, 
              11.29167, 11.59167, 11.89167, 12.19167, 12.49167]   #[0, 5, 9, 12]
total_steps = 500  # steps between each waypoint

# --- Define Drone Parameters ---
m = 1.626  # Mass of the quadcopter in kg
l_arm = 0.26608
g = 9.81  # Gravity in m/s^2
I = np.array([[0.0213,  9.0792e-7, 1.7882e-7],
     [9.0792e-7, 0.0207, 1.0837e-4],
     [1.7882e-7, 1.0837e-4, 0.0325]])

# --- Create Drone Object ---
drone = Drone(m, l_arm, I, n_rotor=4)

# --- Generate Trajectory for Drone 1 (standard mass) ---
Traj_1 = Trajectory(x_waypoint, y_waypoint, z_waypoint, t_waypoint, "Traj1", total_steps, drone)

# --- Plotting ---
Traj_1.plot_trajectory(data_h4) # Plot generated trajectory vs real measured data
Traj_1.plot_individual_rpm() # Plot computed individual propeller RPMs

# =============================================================================
# drone_2 = Drone(m*1.2, l_arm, I, n_rotor=4)
# Traj_2 = Trajectory(x_waypoint, y_waypoint, z_waypoint, t_waypoint, "Traj2", total_steps, drone_2)
# #Traj_2.plot_trajectory(data_h4)
# Traj_2.plot_individual_rpm()
# =============================================================================
#%% Cost function analysis

def rpm_cost_function(simulated_rpm_dict, experimental_rpm_array):
    """
    Cost function to minimize the distance between the simulated and experimental RPMs.

    Parameters:
    -----------
    simulated_rpm_dict : dict
        Dictionary containing simulated RPMs with keys 'rpm_1', 'rpm_2', 'rpm_3', 'rpm_4'.
        Each key maps to a numpy array of RPM values over time.
        
    experimental_rpm_array : np.ndarray
        Experimental RPMs, shape (4, N) where each row corresponds to a propeller (same order).
        
    Returns:
    --------
    cost : float
        Sum of squared errors between simulated and experimental RPMs.
    """
    # Stack the simulated RPMs into a 4 x N array
    simulated_rpm_array = np.vstack([
        simulated_rpm_dict['rpm_1'],
        simulated_rpm_dict['rpm_2'],
        simulated_rpm_dict['rpm_3'],
        simulated_rpm_dict['rpm_4']
    ])
    
    # Ensure both arrays have the same shape
    if simulated_rpm_array.shape != experimental_rpm_array.shape:
        raise ValueError(f"Shape mismatch: simulated {simulated_rpm_array.shape} vs experimental {experimental_rpm_array.shape}")

    # Compute the element-wise difference
    diff = simulated_rpm_array - experimental_rpm_array

    # Return the sum of squared differences
    cost = np.sum(diff**2)
    return cost

def simulate_with_parameters(Traj_obj, scale_factor):
    """
    Recompute simulated RPMs with updated parameters.

    Parameters:
    -----------
    Traj_obj : Trajectory
        Your trajectory object (like Traj_1).
        
    scale_factor : float
        Factor to scale thrusts (or another model parameter).
        
    Returns:
    --------
    simulated_rpm_dict : dict
        Updated simulated RPMs.
    """
    # Apply scaling to thrust
    T_prop_scaled = Traj_obj.T_propeller * scale_factor  # Scale the thrust
    actual_velocities_bodyframe = Traj_obj._inertial_to_body_frame(Traj_obj.actual_velocities, Traj_obj.euler_angles)
    alfa_disk_rad, V_inflow = Traj_obj._get_inflow_angle_alfadisc_and_velocity(actual_velocities_bodyframe, False)
    
    # Get RPMs with updated thrust
    simulated_rpm_dict = Traj_obj._get_rpm_and_Q_net_from_NN(V_inflow, alfa_disk_rad, T_prop_scaled)
    
    return simulated_rpm_dict

def optimization_cost(scale_factor, Traj_obj, experimental_rpm_array):
    """
    Wrapper for the optimizer.
    """
    simulated_rpm_dict = simulate_with_parameters(Traj_obj, scale_factor)
    return rpm_cost_function(simulated_rpm_dict, experimental_rpm_array)

def plot_rpm_comparison(Traj_obj, experimental_rpm_array, optimal_scale):
    """
    Plot simulated vs experimental RPM before and after optimization.
    """
    
    
    # Simulated BEFORE optimization
    simulated_before = np.vstack([
    Traj_obj.propeller_dictionary['rpm_1'],
    Traj_obj.propeller_dictionary['rpm_2'],
    Traj_obj.propeller_dictionary['rpm_3'],
    Traj_obj.propeller_dictionary['rpm_4']
    ])
    
    # Simulated AFTER optimization
    simulated_after_dict = simulate_with_parameters(Traj_obj, optimal_scale)
    simulated_after = np.vstack([
    simulated_after_dict['rpm_1'],
    simulated_after_dict['rpm_2'],
    simulated_after_dict['rpm_3'],
    simulated_after_dict['rpm_4']
    ])
    
    time = Traj_obj.t_vec  # Time vector
    
    prop_labels = ['Propeller 1 (FR)', 'Propeller 2 (FL)', 'Propeller 3 (RL)', 'Propeller 4 (RR)']
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    for i in range(4):
        axs[i].plot(time, experimental_rpm_array[i, :], label='Experimental', color='black', linestyle='dashed')
        axs[i].plot(time, simulated_before[i, :], label='Simulated (before)', color='red', alpha=0.7)
        axs[i].plot(time, simulated_after[i, :], label='Simulated (after)', color='blue', alpha=0.7)
        
        axs[i].set_ylabel('RPM')
        axs[i].set_title(prop_labels[i])
        axs[i].grid(True)
        axs[i].legend()
        
    axs[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    
    
resampled_experimental_rpm_array = np.vstack([
    resample(data_rpm_filtered['RPM_front_right']['RPM'][78:102].values, Traj_1.t_vec.size),
    resample(data_rpm_filtered['RPM_front_left']['RPM'][78:102].values, Traj_1.t_vec.size),
    resample(data_rpm_filtered['RPM_aft_left']['RPM'][78:102].values, Traj_1.t_vec.size),
    resample(data_rpm_filtered['RPM_aft_right']['RPM'][78:102].values, Traj_1.t_vec.size)
])

Traj_1.calc_individual_Thrust()
Traj_1._calc_rpm()

# Optimize
result = minimize(optimization_cost, 
                  x0=[1.0], 
                  args=(Traj_1, resampled_experimental_rpm_array), 
                  bounds=[(0.5, 5)], 
                  method='L-BFGS-B')

optimal_scale = result.x[0]
print(f"Optimal scale factor found: {optimal_scale:.5f}")
print(f"Final cost: {result.fun:.5f}")

# Plot
plot_rpm_comparison(Traj_1, resampled_experimental_rpm_array, optimal_scale)    