import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
from matplotlib.patches import Ellipse

"""
Author: Andrew Cunningham (unless otherwise noted in specific parts)
Description: A simple Kalman filter for robot localization.  This file contains
classes for a simple planar, holonomic robot with linear transition and observation
models. This was implemented for RPI's PGM course final project
"""

# Adapted from http://greg.czerniak.info/guides/kalman1/
class KalmanFilterLinear:
    def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
        self.A = _A                      # State transition matrix.
        self.B = _B                      # Control matrix.
        self.H = _H                      # Observation matrix.
        self.current_state_estimate = _x # Initial state estimate.
        self.current_prob_estimate = _P  # Initial covariance estimate.
        self.Q = _Q                      # Estimated error in process.
        self.R = _R                      # Estimated error in measurements.
        
    def get_current_state(self):
        return self.current_state_estimate
        
    def sense_update(self, measurement_vector):
        """ Update Kalman filter if sensing information is received """
       #--------------------------Observation step-----------------------------
        innovation = measurement_vector - np.dot(self.H, self.current_state_estimate)
        innovation_covariance = np.dot(np.dot(self.H, self.current_prob_estimate), np.transpose(self.H)) + self.R
        #-----------------------------Update step-------------------------------
        kalman_gain = np.dot(np.dot(self.current_prob_estimate, np.transpose(self.H)), np.linalg.inv(innovation_covariance))
        self.current_state_estimate = self.current_state_estimate + np.dot(kalman_gain, innovation)
        # We need the size of the matrix so we can make an identity matrix.
        size = self.current_prob_estimate.shape[0]
        # eye(n) = nxn identity matrix.
        self.current_prob_estimate = np.dot((np.eye(size)-np.dot(kalman_gain,self.H)), self.current_prob_estimate)
        
    def step(self, control_vector):
        """ Step without sensing """
        self.current_state_estimate = np.dot(self.A, self.current_state_estimate) + np.dot(self.B, control_vector)
        self.current_prob_estimate = np.dot(np.dot(self.A, self.current_prob_estimate), np.transpose(self.A)) + self.Q

        
class PlanarFeature:
    """ Feature for a planar robot to localize with, think of it as a local GPS"""
    def __init__(self,position,  _R):
        self.position = position.reshape([2,1])
        self._C = np.eye(2)
        self._R = _R
        return

class PlanarRobot:
    """ Robot contains kalman filter and controls """
    def __init__(self, starting_position, kfilter, feature_list, noise_cov, dist_thresh):
        self.kfilter = kfilter 
        self.feature_list = feature_list         # effectively a map
        self.true_position = starting_position   
        
        self.noise_mean = np.array([0,0])
        self.noise_cov = noise_cov
        self.dist_thresh = dist_thresh
        
        self.p_state_history = []
        self.true_state_history = []
        self.prob_history = []
        
    def command_robot(self, u):
        """ command robot's linear velocity (it is a holonomic vehicle) """
        # step true position of the robot
        noise = np.random.multivariate_normal(self.noise_mean, self.noise_cov, 1)
        noise = noise.reshape([2,1])
        self.true_position += u + noise
        self.kfilter.step(u)
        
        # check to see if any sensors get readings
        self.sense()        
        
        self.true_state_history.append(copy.deepcopy(self.true_position))
        self.p_state_history.append(copy.deepcopy(self.kfilter.current_state_estimate))
        self.prob_history.append(copy.deepcopy(self.kfilter.current_prob_estimate))
        
    def sense(self):
        """ Return measurement of distance if in range of landmark """
        for feature in self.feature_list:
            if np.linalg.norm(self.true_position - feature.position) < self.dist_thresh:
                sense_noise_mean = np.array([0,0])
                sense_noise_cov = feature._R
                clean_measurement = self.true_position
                noise = np.random.multivariate_normal(sense_noise_mean, sense_noise_cov, 1)
                noise = noise.reshape([2,1])
                noisy_measurement = clean_measurement + noise
                self.kfilter.sense_update(noisy_measurement)
                # assume only one feature can be read at a time
                return         

class RobotAnimator:
    def __init__(self, robot):
        self.robot_obj = robot
        
        self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(7, 6.5)
        
        self.ax = plt.axes(xlim=(0, 110), ylim=(0, 110))
        feature = self.robot_obj.feature_list[0]
        self.landmark = plt.Circle((feature.position[0], feature.position[1]), self.robot_obj.dist_thresh, fc='g', label='Landmark 1', alpha=.2)
        self.robot = plt.Circle((-10, -10), 0.75, fc='r', label='Actual position')
        self.uncertainty = Ellipse(xy=np.array([50,50]), width=2, height=2, alpha=.3, label='Covariance of estimate')

    def init_animation(self):
        self.robot.center = (-1000, -1000)
        self.uncertainty.center = (-1000, -1000)
        self.ax.add_patch(self.robot)
        self.ax.add_patch(self.uncertainty)
        self.ax.add_patch(self.landmark)
        
        return self.robot, self.uncertainty, 
        
    def animate(self, i):
        state = self.robot_obj.true_state_history[i]
        uncertainty_center = self.robot_obj.p_state_history[i]
        uncertainty_matrix = self.robot_obj.prob_history[i]
        self.robot.center = (state[0], state[1])
        self.uncertainty.center = (uncertainty_center[0], uncertainty_center[1])
        self.uncertainty.width = uncertainty_matrix[0,0]*5
        self.uncertainty.height = uncertainty_matrix[1,1]*5
        return self.uncertainty, self.robot,