"""
Forward propogation for sensor fusion model. 
IMU integration
"""
#IMU used is BNO055, Lidar is RPLidar A1M8, Camera is PiCamZero
from dataclasses import dataclass
from utils.so3_rotation import exp, skew
import numpy as np
#Kalman filter --- Prediction, Update/Correction

@dataclass
class State:
    R:np.ndarray  #Orientation (3,3)
    p:np.ndarray  #Translation (3,)
    v:np.ndarray  #Velocity (3,)
    bg:np.ndarray #Gyro Bias (3,)
    ba:np.ndarray #Accel Bias (3,)
    g:np.ndarray #Gravity (3,)

class ESIKFStateEstimator:
    def __init__(self):
        self.P = np.eye(18) # process covariance matrix
        self.Q = np.eye(18) # process noise covariance
        self.R = np.eye(3) # measurement matrix
        dt = 0.01  # IMU is at 100Hz, so time step is 0.01 seconds
        self.state = State(
            R = np.eye(3,3),
            p = np.zeros(3),
            v = np.zeros(3),
            bg = np.zeros(3),
            ba = np.zeros(3),
            g = np.array([0,0,-9.81]),
        )
    def compute_jacobian(self, x_prev, u, dt):
        #Computing Jacobian of the state model wrt the state error delta_x
        # dt = dt
        A = np.eye(18)
        a_I = u[1] - x_prev.ba
        w_I = u[0] - x_prev.bg
        a_skew = skew(a_I)
        w_skew = skew(w_I)

        # 1. Rotation error evolution
        # Approximation of expm(-w_skew * dt)
        A[0:3, 0:3] = np.eye(3) - (w_skew * dt) 
        
        # 2. Gyro bias impact on rotation error
        A[0:3, 9:12] = -np.eye(3) * dt

        A[3:6, 0:3] = -0.5 * x_prev.R @ a_skew * dt**2
        A[3:6, 6:9] = np.eye(3) * dt
        A[3:6, 12:15] = -0.5 * x_prev.R * dt**2
        A[3:6, 15:18] = np.eye(3) * 0.5 * dt**2

        A[6:9, 0:3] = -x_prev.R @ a_skew * dt
        A[6:9, 12:15] = -x_prev.R * dt
        A[6:9, 15:18] = np.eye(3) * dt

        return A
    
    def predict(self, u: list, dt):#u = [w,a]
        x_prev = self.state

        A = self.compute_jacobian(x_prev, u, dt)#compute the jacobian of the state transition model

        # Implement the prediction step of the Kalman filter here
        ang_act = u[0] - x_prev.bg #wm = wa + bg + ng -> wa = wm - bg - ng
        accel = (x_prev.R @ (u[1] - x_prev.ba) - x_prev.g)

        delta_theta = ang_act * dt
        delta_R = exp(delta_theta)
        self.state.R = self.state.R @ delta_R  #del_theta = w*del_t --> converted to proper SO(3) before adding to the rotation matrix(SO(3))
        self.state.p += (self.state.v * dt) + (0.5 * accel * dt * dt) 
        self.state.v += accel * dt

        #Covariance update
        self.P = A @ self.P @ A.T + self.Q
        return self.state, self.P

    def update(self):
        # Implement the update/correction step of the Kalman filter here
        # After LIDAR data arrives and it is backward propogated
        error_meas = self.R #measurement noise covariance
        # gain = error_pred / (error_pred+ error_meas)
        pass

