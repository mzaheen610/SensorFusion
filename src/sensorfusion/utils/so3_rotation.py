"""
Implement the Rodrigues formula based rotation update for forward propogation.
"""
import numpy as np

def skew(var):
    x = var[0]
    y = var[1]
    z = var[2]
    skew_var = np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])
    return skew_var

def exp(delta_theta):
    #Convert the IMU based angles to the incremental rotation matrix - delta_R 
    # (so(3)->SO(3))

    dx = delta_theta[0]
    dy = delta_theta[1]
    dz = delta_theta[2]
    sigma = np.linalg.norm(delta_theta)
    if sigma > 0.0000001:
        skew_delta = np.array([
            [0, -dz, dy],
            [dz, 0, -dx],
            [-dy, dx, 0]
        ])
        K = skew_delta / sigma
        #Rodrigues formula for incremental rotation
        delta_R = (np.eye(3) 
                + np.sin(sigma) * K 
                + ((1-np.cos(sigma)) * (K @ K))
                )
        return delta_R
    else: #to avoid division by 0 
        return np.eye(3)