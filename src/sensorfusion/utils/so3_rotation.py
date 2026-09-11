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

def log(R):
    """
    Inverse of exp(): SO(3) --> logarithm map
    Calculates the axis-angle from the given rotation matrix"""
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    #small-angle approximation.
    if angle < 1e-8:
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / 2.0

    # Near-pi rotation: sin(angle) is close to zero 
    if np.pi - angle < 1e-6:
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        if B[0, 1] < 0:
            axis[1] = -axis[1]
        if B[0, 2] < 0:
            axis[2] = -axis[2]
        if axis[1] * axis[2] * B[1, 2] < 0:
            axis[2] = -axis[2]
        return axis * angle

    # General case: standard closed-form log map.
    vec = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return (angle / (2.0 * np.sin(angle))) * vec

def reorthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt
    return R_fixed