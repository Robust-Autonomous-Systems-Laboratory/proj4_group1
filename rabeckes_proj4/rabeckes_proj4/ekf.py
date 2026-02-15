import numpy as np

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear systems.

    Models:
        x_k   = f(x_{k-1}, u_k) + w_k
        z_k   = h(x_k)          + v_k

    where:
        f : nonlinear state transition function
        h : nonlinear measurement function
        w_k ~ N(0, Q)
        v_k ~ N(0, R)
    """

    def __init__(self, f, h, F_jac, H_jac, Q, R, x0, P0):
        """
        Parameters
        ----------
        f : function
            Nonlinear state transition f(x, u)
        h : function
            Nonlinear measurement model h(x)
        F_jac : function
            Jacobian of f wrt state: F_jac(x, u)
        H_jac : function
            Jacobian of h wrt state: H_jac(x)
        Q : (n,n) ndarray
            Process noise covariance
        R : (k,k) ndarray
            Measurement noise covariance
        x0 : (n,) ndarray
            Initial state estimate
        P0 : (n,n) ndarray
            Initial covariance
        """
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac

        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)

        self.x = np.atleast_1d(x0).astype(float)
        self.P = np.atleast_2d(P0).astype(float)

        self.n = self.x.shape[0]

    def predict(self, u=None):
        """
        EKF prediction step.
        """
        # Nonlinear state prediction
        self.x = self.f(self.x, u)

        # Jacobian of f
        F = self.F_jac(self.x, u)

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        EKF update step.
        """
        z = np.atleast_1d(z)

        # Innovation
        y = z - self.h(self.x)

        # Jacobian of h
        H = self.H_jac(self.x)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x.copy(), self.P.copy()

    def step(self, z, u=None):
        """
        Predict + Update.
        """
        self.predict(u)
        return self.update(z)
