import numpy as np

class KalmanFilter:
    """
    Discrete-time linear Kalman Filter.

    Models:
        x_k   = A x_{k-1} + B u_k + w_k
        z_k   = H x_k     + v_k

    where:
        x_k : state vector
        u_k : control input
        z_k : measurement
        w_k ~ N(0, Q) process noise
        v_k ~ N(0, R) measurement noise
    """

    def __init__(self, A, B, H, Q, R, x0, P0):
        """
        Initialize the Kalman Filter.

        Parameters
        ----------
        A : (n, n) ndarray
            State transition matrix.
        B : (n, m) ndarray or None
            Control input matrix. Use None if no control.
        H : (k, n) ndarray
            Measurement matrix.
        Q : (n, n) ndarray
            Process noise covariance.
        R : (k, k) ndarray
            Measurement noise covariance.
        x0 : (n,) ndarray
            Initial state estimate.
        P0 : (n, n) ndarray
            Initial estimate covariance.
        """
        self.A = np.atleast_2d(A)
        self.B = None if B is None else np.atleast_2d(B)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)

        self.x = np.atleast_1d(x0).astype(float)
        self.P = np.atleast_2d(P0).astype(float)

        self.n = self.x.shape[0]

    def predict(self, u=None):
        """
        Time update (prediction step).

        Parameters
        ----------
        u : (m,) ndarray or None
            Control input. If None, no control is applied.

        Returns
        -------
        x_pred : (n,) ndarray
            Predicted state estimate.
        P_pred : (n, n) ndarray
            Predicted estimate covariance.
        """
        if u is not None and self.B is not None:
            u = np.atleast_1d(u)
            self.x = self.A @ self.x + self.B @ u
        else:
            self.x = self.A @ self.x

        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Measurement update (correction step).

        Parameters
        ----------
        z : (k,) ndarray
            Measurement vector.

        Returns
        -------
        x_upd : (n,) ndarray
            Updated state estimate.
        P_upd : (n, n) ndarray
            Updated estimate covariance.
        """
        z = np.atleast_1d(z)

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability, optional)
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy(), self.P.copy()

    def step(self, z, u=None):
        """
        Convenience method: predict then update in one call.

        Parameters
        ----------
        z : (k,) ndarray
            Measurement vector.
        u : (m,) ndarray or None
            Control input.

        Returns
        -------
        x : (n,) ndarray
            Posterior state estimate after prediction and update.
        P : (n, n) ndarray
            Posterior covariance.
        """
        self.predict(u=u)
        return self.update(z)

