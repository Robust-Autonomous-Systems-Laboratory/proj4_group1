from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, z):
        pass

    @abstractmethod
    def step(self, z, P=None):
        pass

class KalmanFilter(Filter):
    def __init__(self, F, H, Q, R, m):
        self.x = np.zeros(m)
        self.P = np.zeros((m,m))
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.m = m
        self.start = True
        return

    def predict(self):
        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T + self.Q
        return

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P@self.H.T@np.linalg.inv(S)
        y = z - self.H@self.x
        self.x = self.x + K@y
        self.P = (np.identity(self.m) - K@self.H)@self.P
        return self.x, self.P

    def step(self, z, P=None):
        if self.start:
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = P
            self.start = False
            return self.x, self.P
        self.predict()
        return self.update(z)

class ExtendedKalmanFilter(Filter):
    def __init__(self, f, F_jacobian, H_jacobian, Q, R, m):
        self.x = np.zeros(m)
        self.P = np.zeros((m,m))
        self.f = f
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.m = m
        self.start = True
        return

    def predict(self):
        self.x = self.f(self.x)
        F = self.F_jacobian(self.x)
        self.P = F@self.P@F.T + self.Q
        return

    def update(self, z):
        H = self.H_jacobian(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P@H.T@np.linalg.inv(S)
        y = z - H@self.x
        self.x = self.x + K@y
        self.P = (np.identity(self.m) - K@H)@self.P
        return self.x, self.P

    def step(self, z, P=None):
        if self.start:
            self.x = np.array([0, 0, z[1], 0, 0, 0])
            self.P = P
            self.start = False
            return self.x, self.P
        self.predict()
        return self.update(z)

class UnscentedKalmanFilter(Filter): 
    def __init__(self, dim_x, dim_z, f, h, Q, R, alpha=0.001, beta=2, kappa=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.start = True
        # Weights
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.wm = np.full(2 * self.dim_x + 1, 0.5 / (self.dim_x + self.lambda_))
        self.wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.wc = np.copy(self.wm)
        self.wc[0] += 1 - self.alpha**2 + self.beta


    def sigma_points(self, x, P):
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        # x[0] is the mean, the rest are the sigma points
        sigma_points[0] = x
        # Use Cholesky decomposition to compute the square root of the covariance matrix
        sqrt_P = np.linalg.cholesky((n + self.lambda_) * P)
        # Generate sigma points
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[:, i]
            sigma_points[n + i + 1] = x - sqrt_P[:, i]
        return sigma_points

    def predict(self):
        sigma_points = self.sigma_points(self.x, self.P)
        sigma_points_pred = np.array([self.f(sp) for sp in sigma_points])
        x_pred = np.mean(sigma_points_pred, axis=0)
        P_pred = self.Q.copy()
        for sp in sigma_points_pred:
            P_pred += np.outer(sp - x_pred, sp - x_pred)
        self.x = x_pred
        self.P = P_pred

    def update(self, z):
        sigma_points = self.sigma_points(self.x, self.P)
        sigma_points_meas = np.array([self.h(sp) for sp in sigma_points])
        z_pred = np.mean(sigma_points_meas, axis=0)
        P_zz = self.R.copy()
        P_xz = np.zeros((self.dim_x, self.dim_z))
        for sp, sp_meas in zip(sigma_points, sigma_points_meas):
            P_zz += np.outer(sp_meas - z_pred, sp_meas - z_pred)
            P_xz += np.outer(sp - self.x, sp_meas - z_pred)
        K = P_xz @ np.linalg.inv(P_zz)
        self.x += K @ (z - z_pred)
        self.P -= K @ P_zz @ K.T
        return self.x, self.P

    def step(self, z, P=None):
        if self.start:
            self.x = np.zeros(self.dim_x)
            self.P = P
            self.start = False
            return self.x, self.P
        self.predict()
        return self.update(z)

