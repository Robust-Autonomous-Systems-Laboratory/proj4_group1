import numpy as np

class KalmanFilter:
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
