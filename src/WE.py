import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import math


class WE:
    def __init__(self, id0, id1, id2):
        self.id0 = id0
        self.id1 = id1
        self.id2 = id2
    def H(self, x, i):
        return x**i

    def dH(self, x,i):
        if i<1:
            return np.zeros_like(x)
        else:
            return i*x**(i-1)

    def d2H(self, x,i):
        if i<2:
            return np.zeros_like(x)
        else:
            return i*(i-1)*x**(i-2)

    def dXY(self, x, y, m):
        return np.sum(abs(x-y)**m, axis=0)

    def Hess_(self, x,Nm):
        return np.array([[self.H(x,i)*self.H(x,j) for i in range(1, Nm+1)] for j in range(1, Nm+1)])

    def L_(self, x,Nm):
        dh =  np.zeros((x.shape[1], Nm, Nm))
        for i in range(Nm):
            for j in range(Nm):
               dh[:, i, j] = self.dH(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i])  * self.H(x[2, :], self.id2[i]) \
                             *self.dH(x[0, :], self.id0[j]) * self.H(x[1, :], self.id1[j]) * self.H(x[2, :], self.id2[j])
               dh[:, i, j] += self.H(x[0, :], self.id0[i])  * self.dH(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i])\
                              *self.H(x[0, :], self.id0[j])  * self.dH(x[1, :], self.id1[j]) * self.H(x[2, :], self.id2[j])
               dh[:, i, j] += self.H(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i]) * self.dH(x[2, :], self.id2[i]) \
                              *self.H(x[0, :], self.id0[j]) * self.H(x[1, :], self.id1[j]) * self.dH(x[2, :], self.id2[j])
        return np.average(dh, axis=0)

    def pdhXY_(self, x, y, Nm, p):
        dh = np.zeros((x.shape[1], Nm))
        dxy = self.dXY(x, y, p-2)
        for i in range(Nm):
                dh[:, i] = self.dH(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i]) \
                              * p*(x[0,:]-y[0,:])*dxy #abs(x[0,:]-y[0,:])**(p-2)
                dh[:, i] += self.H(x[0, :], self.id0[i]) * self.dH(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i]) \
                              * p*(x[1,:]-y[1,:])*dxy #abs(x[1,:]-y[1,:])**(p-2)
                dh[:, i] += self.H(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i]) * self.dH(x[2, :], self.id2[i]) \
                            * p * (x[2, :] - y[2, :])*dxy ## * abs(x[2, :] - y[2, :]) ** (p - 2)
        return np.average(dh, axis=0)


    def moms(self, x,Nm):
      mom = np.zeros(Nm)
      for i in range(Nm):
          mom[i] = np.average(self.H(x[0,:], self.id0[i]) * self.H(x[1,:], self.id1[i]) * self.H(x[2,:], self.id2[i]))
      return mom

    def dH_(self, x, Nm):
        dh = np.zeros((3,x.shape[1], Nm))
        for i in range(Nm):
            dh[0, :, i] = self.dH(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i])  * self.H(x[2, :], self.id2[i])
            dh[1, :, i] =  self.H(x[0, :], self.id0[i]) * self.dH(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i])
            dh[2, :, i] =  self.H(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i])  * self.dH(x[2, :], self.id2[i])
        return dh

    def dmoms(self, x,Nm):
      mom = np.zeros((3,Nm))
      for i in range(Nm):
          mom[0, i] = np.average(self.dH(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i])  * self.H(x[2, :], self.id2[i]))
          mom[1, i] = np.average(self.H(x[0, :], self.id0[i])  * self.dH(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i]))
          mom[2, i] = np.average(self.H(x[0, :], self.id0[i])  * self.H(x[1, :], self.id1[i])  * self.dH(x[2, :], self.id2[i]))
      return mom

    def d2moms(self, x,Nm):
      Np = x.shape[1]
      mom = np.zeros((Np, Nm))
      for i in range(Nm):
          mom[:, i] =  self.d2H(x[0, :], self.id0[i]) * self.H(x[1, :], self.id1[i])   * self.H(x[2, :], self.id2[i])
          mom[:, i] += self.H(x[0, :], self.id0[i])   * self.d2H(x[1, :], self.id1[i]) * self.H(x[2, :], self.id2[i])
          mom[:, i] += self.H(x[0, :], self.id0[i])   * self.H(x[1, :], self.id1[i])   * self.d2H(x[2, :], self.id2[i])
      return np.average(mom, axis=0)

    def forward(self, Xs, Xt, id0, id1, id2, p=2, Nt=100, dt=1.e-9, dt0=1e-12, beta=1e-2):
      Nm = len(id0)
      X = Xs.T.copy()
      X0 = X.copy()
      Y = Xt.T.copy()
      Y0 = Y.copy()
      PXt = self.moms(Xt.T, Nm)
      PYt = self.moms(Xs.T, Nm)

      lx = np.zeros(Nm)
      wdist = [ np.average(np.linalg.norm(X-Y,axis=0) ) ]
      dtX = dt0
      dtY = dt0
      for i in range(Nt):
        tau = 10*dtX

        d2h = self.d2moms(X,Nm)
        P = self.moms(X, Nm)

        L = self.L_(X,Nm)
        pdhXY = self.pdhXY_(X, Y, Nm, p)
        b = (PXt-P)/tau - d2h + pdhXY/beta
        lx = np.linalg.solve(L, b)
        dxy = self.dXY(X, Y, p-2)
        A = np.sum(lx* self.dH_(X,Nm), axis=2) - p*(X-Y)*dxy/beta
        dW = np.random.normal(0., 1., X.shape)

        X = X + A*dtX + (2.*dtX)**0.5*dW
        dtX = min(dt, 0.01/np.max(abs(A)) )

        tau = 10*dtY
        d2h = self.d2moms(Y,Nm)
        L = self.L_(Y,Nm)
        P = self.moms(Y, Nm)
        pdhXY = self.pdhXY_(Y, X, Nm, p)
        b = (PYt-P)/tau - d2h + pdhXY/beta
        lx = np.linalg.solve(L, b)
        dxy = self.dXY(X, Y, p-2)
        A = np.sum(lx* self.dH_(Y,Nm), axis=2) - p*(Y-X)*dxy/beta
        dW = np.random.normal(0., 1., X.shape)

        Y = Y + A*dtY + (2.*dtY)**0.5*dW
        dtY = min(dt, 0.01/np.max(abs(A)) )

        dist = np.average(np.linalg.norm(X-Y,axis=0) )
        wdist.append( dist )
      return X, Y, wdist

