import cv2
import numpy as np
import pySaliencyMap
import matplotlib.pyplot as pyplot
import cvxopt
import pyFrameSTemporal
import pyARAP
import time

class pyTotalARAP:
    def __init__(self,img,tgImg, th,tw):
        self.img = img
        self.tgImg = tgImg
        self.th = th
        self.tw = tw
        self.imgsize = img.shape
        self.W = self.imgsize[1]
        self.H = self.imgsize[0]

        self.M = 4
        self.N = 4
        self.Wr = (self.W)/(self.W/(self.W-self.tw))
        self.Hr = (self.H)/(self.H/(self.H-self.th))

        self.sm = pySaliencyMap.pySaliencyMap(self.W, self.H)
        self.saliency_map = self.sm.SMGetSM(self.img)

        self.obj = pyFrameSTemporal.temporal_S(self.img,self.tgImg,self.th,self.tw)
        self.STemp = self.obj.Frame_S_temporal(self.img, self.tgImg)


    # this calclutes importance value of a grid by averaging the saliency values of pixel in it.
    def avg_saliency(self,i,j):
        sum = 0
        tot_pixels = 0
        rb = (int)(self.H/self.M)
        cb = (int)(self.W/self.N)

        for k in range((i*rb),(i+1)*rb):
            for l in range((j*cb), ((j+1)*cb)):
                sum = sum + self.saliency_map[k][l]
                tot_pixels = tot_pixels+1

        avg = sum/tot_pixels
        return avg


    # returns the grid saliency_map.
    def get_gridSM(self):
        grid_saliency = np.zeros(shape=(self.M,self.N))
        for i in range(0,self.M):
            for j in range(0,self.N):
                grid_saliency[i][j] = self.avg_saliency(i,j)
        #print(grid_saliency)
        return grid_saliency


    def r(self,k):
        a = int(k/self.N)
        return(a+1)
    def c(self,k):
        a = ((k-1)%self.N)+1
        return a



    # this function minimizes the total energy function and return the solution to get vector S along with the time taken to compute it.
    def getVector_Stotal(self):
        start_time = time.process_time()

        R_top = np.zeros(shape=((self.M * self.N), (self.M + self.N)))
        R_btm = np.zeros(shape=((self.M * self.N), (self.M + self.N)))
        v = np.zeros(shape=(self.M*self.N))

        grid_saliency = self.get_gridSM()
        S_temp = self.STemp


        for k in range(0,(self.M*self.N)):
            for l in range(0,(self.M + self.N)):
                if l == (self.r(k)):
                    R_top[k][l] = (grid_saliency[(self.r(k))-1][(self.c(k))-1]) * ((self.M/self.H) + (1/S_temp[self.r(k)-1][self.c(k)-1][0]))
                else:
                    R_top[k][l] = 0

        for k in range(0,(self.M*self.N)):
            for l in range(0,(self.M + self.N)):
                if l == (self.M + (self.c(k))):
                    R_btm[k][l] = (grid_saliency[(self.r(k))-1][(self.c(k))-1]) * ((self.N/self.W) + (1/S_temp[self.r(k)-1][self.c(k)-1][1]))
                else:
                    R_btm[k][l] = 0

        for k in range(0,(self.M * self.N)):
            v[k] = grid_saliency[(self.r(k))-1][(self.c(k))-1]


        Q1 = np.concatenate((R_top, R_btm),axis = 0)
        Qt = Q1.transpose()
        Q = np.matmul(Qt,Q1)

        b1 = np.concatenate((R_top, R_btm),axis = 0)
        bt = b1.transpose()
        v1 = np.concatenate((v,v),axis = 0)
        b = -2*(np.matmul(bt,v1))

        G1 = np.matrix(np.identity(self.M + self.N))
        G1 = -1*G1

        h1 = np.zeros(shape=(self.M+self.N))
        for i in range((int)((self.M + self.N)/2)):
            h1[i] = 0.5*(self.H/self.M)

        for i in range((int)((self.M+self.N)/2), (int)(self.M + self.N)):
            h1[i] = 0.3*(self.W/self.N)


        h1 = np.array([h1])
        ht = -1*(h1.T)

        a_row = np.zeros(shape=(self.M + self.N))
        a_col = np.ones(shape=(self.M + self.N))
        for i in range((int)((self.M+self.N)/2)):
            a_row[i] = 1
            a_col[i] = 0
        a = np.vstack((a_row,a_col))

        b1 = np.array([self.Hr, self.Wr])
        b1 = np.array([b1])
        bt = b1.T

        q1 = b.transpose()

        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(q1)
        G = cvxopt.matrix(G1)
        h = cvxopt.matrix(ht)
        A = cvxopt.matrix(a)
        b = cvxopt.matrix(bt)

        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        S1 = sol['x']
        St = np.array(cvxopt.matrix(S1))
        E1 = sol['primal objective']
        #print("#####################")
        #print(St)
        end_time = time.process_time()
        time1 = end_time - start_time

        return time1, St
