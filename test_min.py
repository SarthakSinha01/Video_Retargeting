import cv2
import numpy as np
import pySaliencyMap
import matplotlib.pyplot as pyplot
import cvxopt
from qpsolvers import solve_qp
import quadprog
import qpoases
import casadi as cas

class pyARAP:
    def __init__(self,img,th,tw):
        self.img = img
        self.th = th
        self.tw = tw
        self.imgsize = img.shape
        self.W = self.imgsize[1]
        self.H = self.imgsize[0]

        self.M = 4
        self.N = 4
        self.Wr = (self.W)/self.tw
        self.Hr = (self.H)/self.th

        self.sm = pySaliencyMap.pySaliencyMap(self.W, self.H)
        self.saliency_map = self.sm.SMGetSM(self.img)

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

    def getVector_S(self):
        R_top = np.zeros(shape=((self.M * self.N), (self.M + self.N)))
        R_btm = np.zeros(shape=((self.M * self.N), (self.M + self.N)))
        v = np.zeros(shape=(self.M*self.N))

        grid_saliency = self.get_gridSM()
        #print("GRID SALIENCY")

        for k in range(0,(self.M*self.N)):
            for l in range(0,(self.M + self.N)):
                if l == (self.r(k)):
                    R_top[k][l] = (grid_saliency[(self.r(k))-1][(self.c(k))-1]) * (self.M/self.H)
                else:
                    R_top[k][l] = 0

        for k in range(0,(self.M*self.N)):
            for l in range(0,(self.M + self.N)):
                if l == (self.M + (self.c(k))):
                    R_btm[k][l] = (grid_saliency[(self.r(k))-1][(self.c(k))-1]) * (self.N/self.W)
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
        Gq = -1*G1


        h1 = np.zeros(shape=(self.M+self.N))
        hzero = h1
        for i in range(self.M):
            h1[i] = 0.1*(self.H/self.M)
            #h1[i] = (self.Hr/self.H)*(self.H/self.M)
        for i in range(self.M, (self.M + self.N)):
            h1[i] = 0.1*(self.W/self.N)
            #h1[i] = (self.Wr/self.W)*(self.W/self.N)

        h1 = np.array([h1])
        ht = -1*(h1.T)
        hq = h1.T

        a_tmp = np.zeros(shape=((self.M + self.N -2),(self.M+self.N)))
        a_row = np.zeros(shape=(self.M + self.N))
        a_col = np.ones(shape=(self.M + self.N))
        for i in range((int)((self.M+self.N)/2)):
            a_row[i] = 1
            a_col[i] = 0
        a = np.vstack((a_row,a_col))

        aq = np.vstack((a,a_tmp))


        bq = np.zeros(shape=(self.M+self.N))
        bq[0] = self.Hr
        bq[1] = self.Wr

        b1 = np.array([self.Hr,self.Wr])
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
        S = np.array(cvxopt.matrix(S1))

        E1 = sol['primal objective']

        hq = hq.reshape((8,))
        bq = bq.reshape((8,))

        #C = np.vstack([Gq,aq])
        #C = aq
        C = a

        lbh = np.zeros(shape=(self.M+self.N))
        for i in range(self.M):
            lbh[i] = 0.1 * (self.H/self.M)
        for i in range(self.M, (self.M+self.N)):
            lbh[i] = 0.1 * (self.W/self.N)


        bC = 10*(np.ones(shape = (self.M+self.N)))
        bC[0] = self.Hr/self.M
        bC[1] = self.Wr/self.N

        lbC = np.array([self.Hr, self.Wr])
        ubC = lbC

        lb  = lbh.T
        ub = hq

        itr = np.array([50])

        options = qpoases.PyOptions()
        qp = qpoases.PyQProblem(Q.shape[0],C.shape[0])
        qp.setOptions(options)


        #print("P", Q)
        #print("q", q1)
        #print("C", C)
        #print("lbC", lbC)
        #print("ubC", ubC)
        #print("lb", lb)
        #print("ub", ub)
        R = qp.init(Q,q1,C,lb,ub,lbC,ubC,itr)

        x_opt = np.ones(shape=(Q.shape[0],))
        ret = qp.getPrimalSolution(x_opt)
        objVal = qp.getObjVal()
        vector = np.array(x_opt)
        print(vector)
        print("############################")
        #print(vector[0],vector[1], vector[2], vector[3])
        #print(vector[4],vector[5], vector[6], vector[7])
        #print("************************")


        #print(S)
        print("############################")
        #print(S[0][0], S[1][0], S[2][0], S[3][0])
        #print(S[4][0], S[5][0], S[6][0], S[7][0])

        print(S)
        print(E1)
        return(S)
